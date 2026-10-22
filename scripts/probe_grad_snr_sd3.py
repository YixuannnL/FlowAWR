# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Offline gradient-SNR probe for FlowAWR on SD3 (PickScore).

Derived from ``scripts/train_awr_sd3.py`` (repo main @ bfcc6eb; the copied helpers are unchanged since 19cc86f). The pipeline
loading, LoRA construction, prompt sampler, sampling call, reward call and the
energy-mode loss are copied from that file so that the probe sees exactly what
training sees. The differences are intentional and limited to:

* no optimizer, no EMA, no "old"/"ref" adapters, no checkpoint or swanlab writes;
* one (or a few) sampling rounds from a fixed LoRA checkpoint;
* for every sample we back-propagate the normalised reconstruction loss once
  and store a count-sketch (random-sign hashing) of the LoRA gradient.

Why one backward per sample is enough
-------------------------------------
With the FlowAWR energy loss and ``pred_x0_old == x0_pred.detach()`` (the only
option at a frozen checkpoint, and within 2e-5 of training where old is a
half-step Polyak average), the gradient of sample *i* is exactly
``e_i * b_i`` with ``b_i = grad of sum_t mean((x0_pred - x0)^2) / w_t``.
Any weighting scheme's full-batch gradient is therefore ``sum_i e_i b_i`` and
can be evaluated offline from the per-sample sketches by
``scripts/analyze_grad_snr.py`` without touching a GPU again.

``--exact_check`` additionally runs the real training loss path with the real
``PerPromptStatTracker`` energies on this rank's samples and stores the sketch
of that accumulated gradient, so the linearity claim and the sketch pipeline
can be verified numerically by the analysis script.

Launch (same environment as training, from the repo root)::

    torchrun --nproc_per_node=8 scripts/probe_grad_snr_sd3.py \
        --config config/awr.py:sd3_pickscore \
        --lora_path /path/to/checkpoint-900/lora \
        --out_dir probe_runs/ckpt900 \
        --num_rounds 2

Nothing in this script writes outside ``--out_dir``.
"""

import datetime
import json
import logging
import os
import random
import time
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
from absl import app, flags
from diffusers import StableDiffusion3Pipeline
from ml_collections import config_flags
from peft import LoraConfig, get_peft_model
from torch.amp import autocast
from torch.utils.data import Dataset, Sampler

import flow_grpo.rewards
from flow_grpo.diffusers_patch.pipeline_with_logprob import pipeline_with_logprob
from flow_grpo.diffusers_patch.train_dreambooth_lora_sd3 import encode_prompt
from flow_grpo.stat_tracking import PerPromptStatTracker

torch_autocast = partial(autocast, "cuda")

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration (same file as training).")
flags.DEFINE_string("lora_path", "", "PEFT LoRA directory (checkpoint-N/lora). Empty = freshly initialised LoRA (base model).")
flags.DEFINE_string("out_dir", "", "Output directory for sketches and metadata (required).")
flags.DEFINE_integer("num_rounds", 2, "Sampling rounds; each round is one training epoch worth of samples.")
flags.DEFINE_integer("samples_per_round", 1152, "Samples per round (48 groups x 24 images in the official config).")
flags.DEFINE_integer("sample_epoch", 900, "Sampler epoch offset, picks which prompts are drawn (any integer).")
flags.DEFINE_integer("sketch_dim", 65536, "Count-sketch dimension K.")
flags.DEFINE_integer("sketch_seed", 0, "Seed of the hashing maps (must be shared by all ranks; it is).")
flags.DEFINE_integer("noise_seed", 1234, "Seed for the per-sample diffusion noise draws.")
flags.DEFINE_float("loss_scale", 1024.0, "Static loss scale for fp16 autocast backward (divided out afterwards).")
flags.DEFINE_bool("exact_check", True, "Also run the real loss path with real energies and store its gradient sketch.")
flags.DEFINE_integer("max_samples_per_rank", 0, "Debug: only back-propagate this many samples per rank (0 = all).")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


# ---------------------------------------------------------------------------
# Copied verbatim from scripts/train_awr_sd3.py
# ---------------------------------------------------------------------------
def setup_distributed(rank, lock_rank, world_size):
    os.environ["MASTER_ADDR"] = os.getenv("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "12355")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(lock_rank)


def cleanup_distributed():
    dist.destroy_process_group()


def is_main_process(rank):
    return rank == 0


def set_seed(seed: int, rank: int = 0):
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)


class TextPromptDataset(Dataset):
    def __init__(self, dataset, split="train"):
        self.file_path = os.path.join(dataset, f"{split}.txt")
        with open(self.file_path, "r") as f:
            self.prompts = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": {}}


class GenevalPromptDataset(Dataset):
    def __init__(self, dataset, split="train"):
        self.file_path = os.path.join(dataset, f"{split}_metadata.jsonl")
        with open(self.file_path, "r", encoding="utf-8") as f:
            self.metadatas = [json.loads(line) for line in f]
            self.prompts = [item["prompt"] for item in self.metadatas]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": self.metadatas[idx]}


class DistributedKRepeatSampler(Sampler):
    def __init__(self, dataset, batch_size, k, num_replicas, rank, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.k = k
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed

        self.total_samples = self.num_replicas * self.batch_size
        assert (
            self.total_samples % self.k == 0
        ), f"k can not div n*b, k{k}-num_replicas{num_replicas}-batch_size{batch_size}"
        self.m = self.total_samples // self.k
        self.epoch = 0

    def __iter__(self):
        while True:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g)[: self.m].tolist()
            repeated_indices = [idx for idx in indices for _ in range(self.k)]

            shuffled_indices = torch.randperm(len(repeated_indices), generator=g).tolist()
            shuffled_samples = [repeated_indices[i] for i in shuffled_indices]

            per_card_samples = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                end = start + self.batch_size
                per_card_samples.append(shuffled_samples[start:end])
            yield per_card_samples[self.rank]

    def set_epoch(self, epoch):
        self.epoch = epoch


def gather_tensor_to_all(tensor, world_size):
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, tensor)
    return torch.cat(gathered_tensors, dim=0).cpu()


def compute_text_embeddings(prompt, text_encoders, tokenizers, max_sequence_length, device):
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = encode_prompt(text_encoders, tokenizers, prompt, max_sequence_length)
        prompt_embeds = prompt_embeds.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
    return prompt_embeds, pooled_prompt_embeds


# ---------------------------------------------------------------------------
# Probe-specific helpers
# ---------------------------------------------------------------------------
def choose_sampling_layout(samples_per_round, world_size, k, preferred_bsz):
    """Per-GPU sampling batch size and batch count satisfying the sampler's constraints.

    Mirrors the search in config/awr.py:_get_config but for the actual world size,
    so the probe can run on 1/4/8/16 GPUs. Prefers the training batch size.
    """
    candidates = list(range(preferred_bsz, 0, -1)) + list(range(preferred_bsz + 1, k + 1))
    for bsz in candidates:
        if (world_size * bsz) % k == 0 and samples_per_round % (world_size * bsz) == 0:
            return bsz, samples_per_round // (world_size * bsz)
    raise ValueError(f"no sampling batch size for samples_per_round={samples_per_round}, world={world_size}, k={k}")


def build_sketch_maps(params, sketch_dim, seed, device):
    """Deterministic count-sketch maps, identical on every rank/run for a given seed."""
    maps = []
    total = 0
    for i, p in enumerate(params):
        n = p.numel()
        g = torch.Generator().manual_seed(seed * 1_000_003 + i)
        idx = torch.randint(0, sketch_dim, (n,), generator=g, dtype=torch.int64)
        sign = torch.randint(0, 2, (n,), generator=g, dtype=torch.int8) * 2 - 1
        maps.append((idx.to(device), sign.to(device=device, dtype=torch.float32)))
        total += n
    return maps, total


def sketch_current_grads(params, maps, sketch_dim, device, scale):
    """Count-sketch of the concatenated .grad of ``params`` (divided by ``scale``)."""
    out = torch.zeros(sketch_dim, dtype=torch.float32, device=device)
    sq = torch.zeros((), dtype=torch.float64, device=device)
    finite = True
    for p, (idx, sign) in zip(params, maps):
        if p.grad is None:
            continue
        g = p.grad.detach().flatten().float() / scale
        if not bool(torch.isfinite(g).all()):
            finite = False
        out.index_add_(0, idx, g * sign)
        sq += (g.double() ** 2).sum()
    return out, float(sq.item()), finite


def energy_loss_for_sample(transformer, x0, timesteps, prompt_embed, pooled_embed, noise, energy, enable_amp, amp_dtype):
    """Sum over train timesteps of the FlowAWR energy-mode loss for ONE sample.

    This is the loss block of scripts/train_awr_sd3.py (energy branch) with the
    batch dimension used for the timesteps instead of for different samples, and
    with ``old_prediction`` replaced by the detached current prediction.
    ``energy=None`` means e = 1, i.e. target_x0 = x0 (the per-sample basis gradient).
    """
    n_t = timesteps.shape[0]
    x0_rep = x0.expand(n_t, -1, -1, -1).float()
    t_expanded = (timesteps / 1000.0).view(-1, 1, 1, 1).float()
    xt = (1 - t_expanded) * x0_rep + t_expanded * noise
    embeds = prompt_embed.expand(n_t, -1, -1)
    pooled = pooled_embed.expand(n_t, -1)

    with torch_autocast(enabled=enable_amp, dtype=amp_dtype):
        forward_prediction = transformer(
            hidden_states=xt,
            timestep=timesteps,
            encoder_hidden_states=embeds,
            pooled_projections=pooled,
            return_dict=False,
        )[0]
        x0_prediction = xt - t_expanded * forward_prediction
        with torch.no_grad():
            weight_factor = (
                torch.abs(x0_prediction.double() - x0_rep.double())
                .mean(dim=tuple(range(1, x0_rep.ndim)), keepdim=True)
                .clip(min=0.00001)
            )
        if energy is None:
            target_x0 = x0_rep
        else:
            pred_x0_old = x0_prediction.detach()
            target_x0 = pred_x0_old + float(energy) * (x0_rep - pred_x0_old)
        ori_policy_loss = ((x0_prediction - target_x0) ** 2 / weight_factor).mean(dim=tuple(range(1, x0_rep.ndim)))
    return ori_policy_loss.sum()


def make_noise(shape, seed, device):
    g = torch.Generator(device=device).manual_seed(int(seed))
    return torch.randn(shape, generator=g, device=device, dtype=torch.float32)


def main(_):
    config = FLAGS.config
    assert FLAGS.out_dir, "--out_dir is required"
    assert config.use_lora, "the probe sketches LoRA gradients; config.use_lora must be True"
    assert config.train.energy_mode, "probe assumes the FlowAWR energy-mode loss"

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    setup_distributed(rank, local_rank, world_size)
    device = torch.device(f"cuda:{local_rank}")
    set_seed(config.seed, rank)

    os.makedirs(FLAGS.out_dir, exist_ok=True)
    logger.info(f"rank {rank}/{world_size} probe -> {FLAGS.out_dir}")

    mixed_precision_dtype = None
    if config.mixed_precision == "fp16":
        mixed_precision_dtype = torch.float16
    elif config.mixed_precision == "bf16":
        mixed_precision_dtype = torch.bfloat16
    enable_amp = mixed_precision_dtype is not None

    # --- Load pipeline and models (as in training) ---
    pipeline = StableDiffusion3Pipeline.from_pretrained(config.pretrained.model)
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.text_encoder_3.requires_grad_(False)
    pipeline.transformer.requires_grad_(False)
    text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3]
    tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2, pipeline.tokenizer_3]
    pipeline.safety_checker = None
    pipeline.set_progress_bar_config(position=1, disable=not is_main_process(rank), leave=False, desc="Timestep")

    text_encoder_dtype = mixed_precision_dtype if enable_amp else torch.float32
    pipeline.vae.to(device, dtype=torch.float32)
    pipeline.text_encoder.to(device, dtype=text_encoder_dtype)
    pipeline.text_encoder_2.to(device, dtype=text_encoder_dtype)
    pipeline.text_encoder_3.to(device, dtype=text_encoder_dtype)
    transformer = pipeline.transformer.to(device)

    target_modules = [
        "attn.add_k_proj",
        "attn.add_q_proj",
        "attn.add_v_proj",
        "attn.to_add_out",
        "attn.to_k",
        "attn.to_out.0",
        "attn.to_q",
        "attn.to_v",
    ]
    transformer_lora_config = LoraConfig(r=32, lora_alpha=64, init_lora_weights="gaussian", target_modules=target_modules)
    transformer = get_peft_model(transformer, transformer_lora_config)
    if FLAGS.lora_path:
        # Same call the training script uses when resuming (adapter "default" already exists).
        transformer.load_adapter(FLAGS.lora_path, adapter_name="default", is_trainable=True)
    transformer.set_adapter("default")
    lora_source = FLAGS.lora_path if FLAGS.lora_path else "fresh-gaussian-init (B=0, equals base model)"

    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    named_params = [(n, p) for n, p in transformer.named_parameters() if p.requires_grad]
    params = [p for _, p in named_params]
    sketch_maps, n_params = build_sketch_maps(params, FLAGS.sketch_dim, FLAGS.sketch_seed, device)
    logger.info(f"rank {rank}: {len(params)} trainable LoRA tensors, {n_params} parameters, sketch K={FLAGS.sketch_dim}")

    # --- Dataset and sampler (as in training) ---
    if config.prompt_fn == "general_ocr":
        train_dataset = TextPromptDataset(config.dataset, "train")
    elif config.prompt_fn == "geneval":
        train_dataset = GenevalPromptDataset(config.dataset, "train")
    else:
        raise NotImplementedError("Prompt function not supported with dataset")

    k = config.sample.num_image_per_prompt
    sample_bsz, num_batches = choose_sampling_layout(
        FLAGS.samples_per_round, world_size, k, config.sample.train_batch_size
    )
    samples_per_rank_round = sample_bsz * num_batches
    logger.info(f"sampling layout: {world_size} ranks x {sample_bsz} per batch x {num_batches} batches = {FLAGS.samples_per_round}/round")

    train_sampler = DistributedKRepeatSampler(
        dataset=train_dataset, batch_size=sample_bsz, k=k, num_replicas=world_size, rank=rank, seed=config.seed
    )
    sampler_iter = iter(train_sampler)

    neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings(
        [""], text_encoders, tokenizers, max_sequence_length=128, device=device
    )
    sample_neg_prompt_embeds = neg_prompt_embed.repeat(sample_bsz, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(sample_bsz, 1)

    reward_fn = getattr(flow_grpo.rewards, "multi_score")(device, config.reward_fn)
    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)

    # --- Sampling rounds ---
    records = []  # one dict per sample on this rank
    sample_tensors = []  # (x0 [C,H,W], prompt_embed [L,D], pooled [D])
    scheduler_timesteps = None
    t_sample0 = time.time()
    for rnd in range(FLAGS.num_rounds):
        pipeline.transformer.eval()
        for i in range(num_batches):
            train_sampler.set_epoch((FLAGS.sample_epoch + rnd) * num_batches + i)
            batch_idx = next(sampler_iter)
            items = [train_dataset[j] for j in batch_idx]
            prompts = [it["prompt"] for it in items]
            prompt_metadata = [it["metadata"] for it in items]

            prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
                prompts, text_encoders, tokenizers, max_sequence_length=128, device=device
            )
            prompt_ids = tokenizers[0](
                prompts, padding="max_length", max_length=256, truncation=True, return_tensors="pt"
            ).input_ids.to(device)

            with torch_autocast(enabled=enable_amp, dtype=mixed_precision_dtype):
                with torch.no_grad():
                    images, latents, _ = pipeline_with_logprob(
                        pipeline,
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        negative_prompt_embeds=sample_neg_prompt_embeds[: len(prompts)],
                        negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds[: len(prompts)],
                        num_inference_steps=config.sample.num_steps,
                        guidance_scale=config.sample.guidance_scale,
                        output_type="pt",
                        height=config.resolution,
                        width=config.resolution,
                        noise_level=config.sample.noise_level,
                        deterministic=config.sample.deterministic,
                        solver=config.sample.solver,
                        model_type="sd3",
                    )
            latents = torch.stack(latents, dim=1)
            x0_batch = latents[:, -1].detach().float()
            if scheduler_timesteps is None:
                scheduler_timesteps = pipeline.scheduler.timesteps.detach().clone().to(device)

            rewards, _ = reward_fn(images, prompts, prompt_metadata, only_strict=True)
            rewards = {kk: torch.as_tensor(v, device=device).float() for kk, v in rewards.items()}

            for b in range(len(prompts)):
                local_index = len(records)
                records.append(
                    {
                        "round": rnd,
                        "rank": rank,
                        "local_index": local_index,
                        "prompt_idx": int(batch_idx[b]),
                        "prompt": prompts[b],
                        "prompt_ids": prompt_ids[b].detach().cpu(),
                        "reward_avg": float(rewards["avg"][b]),
                        "rewards": {kk: float(v[b]) for kk, v in rewards.items()},
                    }
                )
                sample_tensors.append((x0_batch[b], prompt_embeds[b].detach(), pooled_prompt_embeds[b].detach()))
            del images, latents
        if is_main_process(rank):
            logger.info(f"round {rnd} sampled ({time.time() - t_sample0:.1f}s so far)")

    train_timesteps = scheduler_timesteps[:num_train_timesteps]
    n_local = len(records)
    assert n_local == samples_per_rank_round * FLAGS.num_rounds

    # --- Real energies via the real tracker (needs all ranks' rewards per round) ---
    energies_baseline = np.full(n_local, np.nan, dtype=np.float64)
    advantages_baseline = np.full(n_local, np.nan, dtype=np.float64)
    for rnd in range(FLAGS.num_rounds):
        sel = [r for r in records if r["round"] == rnd]
        rew = torch.tensor([r["reward_avg"] for r in sel], device=device, dtype=torch.float32)
        pid = torch.stack([r["prompt_ids"] for r in sel]).to(device)
        rew_all = gather_tensor_to_all(rew, world_size).numpy()
        pid_all = gather_tensor_to_all(pid, world_size)
        prompts_all_decoded = tokenizers[0].batch_decode(pid_all.cpu().numpy(), skip_special_tokens=True)
        tracker = PerPromptStatTracker(
            config.sample.global_std, adv_clip_max=config.train.adv_clip_max, temperature=config.train.temperature
        )
        adv_all, en_all, _ = tracker.update(
            prompts_all_decoded, rew_all, exp=config.train.energy_mode, hard_gating=config.train.hard_gating
        )
        per_rank = len(sel)
        en_rank = en_all.reshape(world_size, per_rank)[rank]
        adv_rank = adv_all.reshape(world_size, per_rank)[rank]
        for j, r in enumerate(sel):
            energies_baseline[r["local_index"]] = en_rank[j]
            advantages_baseline[r["local_index"]] = adv_rank[j]

    # --- Per-sample basis gradients -> sketches ---
    transformer.train()
    for p in params:
        p.grad = None
    n_do = n_local if FLAGS.max_samples_per_rank <= 0 else min(n_local, FLAGS.max_samples_per_rank)
    sketches = np.zeros((n_do, FLAGS.sketch_dim), dtype=np.float32)
    grad_sqnorm = np.zeros(n_do, dtype=np.float64)
    finite_flags = np.ones(n_do, dtype=bool)
    loss_values = np.zeros(n_do, dtype=np.float64)
    noise_shape = (num_train_timesteps,) + tuple(sample_tensors[0][0].shape)

    def noise_seed_for(rec):
        return FLAGS.noise_seed + ((rec["round"] * world_size + rec["rank"]) * samples_per_rank_round + rec["local_index"])

    scale_used = np.full(n_do, FLAGS.loss_scale, dtype=np.float64)

    t_grad0 = time.time()
    for s in range(n_do):
        rec = records[s]
        x0, pe, ppe = sample_tensors[s]
        scale = FLAGS.loss_scale
        for _attempt in range(4):  # static loss scale with overflow retry (fp16 autocast)
            noise = make_noise(noise_shape, noise_seed_for(rec), device)
            loss = energy_loss_for_sample(
                transformer, x0.unsqueeze(0), train_timesteps, pe, ppe, noise, None, enable_amp, mixed_precision_dtype
            )
            (loss * scale).backward()
            sk, sq, ok = sketch_current_grads(params, sketch_maps, FLAGS.sketch_dim, device, scale)
            for p in params:
                p.grad = None
            if ok or scale <= 1.0:
                break
            scale = scale / 8.0
        sketches[s] = sk.cpu().numpy()
        grad_sqnorm[s] = sq
        finite_flags[s] = ok
        loss_values[s] = float(loss.detach())
        scale_used[s] = scale
        if is_main_process(rank) and (s + 1) % 16 == 0:
            logger.info(f"basis grads {s + 1}/{n_do} ({time.time() - t_grad0:.1f}s)")

    # --- Exact check: real loss path with real energies, accumulated over this rank ---
    exact_sketch = None
    exact_sqnorm = float("nan")
    exact_finite = True
    exact_scale = float(scale_used.min())
    if FLAGS.exact_check:
        t_ex0 = time.time()
        for s in range(n_do):
            rec = records[s]
            x0, pe, ppe = sample_tensors[s]
            noise = make_noise(noise_shape, noise_seed_for(rec), device)
            loss = energy_loss_for_sample(
                transformer, x0.unsqueeze(0), train_timesteps, pe, ppe, noise,
                energies_baseline[s], enable_amp, mixed_precision_dtype,
            )
            (loss * exact_scale).backward()
        sk, sq, ok = sketch_current_grads(params, sketch_maps, FLAGS.sketch_dim, device, exact_scale)
        exact_sketch = sk.cpu().numpy()
        exact_sqnorm, exact_finite = sq, ok
        for p in params:
            p.grad = None
        if is_main_process(rank):
            logger.info(f"exact-check gradient done ({time.time() - t_ex0:.1f}s)")

    # --- Save per-rank outputs ---
    npz_path = os.path.join(FLAGS.out_dir, f"sketch_rank{rank:02d}.npz")
    payload = dict(
        sketch=sketches,
        grad_sqnorm=grad_sqnorm,
        finite=finite_flags,
        basis_loss=loss_values,
        loss_scale_used=scale_used,
        round=np.array([r["round"] for r in records[:n_do]], dtype=np.int64),
        rank=np.full(n_do, rank, dtype=np.int64),
        local_index=np.array([r["local_index"] for r in records[:n_do]], dtype=np.int64),
        prompt_idx=np.array([r["prompt_idx"] for r in records[:n_do]], dtype=np.int64),
        reward_avg=np.array([r["reward_avg"] for r in records[:n_do]], dtype=np.float64),
        energy_baseline=energies_baseline[:n_do],
        advantage_baseline=advantages_baseline[:n_do],
    )
    for key in records[0]["rewards"]:
        payload[f"reward_{key}"] = np.array([r["rewards"][key] for r in records[:n_do]], dtype=np.float64)
    if exact_sketch is not None:
        payload["exact_sketch"] = exact_sketch
        payload["exact_sqnorm"] = np.array(exact_sqnorm)
        payload["exact_finite"] = np.array(exact_finite)
        payload["exact_loss_scale"] = np.array(exact_scale)
    np.savez(npz_path, **payload)
    with open(os.path.join(FLAGS.out_dir, f"prompts_rank{rank:02d}.json"), "w", encoding="utf-8") as f:
        json.dump([{"local_index": r["local_index"], "prompt_idx": r["prompt_idx"], "prompt": r["prompt"]} for r in records[:n_do]], f, ensure_ascii=False)

    if is_main_process(rank):
        meta = {
            "created_utc": datetime.datetime.utcnow().isoformat() + "Z",
            "lora_source": lora_source,
            "pretrained_model": config.pretrained.model,
            "config_run_name": str(config.run_name),
            "dataset": str(config.dataset),
            "world_size": world_size,
            "num_rounds": FLAGS.num_rounds,
            "samples_per_round": FLAGS.samples_per_round,
            "samples_per_rank_round": samples_per_rank_round,
            "sample_epoch": FLAGS.sample_epoch,
            "group_size_k": k,
            "sketch_dim": FLAGS.sketch_dim,
            "sketch_seed": FLAGS.sketch_seed,
            "noise_seed": FLAGS.noise_seed,
            "loss_scale": FLAGS.loss_scale,
            "num_lora_params": n_params,
            "lora_param_names": [n for n, _ in named_params],
            "train_timesteps": [float(t) for t in train_timesteps.cpu().tolist()],
            "num_train_timesteps": num_train_timesteps,
            "mixed_precision": config.mixed_precision,
            "sampler": {
                "num_steps": config.sample.num_steps,
                "guidance_scale": config.sample.guidance_scale,
                "noise_level": config.sample.noise_level,
                "deterministic": bool(config.sample.deterministic),
                "solver": config.sample.solver,
                "resolution": config.resolution,
            },
            "tracker": {
                "global_std": bool(config.sample.global_std),
                "adv_clip_max": config.train.adv_clip_max,
                "temperature": config.train.temperature,
                "hard_gating": bool(config.train.hard_gating),
                "energy_clip": [-3.0, 3.0],
            },
            "reward_fn": {kk: float(v) for kk, v in config.reward_fn.items()},
            "exact_check": bool(FLAGS.exact_check),
            "torch_version": torch.__version__,
            "note": "old_prediction := x0_pred.detach(); no optimizer step performed; gradients are of the policy loss only (no KL term).",
        }
        with open(os.path.join(FLAGS.out_dir, "probe_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        logger.info(f"wrote {npz_path} and probe_meta.json; total {time.time() - t_sample0:.1f}s")

    dist.barrier()
    cleanup_distributed()


if __name__ == "__main__":
    app.run(main)
