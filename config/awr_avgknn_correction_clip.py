"""A4.22: per-group/per-timestep AvgKNN correction clipping on A4.20.1.

Source: Fuzm7/FlowAWR @ 5ae282ef34fd664d9828fe62817f1960534f1cce.
Use scripts/train_awr_sd3_avgknn_correction_clip.py; override settings via
--config.train.<field>=<value>, as with the official config.
"""

import importlib.util
import os

base_path = os.path.join(os.path.dirname(__file__), "base.py")
spec = importlib.util.spec_from_file_location("base", base_path)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

def get_config(name):
    return globals()[name]()


def _get_config(base_model="sd3", n_gpus=1, gradient_step_per_epoch=1, dataset="pickscore", reward_fn={}, name="", train_batch_size=None):
    config = base.get_config()
    assert base_model in ["sd3", "sd3m", "flux"]
    assert dataset in ["pickscore", "ocr", "geneval", "hpsv2"]

    config.base_model = base_model
    config.dataset = os.path.join(os.getcwd(), f"dataset/{dataset}")
    if base_model == "sd3":
        config.pretrained.model = os.environ.get("SD3_MODEL", "stabilityai/stable-diffusion-3.5-medium")
        config.sample.num_steps = 10
        config.sample.eval_num_steps = 40
        config.sample.guidance_scale = 4.5
        config.resolution = 512
        config.train.beta = 0.0001
        config.sample.noise_level = 0.7
        bsz = 9
    elif base_model == "sd3m":
        # SD3 Medium only ships fp16-variant weight files, hence pretrained.variant.
        config.pretrained.model = os.environ.get("SD3_MEDIUM_MODEL", "stabilityai/stable-diffusion-3-medium-diffusers")
        config.pretrained.variant = "fp16"
        config.sample.num_steps = 10
        config.sample.eval_num_steps = 28
        config.sample.guidance_scale = 7.0
        # SD3 Medium is trained at 1024 (transformer sample_size=128).
        config.resolution = 1024
        config.train.beta = 0.0001
        config.sample.noise_level = 0.7
        # 1024 quadruples the joint-attention sequence length (1152 -> 4224) relative to 512.
        # Feasible batch sizes under the divisibility constraints below are {12, 9, 6, 3}.
        # Total samples per epoch remain num_groups * k = 1152.
        bsz = 9
    elif base_model == "flux":
        config.pretrained.model = os.environ.get("FLUX_MODEL", "black-forest-labs/FLUX.1-dev")
        config.mixed_precision = "bf16"
        # Aligned with TempFlow-GRPO config/dgx.py::flux_pr (10 / 50 / 1024).
        config.sample.num_steps = 10
        config.sample.eval_num_steps = 50
        config.sample.eval_resolution = 1024
        config.sample.guidance_scale = 3.5
        config.resolution = 1024
        config.train.beta = 0.0001
        config.sample.noise_level = 0.7
        config.train.gradient_checkpointing = False
        bsz = 3

    if train_batch_size is not None:
        bsz = train_batch_size

    config.sample.num_image_per_prompt = 24
    num_groups = 48

    while True:
        if bsz < 1:
            assert False, "Cannot find a proper batch size."
        if (
            num_groups * config.sample.num_image_per_prompt % (n_gpus * bsz) == 0
            and bsz * n_gpus % config.sample.num_image_per_prompt == 0
        ):
            n_batch_per_epoch = num_groups * config.sample.num_image_per_prompt // (n_gpus * bsz)
            if n_batch_per_epoch % gradient_step_per_epoch == 0:
                config.sample.train_batch_size = bsz
                config.sample.num_batches_per_epoch = n_batch_per_epoch
                config.train.batch_size = config.sample.train_batch_size
                config.train.gradient_accumulation_steps = (
                    config.sample.num_batches_per_epoch // gradient_step_per_epoch
                )
                break
        bsz -= 1

    # special design, the test set has a total of 1018/2212/2048/400 for ocr/geneval/pickscore/hpsv2, to make gpu_num*bs*n as close as possible to it, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.
    if dataset == "geneval":
        config.sample.test_batch_size = 14
    elif dataset == "hpsv2":
        # 8 * 10 * 5 == 400: no last-batch padding.
        config.sample.test_batch_size = 10
    else:
        config.sample.test_batch_size = 16
    if n_gpus > 32:
        config.sample.test_batch_size = config.sample.test_batch_size // 2

    config.prompt_fn = "geneval" if dataset == "geneval" else "general_ocr"

    config.run_name = f"flowawr_{base_model}_{name}_avgknn_correction_clip"
    config.save_dir = f"{config.logdir}/{base_model}/{name}_avgknn_correction_clip"
    config.reward_fn = reward_fn

    config.decay_type = 1
    config.beta = 1.0
    config.train.energy_mode = True
    config.train.hard_gating = False # hard gating for Geneval and OCR that are rule-based score
    config.train.adv_mode = "all"

    # tmp.tex Distance/Similarity-Based Correction-Norm Clipping.
    # All algorithm hyperparameters live in config, not the training script.
    config.sample.global_std = True
    # The official base field is int; allow noninteger CLI clip thresholds.
    del config.train.adv_clip_max
    config.train.adv_clip_max = 5.0
    config.train.temperature = 1.0
    config.train.energy_min = -0.6
    config.train.energy_max = 5.0
    config.train.correction_knn_k = 3
    config.train.correction_score_std_multiplier = 2.0
    config.train.reward_std_epsilon = 1e-4
    config.train.adaptive_weight_floor = 1e-5
    # A1.9: train every sample on the complete scheduler grid (10 for SD3).
    config.train.timestep_fraction = 1.0

    # A4.20.1 trainer defaults, exposed here for CLI overrides and config logs.
    config.train.lora_rank = 32
    config.train.lora_alpha = 64
    config.train.lora_init_weights = "gaussian"
    config.train.lora_target_modules = [
        "attn.add_k_proj",
        "attn.add_q_proj",
        "attn.add_v_proj",
        "attn.to_add_out",
        "attn.to_k",
        "attn.to_out.0",
        "attn.to_q",
        "attn.to_v",
    ]
    config.train.ema_decay = 0.9
    config.train.ema_update_step_interval = 1
    config.train.old_policy_decay_schedules = {
        "0": {"flat": 0, "rate": 0.0, "cap": 0.0},
        "1": {"flat": 0, "rate": 0.001, "cap": 0.5},
        "2": {"flat": 75, "rate": 0.0075, "cap": 0.999},
    }
    config.sample.max_sequence_length = 128
    config.sample.eval_deterministic = True
    config.sample.eval_solver = "flow"
    config.data_loader_workers = 0
    config.reward_workers = 8

    config.sample.guidance_scale = 1.0
    config.sample.deterministic = True
    config.sample.solver = "dpm2"
    return config


def sd3_ocr():
    reward_fn = {
        "ocr": 1.0,
    }
    config = _get_config(
        base_model="sd3", n_gpus=8, gradient_step_per_epoch=2, dataset="ocr", reward_fn=reward_fn, name="ocr"
    )
    config.beta = 0.1
    config.decay_type = 1
    config.train.hard_gating = False  # Use K * softmax - 1, as in tmp.tex.
    return config


def sd3_geneval():
    # beta setting 1.0
    reward_fn = {
        "geneval": 1.0,
    }
    config = _get_config(
        base_model="sd3",
        n_gpus=8,
        gradient_step_per_epoch=1,
        dataset="geneval",
        reward_fn=reward_fn,
        name="geneval",
    )
    config.train.hard_gating = False  # Use K * softmax - 1, as in tmp.tex.
    return config


def sd3_pickscore():
    reward_fn = {
        "pickscore": 1.0,
    }
    config = _get_config(
        base_model="sd3",
        n_gpus=8,
        gradient_step_per_epoch=1,
        dataset="pickscore",
        reward_fn=reward_fn,
        name="pickscore",
    )
    config.log_raw_data = True
    # Match the actual A4.20.1 PickScore evaluation/checkpoint cadence.
    config.eval_freq = 10
    config.save_freq = 10
    return config


def sd3_hpsv2():
    reward_fn = {
        "hpsv2": 1.0,
    }
    config = _get_config(
        base_model="sd3", n_gpus=8, gradient_step_per_epoch=1, dataset="pickscore", reward_fn=reward_fn, name="hpsv2"
    )
    return config


def sd3_multi_reward():
    reward_fn = {
        "pickscore": 1.0,
        "hpsv2": 1.0,
        "clipscore": 1.0,
    }
    config = _get_config(
        base_model="sd3",
        n_gpus=8,
        gradient_step_per_epoch=1,
        dataset="pickscore",
        reward_fn=reward_fn,
        name="multi_reward",
    )
    config.sample.num_steps = 25
    config.beta = 0.1
    return config


def sd3_multi_reward_geneval():
    reward_fn = {
        "pickscore": 1.0,
        "hpsv2": 1.0,
        "geneval": 1.0,
        "clipscore": 1.0,
    }
    config = _get_config(
        base_model="sd3",
        n_gpus=8,
        gradient_step_per_epoch=1,
        dataset="geneval",
        reward_fn=reward_fn,
        name="multi_reward",
    )
    config.sample.num_steps = 10
    return config


def sd3_multi_reward_pickscore():
    reward_fn = {
        "pickscore": 1.0,
        "hpsv2": 1.0,
        "clipscore": 1.0,
    }
    config = _get_config(
        base_model="sd3",
        n_gpus=8,
        gradient_step_per_epoch=1,
        dataset="pickscore",
        reward_fn=reward_fn,
        name="multi_reward",
    )
    config.sample.num_steps = 10
    return config


def sd3_multi_reward_ocr():
    reward_fn = {
        "pickscore": 1.0,
        "clipscore": 1.0,
        "ocr": 1.0,
        "hpsv2": 1.0,
    }
    config = _get_config(
        base_model="sd3",
        n_gpus=8,
        gradient_step_per_epoch=1,
        dataset="ocr",
        reward_fn=reward_fn,
        name="multi_reward",
    )
    config.sample.num_steps = 20
    config.decay_type = 1
    return config


# ===================== FLUX Configurations =====================


def flux_pickscore():
    reward_fn = {
        "pickscore": 1.0,
    }
    config = _get_config(
        base_model="flux",
        n_gpus=8,
        gradient_step_per_epoch=1,
        dataset="pickscore",
        reward_fn=reward_fn,
        name="pickscore",
    )
    return config


def flux_hpsv2():
    reward_fn = {
        "hpsv2": 1.0,
    }
    config = _get_config(
        base_model="flux",
        n_gpus=8,
        gradient_step_per_epoch=1,
        dataset="pickscore",
        reward_fn=reward_fn,
        name="hpsv2",
    )
    return config


def flux_hpsv3():
    reward_fn = {
        "hpsv3": 1.0,
    }
    config = _get_config(
        base_model="flux",
        n_gpus=8,
        gradient_step_per_epoch=1,
        dataset="hpsv2",
        reward_fn=reward_fn,
        name="hpsv3",
        train_batch_size=6,
    )
    config.resolution = 720
    return config


def flux_geneval():
    reward_fn = {
        "geneval": 1.0,
    }
    config = _get_config(
        base_model="flux",
        n_gpus=8,
        gradient_step_per_epoch=1,
        dataset="geneval",
        reward_fn=reward_fn,
        name="geneval",
    )
    return config


def flux_ocr():
    reward_fn = {
        "ocr": 1.0,
    }
    config = _get_config(
        base_model="flux",
        n_gpus=8,
        gradient_step_per_epoch=2,
        dataset="ocr",
        reward_fn=reward_fn,
        name="ocr",
    )
    config.beta = 0.1
    config.decay_type = 2
    return config


def flux_multi_reward_pickscore():
    reward_fn = {
        "pickscore": 1.0,
        "hpsv2": 1.0,
        "clipscore": 1.0,
    }
    config = _get_config(
        base_model="flux",
        n_gpus=8,
        gradient_step_per_epoch=1,
        dataset="pickscore",
        reward_fn=reward_fn,
        name="multi_reward",
    )
    config.sample.num_steps = 10
    return config


def flux_multi_reward_geneval():
    reward_fn = {
        "pickscore": 1.0,
        "hpsv2": 1.0,
        "geneval": 1.0,
        "clipscore": 1.0,
    }
    config = _get_config(
        base_model="flux",
        n_gpus=8,
        gradient_step_per_epoch=1,
        dataset="geneval",
        reward_fn=reward_fn,
        name="multi_reward",
    )
    config.sample.num_steps = 10
    return config


