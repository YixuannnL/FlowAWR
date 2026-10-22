"""SD3 单 PickScore 实验的逐样本日志，不跨样本或时间点求平均。"""

import datetime
import hashlib
import json
import math
from pathlib import Path

import torch
import torch.distributed as dist


TRAINING_FIELDS = (
    "sample_index", "t", "policy_loss", "weight_factor",
    "residual_l2", "delta_l2", "delta_residual_dot", "energy_used",
)


def create_raw_logger(config, rank, world_size):
    if not config.log_raw_data:
        return None
    # 日志只对应当前单 PickScore 的 AWR energy 分支。
    assert (set(config.reward_fn) == {"pickscore"} and config.train.energy_mode
            and not config.gdpo and not config.filter_zero_advantage)

    # 各卡使用同一个目录；每次启动（含续训）新建一段，避免样本编号重复。
    info = [None]
    if rank == 0:
        attempt = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        info[0] = (config.save_dir, config.run_name, attempt)
    if world_size > 1:
        dist.broadcast_object_list(info, src=0)
    save_dir, run_id, attempt = info[0]
    root = Path(save_dir) / "raw_data" / attempt
    path = root / f"rank-{rank:05d}"
    path.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        (root / "config.json").write_text(
            json.dumps(config.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Raw PickScore logs: {root}")
    return dict(path=path, run_id=run_id, attempt_id=attempt, rank=rank,
                prompts=set(), training=[])


def _write_rows(log, table, rows):
    # 每卡独立写文件，每行一个样本或一次更新；写完即关闭文件。
    with (log["path"] / f"{table}.jsonl").open("a", encoding="utf-8") as file:
        for row in rows:
            row = dict(row, run_id=log["run_id"], attempt_id=log["attempt_id"], rank=log["rank"])
            # 非有限数值记为 null 并注明原因，避免被误当成 0。
            invalid = {k: str(v) for k, v in row.items()
                       if isinstance(v, float) and not math.isfinite(v)}
            for key in invalid:
                row[key] = None
            if invalid:
                row["nonfinite_fields"] = invalid
            file.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def _sample_id(log, split, epoch, index):
    return f"{log['attempt_id']}:{log['rank']}:{split}:{epoch}:{index}"


def _prompt_ids(log, prompts):
    ids, rows = [], []
    for prompt in prompts:
        key = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        ids.append(key)
        if key not in log["prompts"]:
            rows.append(dict(prompt_id=key, prompt=prompt))
            log["prompts"].add(key)
    if rows:
        _write_rows(log, "prompts", rows)
    return ids


def log_sample_batch(log, *, split, epoch, global_step, sample_offset,
                     prompts, reward_metadata, model_adapter, dataset_size=None, world_size=1):
    """逐张保存原始 PickScore；分数由评分器在除以 26 之前提供。"""
    scores = reward_metadata["pickscore_raw"]
    rows = []
    for offset, (prompt_id, score) in enumerate(zip(_prompt_ids(log, prompts), scores, strict=True)):
        index = sample_offset + offset
        row = dict(sample_id=_sample_id(log, split, epoch, index), sample_index=index,
                   split=split, epoch=epoch, global_step=global_step,
                   prompt_id=prompt_id, pickscore_raw=float(score), model_adapter=model_adapter)
        if dataset_size is not None:
            # eval 的 DistributedSampler 会补齐重复样本，分析时可据此排除。
            position = log["rank"] + index * world_size
            row.update(dataset_index=position % dataset_size, is_padding=position >= dataset_size)
        rows.append(row)
    _write_rows(log, "samples", rows)


def log_weight_batch(log, *, epoch, global_step, group_keys, diagnostics, samples_per_rank):
    """保存原计算产生的 advantage、softmax 概率、裁剪前后 energy。"""
    start = log["rank"] * samples_per_rank
    stop = start + samples_per_rank
    group_ids = _prompt_ids(log, group_keys[start:stop])
    # reward 只是复制到各 timestep，因此取一份系数，不做平均。
    values = {key: value[start:stop, 0] for key, value in diagnostics.items()}
    rows = []
    for index, group_prompt_id in enumerate(group_ids):
        rows.append(dict(
            sample_id=_sample_id(log, "train", epoch, index), sample_index=index,
            epoch=epoch, global_step=global_step,
            # 用 tracker 实际收到的 decoded prompt 分组，跨卡使用相同组 ID。
            group_id=f"{log['attempt_id']}:{epoch}:{group_prompt_id}",
            group_prompt_id=group_prompt_id,
            **{key: float(value[index]) for key, value in values.items()},
        ))
    _write_rows(log, "weights", rows)


@torch.no_grad()
def log_training_batch(log, *, sample_index, t, energy_used, x0, pred_x0_old,
                       pred_x0_current, policy_loss, weight_factor, epoch,
                       global_step, inner_epoch, microbatch, timestep_index):
    # energy 实际乘的残差，以及当前预测相对旧预测的变化，均在 x0 空间。
    # 先按训练 dtype 相减，再用 FP32 在单个样本内部求范数和内积。
    residual = (x0 - pred_x0_old).float().flatten(1)
    delta = (pred_x0_current - pred_x0_old).float().flatten(1)
    columns = (
        sample_index, t, policy_loss, weight_factor.flatten(),
        residual.norm(dim=1), delta.norm(dim=1), (delta * residual).sum(dim=1), energy_used,
    )
    # 只缓冲 detached 小标量，每次参数更新时统一拷到 CPU，不保存完整张量。
    values = torch.stack([v.detach().double() for v in columns], dim=1)
    context = dict(epoch=epoch, global_step=global_step, inner_epoch=inner_epoch,
                   microbatch=microbatch, timestep_index=timestep_index)
    log["training"].append((values, context))


def flush_training(log):
    if not log["training"]:
        return
    values = torch.cat([batch for batch, _ in log["training"]]).cpu().tolist()
    rows, offset = [], 0
    for batch, context in log["training"]:
        for numbers in values[offset:offset + len(batch)]:
            row = dict(context, **dict(zip(TRAINING_FIELDS, numbers)))
            row["sample_index"] = int(row["sample_index"])
            row["sample_id"] = _sample_id(log, "train", row["epoch"], row["sample_index"])
            row["split"] = "train"
            rows.append(row)
        offset += len(batch)
    _write_rows(log, "training_terms", rows)
    log["training"].clear()


def log_update(log, *, epoch, global_step, inner_epoch, grad_norm_before_clip,
               max_grad_norm, scaler_scale, optimizer_step_skipped):
    # 复用原 clip_grad_norm_ 的返回值，不再裁剪一次。系数与 PyTorch 一致。
    norm = grad_norm_before_clip.detach()
    coefficient = (float(max_grad_norm) / (norm + 1e-6)).clamp(max=1.0)
    norm, coefficient = torch.stack((norm, coefficient)).cpu().tolist()
    flush_training(log)
    _write_rows(log, "updates", [dict(
        epoch=epoch, global_step=global_step, inner_epoch=inner_epoch,
        grad_norm_before_clip=norm, grad_clip_coefficient=coefficient,
        max_grad_norm=float(max_grad_norm), optimizer_step_skipped=optimizer_step_skipped,
        # 本次梯度使用的 scale（scaler.update 前）；未使用缩放时为 null。
        scaler_scale=scaler_scale,
    )])
