"""A1 diagnostic raw logging: preserve training and add absolute x0-space norms."""

import json
import math

import torch

from flow_grpo import pickscore_logging as base


EXTRA_TRAINING_FIELDS = (
    "old_l2", "old_rms", "x0_l2", "current_l2", "old_residual_dot",
)
TRAINING_FIELDS = base.TRAINING_FIELDS + EXTRA_TRAINING_FIELDS

# Samples, prompts, rewards and energy diagnostics retain the frozen A1 logger.
log_sample_batch = base.log_sample_batch
log_weight_batch = base.log_weight_batch


def create_raw_logger(config, rank, world_size):
    log = base.create_raw_logger(config, rank, world_size)
    if log is not None and rank == 0:
        schema = {
            "schema_version": 2,
            "variant": "a1_absolute_old_norm_logging_only",
            "training_fields": list(TRAINING_FIELDS),
            "measurement": "Each sample and actual training t, before optimizer update; all norms in x0 prediction space.",
            "old_definition": "pred_x0_old = xt - t * old_velocity_prediction",
            "old_l2": "L2(pred_x0_old), not parameter norm, velocity norm or current-old difference",
            "old_rms": "old_l2 / sqrt(latent_numel)",
            "x0_l2": "L2(generated clean latent x0)",
            "current_l2": "L2(pred_x0_current)",
            "old_residual_dot": "dot(pred_x0_old, x0-pred_x0_old); residual subtraction uses the original training dtype",
            "relative_unweighted_correction": "residual_l2 / old_l2; exact old_l2==0 is an undefined relative norm, not silently floored",
            "relative_actual_correction": "abs(energy_used) * residual_l2 / old_l2",
            "target_l2_squared_reconstruction": "old_l2**2 + 2*energy_used*old_residual_dot + energy_used**2*residual_l2**2; subject to scalar rounding",
            "precision": "Existing columns are generated unchanged by the A1 logger. Added vector norms use max-scaled FP32 reduction with FP64 final scaling; added dot uses FP64 accumulation.",
            "training_unchanged": "Detached diagnostics only; no clipping, loss change, optimizer change, random draw or model forward added.",
        }
        (log["path"].parent / "old_norm_schema.json").write_text(
            json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return log


def _scaled_l2(flat):
    """Keep finite FP32 vectors measurable even when naive squaring overflows."""
    scale = flat.abs().amax(dim=1)
    divisor = scale.clamp_min(torch.finfo(torch.float32).tiny)
    unit_norm = (flat / divisor[:, None]).norm(dim=1)
    return scale.double() * unit_norm.double()


@torch.no_grad()
def log_training_batch(log, *, sample_index, t, energy_used, x0, pred_x0_old,
                       pred_x0_current, policy_loss, weight_factor, epoch,
                       global_step, inner_epoch, microbatch, timestep_index):
    # Reuse the original implementation so all historical fields stay identical.
    base.log_training_batch(
        log, sample_index=sample_index, t=t, energy_used=energy_used, x0=x0,
        pred_x0_old=pred_x0_old, pred_x0_current=pred_x0_current,
        policy_loss=policy_loss, weight_factor=weight_factor, epoch=epoch,
        global_step=global_step, inner_epoch=inner_epoch, microbatch=microbatch,
        timestep_index=timestep_index,
    )
    old = pred_x0_old.detach().float().flatten(1)
    clean = x0.detach().float().flatten(1)
    current = pred_x0_current.detach().float().flatten(1)
    residual = (x0 - pred_x0_old).detach().float().flatten(1)
    old_l2 = _scaled_l2(old)
    extra = torch.stack((
        old_l2, old_l2 / math.sqrt(old.shape[1]),
        _scaled_l2(clean), _scaled_l2(current),
        (old.double() * residual.double()).sum(dim=1),
    ), dim=1)
    values, context = log["training"][-1]
    context = dict(context, latent_numel=old.shape[1], training_terms_schema_version=2)
    log["training"][-1] = (torch.cat((values, extra), dim=1), context)


def flush_training(log):
    if not log["training"]:
        return
    values = torch.cat([batch for batch, _ in log["training"]]).cpu().tolist()
    rows, offset = [], 0
    for batch, context in log["training"]:
        for numbers in values[offset:offset + len(batch)]:
            row = dict(context, **dict(zip(TRAINING_FIELDS, numbers, strict=True)))
            row["sample_index"] = int(row["sample_index"])
            row["sample_id"] = base._sample_id(log, "train", row["epoch"], row["sample_index"])
            row["split"] = "train"
            rows.append(row)
        offset += len(batch)
    base._write_rows(log, "training_terms", rows)
    log["training"].clear()


def log_update(log, **kwargs):
    # Empty the extended buffer before the baseline update logger flushes it.
    flush_training(log)
    base.log_update(log, **kwargs)
