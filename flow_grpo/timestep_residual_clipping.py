"""Fixed empirical, sign-specific x0-residual RMS caps at actual scheduler t.

No online adaptation: a calibrated table is frozen before distributed setup.
The raw logger preserves the historical fields and adds old norms and the
applied per-record cap. Unknown time points are errors, never interpolated.
"""

import json
import math

import torch

from flow_grpo import pickscore_logging as base_raw_logging


ACTUAL_T_GRID = (
    0.008928571827709675, 0.2780488133430481, 0.46487605571746826,
    0.6021506190299988, 0.7072785496711731, 0.7903683185577393,
    0.8576923608779907, 0.9133490324020386, 0.9601293802261353, 1.0,
)
OLD_NORM_FIELDS = (
    "old_l2", "old_rms", "x0_l2", "current_l2", "old_residual_dot",
)
CLIP_FIELDS = (
    "residual_rms_before", "residual_rms_after", "residual_clip_scale",
    "residual_l2_after_clip", "delta_residual_dot_after_clip", "energy_effective",
    "residual_rms_threshold", "residual_l2_threshold", "energy_sign",
    "actual_correction_l2", "old_used_residual_dot",
)
TRAINING_FIELDS = base_raw_logging.TRAINING_FIELDS + OLD_NORM_FIELDS + CLIP_FIELDS
log_sample_batch = base_raw_logging.log_sample_batch
log_weight_batch = base_raw_logging.log_weight_batch


def _finite_positive(values, label):
    values = tuple(values)
    if not values or any(isinstance(v, bool) or not math.isfinite(float(v)) or float(v) <= 0 for v in values):
        raise ValueError(f"{label} must contain calibrated finite positive values")
    return tuple(float(v) for v in values)


class FixedTimestepResidualClip:
    """Per-sample radial clipping; sign selects a column, actual t selects a row."""

    def __init__(self, timesteps, positive_rms, negative_rms, *, enabled=True):
        if not isinstance(enabled, bool):
            raise ValueError("residual_clip_enabled must be a boolean")
        grid = _finite_positive(timesteps, "residual_t_grid")
        positive = _finite_positive(positive_rms, "residual_rms_positive")
        negative = _finite_positive(negative_rms, "residual_rms_negative")
        if not len(grid) == len(positive) == len(negative):
            raise ValueError("Both sign tables must have one cap per actual timestep")
        keys = torch.tensor(grid, dtype=torch.float32)
        if not bool(torch.all(torch.isfinite(keys) & (keys > 0) & (keys <= 1))):
            raise ValueError("Actual t keys must be finite float32 values in (0, 1]")
        if keys.numel() > 1 and not bool(torch.all(keys[1:] > keys[:-1])):
            raise ValueError("Actual t keys must be strictly increasing after float32 conversion")
        caps = torch.tensor([positive, negative], dtype=torch.float32)
        if not bool(torch.all(torch.isfinite(caps) & (caps > 0))):
            raise ValueError("RMS caps must remain finite and positive in FP32")
        self.keys, self.positive, self.negative = keys, positive, negative
        self.enabled = enabled
        self._device_tables = {}

    @torch.no_grad()
    def lookup(self, t, energy):
        keys = t.detach().reshape(-1).to(dtype=torch.float32)
        energy = energy.detach().reshape(-1)
        if keys.numel() != energy.numel() or keys.device != energy.device:
            raise ValueError("t and energy must contain one value per sample on the same device")
        if keys.device not in self._device_tables:
            self._device_tables[keys.device] = (
                self.keys.to(keys.device),
                torch.tensor(self.positive, dtype=torch.float64, device=keys.device),
                torch.tensor(self.negative, dtype=torch.float64, device=keys.device),
            )
        grid, positive, negative = self._device_tables[keys.device]
        positions = torch.searchsorted(grid, keys)
        safe_positions = positions.clamp_max(grid.numel() - 1)
        known = (positions < grid.numel()) & (grid[safe_positions] == keys)
        if not bool(torch.all(known)):
            unknown = keys[~known].unique().cpu().tolist()
            raise ValueError(f"Uncalibrated actual t values: {unknown}; interpolation is disabled")
        # Preserve A4.4's all-record clipping semantics. Zero energy uses the
        # positive cap; the target remains old because its multiplier is zero.
        return torch.where(energy < 0, negative[safe_positions], positive[safe_positions])

    @torch.no_grad()
    def __call__(self, residual, t, energy):
        if residual.ndim < 2 or any(n == 0 for n in residual.shape):
            raise ValueError("residual must have nonempty batch and latent dimensions")
        if not residual.is_floating_point():
            raise TypeError("residual must be floating point")
        if t.numel() != residual.shape[0] or energy.numel() != residual.shape[0]:
            raise ValueError("t and energy must contain one value per residual sample")
        if residual.device != t.device or residual.device != energy.device:
            raise ValueError("residual, t and energy must be on the same device")
        work = residual.detach()
        if work.dtype in (torch.float16, torch.bfloat16):
            work = work.float()
        shape = (work.shape[0],) + (1,) * (work.ndim - 1)
        threshold = self.lookup(t, energy).to(dtype=work.dtype).reshape(shape)
        dims = tuple(range(1, work.ndim))
        normalizer = work.abs().amax(dim=dims, keepdim=True).clamp_min(torch.finfo(work.dtype).tiny)
        rms = (work / normalizer).square().mean(dim=dims, keepdim=True).sqrt() * normalizer
        if self.enabled:
            # Match A4.4's Python-scalar / Tensor operation order (reciprocal,
            # then multiply), including FP32 rounding when all caps are equal.
            cap_scale = (rms.clamp_min(torch.finfo(work.dtype).tiny).reciprocal() * threshold).clamp(max=1.0)
            # Exact boundary values must not shrink because reciprocal/multiply
            # rounds slightly below one for some nonbinary empirical caps.
            cap_scale = torch.where(rms <= threshold, torch.ones_like(cap_scale), cap_scale)
            scale = cap_scale
        else:
            scale = torch.ones_like(rms)
        return work * scale, rms, scale, threshold


def validate_timestep_residual_clip_config(config):
    """Reject uncalibrated candidates before distributed setup or model loading."""
    if not config.train.energy_mode or not config.log_raw_data:
        raise ValueError("Timestep residual clipping requires energy mode and raw logging")
    lower, upper = float(config.train.energy_min), float(config.train.energy_max)
    if not (math.isfinite(lower) and math.isfinite(upper) and lower < 0 < upper):
        raise ValueError("Energy bounds must be finite and straddle zero to preserve raw signs")
    calibration = config.train.residual_clip_calibration_id
    if not isinstance(calibration, str) or not calibration.strip():
        raise ValueError("No approved empirical residual clipping calibration is configured")
    clipper = FixedTimestepResidualClip(
        config.train.residual_t_grid, config.train.residual_rms_positive,
        config.train.residual_rms_negative, enabled=config.train.residual_clip_enabled,
    )
    if not torch.equal(clipper.keys, torch.tensor(ACTUAL_T_GRID, dtype=torch.float32)):
        raise ValueError("This experiment requires the calibrated ten-point SD3 actual-t grid")
    return clipper


def _scaled_l2(flat):
    scale = flat.abs().amax(dim=1)
    divisor = scale.clamp_min(torch.finfo(torch.float32).tiny)
    return scale.double() * (flat / divisor[:, None]).norm(dim=1).double()


def create_raw_logger(config, rank, world_size):
    log = base_raw_logging.create_raw_logger(config, rank, world_size)
    if log is not None:
        log["timestep_clip_metadata"] = {
            "residual_clip_enabled": config.train.residual_clip_enabled,
            "residual_clip_profile": config.train.residual_clip_profile,
            "residual_clip_calibration_id": config.train.residual_clip_calibration_id,
        }
    if log is not None and rank == 0:
        schema = {
            "schema_version": 3,
            "variant": "fixed_timestep_signed_unweighted_x0_residual_rms_clip",
            "training_fields": list(TRAINING_FIELDS),
            "norm": "RMS(x0-pred_x0_old), over all latent dimensions, separately per sample and actual t; no energy or division by t",
            "t_grid": list(config.train.residual_t_grid),
            "positive_rms_thresholds": list(config.train.residual_rms_positive),
            "negative_rms_thresholds": list(config.train.residual_rms_negative),
            "lookup": "Exact float32 actual t key; unknown keys fail; no interpolation or online adaptation",
            "zero_energy": "Uses the positive cap, preserving A4.4 all-record clipping; the target remains old",
            "target": "pred_x0_old + energy_used * residual_clip_scale * (x0-pred_x0_old)",
            "energy_effective": "energy_used * residual_clip_scale, per actual t; never overwrites weights.energy_used",
            "original_fields": "residual_l2 and delta_residual_dot describe the original unclipped residual; energy_used is the reward coefficient after its independent energy clip",
            "policy_loss_reconstruction": "(delta_l2**2 + energy_used**2 * residual_l2_after_clip**2 - 2*energy_used*delta_residual_dot_after_clip)/(latent_numel*weight_factor)",
            "old_definition": "pred_x0_old = xt - t*old_velocity_prediction; x0 prediction norm, not model parameter norm",
            "old_rms": "old_l2/sqrt(latent_numel)",
            "relative_actual_correction": "actual_correction_l2/old_l2; zero old norm makes ratio undefined",
            "target_l2_squared_reconstruction": "old_l2**2 + 2*energy_used*old_used_residual_dot + energy_used**2*residual_l2_after_clip**2",
            "precision": "Legacy fields preserve A4.4 reductions; new old norms use scaled FP32 reduction/FP64 final scale and old dot products use FP64 accumulation; clipping itself is FP32 for FP16/BF16 inputs",
            **log["timestep_clip_metadata"],
        }
        (log["path"].parent / "timestep_residual_clip_schema.json").write_text(
            json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return log


@torch.no_grad()
def log_training_batch(log, *, sample_index, t, energy_used, x0, pred_x0_old,
                       pred_x0_current, policy_loss, weight_factor, epoch,
                       global_step, inner_epoch, microbatch, timestep_index,
                       residual_used, residual_rms_before, residual_clip_scale,
                       residual_rms_threshold):
    residual = (x0 - pred_x0_old).float().flatten(1)
    used = residual_used.float().flatten(1)
    delta = (pred_x0_current - pred_x0_old).float().flatten(1)
    old, clean, current = (v.detach().float().flatten(1) for v in (pred_x0_old, x0, pred_x0_current))
    old_l2, used_l2 = _scaled_l2(old), used.norm(dim=1)
    scale, threshold = residual_clip_scale.flatten(), residual_rms_threshold.flatten()
    root_dimension = math.sqrt(used.shape[1])
    columns = (
        sample_index, t, policy_loss, weight_factor.flatten(),
        residual.norm(dim=1), delta.norm(dim=1), (delta * residual).sum(dim=1), energy_used,
        old_l2, old_l2 / root_dimension, _scaled_l2(clean), _scaled_l2(current),
        (old.double() * residual.double()).sum(dim=1),
        residual_rms_before.flatten(), used_l2 / root_dimension, scale,
        used_l2, (delta * used).sum(dim=1), energy_used * scale,
        threshold, threshold * root_dimension, energy_used.sign(),
        energy_used.abs() * used_l2, (old.double() * used.double()).sum(dim=1),
    )
    values = torch.stack([v.detach().double() for v in columns], dim=1)
    context = dict(epoch=epoch, global_step=global_step, inner_epoch=inner_epoch,
                   microbatch=microbatch, timestep_index=timestep_index,
                   latent_numel=used.shape[1], training_terms_schema_version=3,
                   **log["timestep_clip_metadata"])
    log["training"].append((values, context))


def flush_training(log):
    if not log["training"]:
        return
    values = torch.cat([batch for batch, _ in log["training"]]).cpu().tolist()
    rows, offset = [], 0
    for batch, context in log["training"]:
        for numbers in values[offset:offset + len(batch)]:
            row = dict(context, **dict(zip(TRAINING_FIELDS, numbers, strict=True)))
            row["sample_index"] = int(row["sample_index"])
            row["sample_id"] = base_raw_logging._sample_id(log, "train", row["epoch"], row["sample_index"])
            row["split"] = "train"
            row["residual_clipped"] = row["residual_clip_scale"] < 1.0
            row["residual_clip_eligible"] = row["residual_clip_enabled"]
            row["residual_clip_side"] = "negative" if row["energy_used"] < 0 else "positive" if row["energy_used"] > 0 else "zero"
            rows.append(row)
        offset += len(batch)
    base_raw_logging._write_rows(log, "training_terms", rows)
    log["training"].clear()


def log_update(log, **kwargs):
    flush_training(log)
    base_raw_logging.log_update(log, **kwargs)
