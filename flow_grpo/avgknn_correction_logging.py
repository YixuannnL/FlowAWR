"""Detached per-sample AvgKNN diagnostics, alongside the unchanged raw tables.

Norms are unweighted x-space L2 norms. ``candidate`` describes the all-sign
detector, whereas ``applied_clip`` describes the actual, already sign-masked
scale supplied by training. Integer sample/group identities never pass through
floating point. Each row joins training_terms on sample_id, inner_epoch and t.
"""

import torch

from flow_grpo.pickscore_logging import _sample_id, _write_rows


_BUFFER = "avgknn_correction_batches"
_INTEGER_FIELDS = (
    "sample_index", "rollout_group_id", "peer_count", "eligible", "candidate",
    "applied_clip",
)
_FLOAT_FIELDS = (
    "t", "scheduler_t", "correction_norm", "avgknn_score",
    "avgknn_score_mean", "avgknn_score_std", "avgknn_score_threshold",
    "norm_cap", "energy", "applied_scale", "clipped_norm",
)


@torch.no_grad()
def log_correction_batch(log, *, batch, timestep_index, t, energy,
                         applied_scale, clipped_correction, epoch, global_step,
                         inner_epoch, microbatch):
    """Buffer detached scalar columns; do not synchronize or retain latents.

    ``batch`` statistics have shape [sample, shuffled timestep position].
    ``timesteps`` supplies the actual scheduler timestep for the selected
    position. ``applied_scale`` must be the scale actually used to construct
    ``clipped_correction``, after applying the negative-coefficient mask.
    """
    if log is None:
        return

    sample_index = batch["raw_sample_index"].detach().reshape(-1)
    group_ids = batch["group_ids"].detach().reshape(-1)
    for name, identities in (("sample_index", sample_index), ("group_ids", group_ids)):
        if identities.is_floating_point() or identities.is_complex() or identities.dtype == torch.bool:
            raise TypeError(f"{name} must retain its integer identity dtype")

    def at_time(name):
        return batch[name][:, timestep_index].detach().reshape(-1)

    scale = applied_scale.detach().reshape(-1)
    integer_columns = (
        sample_index, group_ids, at_time("correction_count"),
        at_time("correction_eligible"), at_time("correction_candidate"),
        scale < 1,
    )
    float_columns = (
        t.detach().reshape(-1), at_time("timesteps"),
        at_time("correction_norm"), at_time("correction_score"),
        at_time("correction_score_mean"), at_time("correction_score_std"),
        at_time("correction_score_threshold"), at_time("correction_norm_cap"),
        energy.detach().reshape(-1), scale,
        clipped_correction.detach().double().flatten(1).norm(dim=1),
    )
    if any(column.numel() != sample_index.numel()
           for column in integer_columns + float_columns):
        raise ValueError("Expected one AvgKNN diagnostic record per sample")
    identities = torch.stack([column.to(dtype=torch.int64) for column in integer_columns], dim=1)
    values = torch.stack([column.double() for column in float_columns], dim=1)
    context = dict(
        epoch=epoch, global_step=global_step, inner_epoch=inner_epoch,
        microbatch=microbatch, timestep_index=timestep_index,
    )
    log.setdefault(_BUFFER, []).append((identities, values, context))


def flush_correction(log):
    """Write buffered records using the shared run/attempt/rank and sample IDs."""
    if log is None or not log.get(_BUFFER):
        return
    pending = log[_BUFFER]
    identities = torch.cat([batch[0] for batch in pending]).cpu().tolist()
    values = torch.cat([batch[1] for batch in pending]).cpu().tolist()
    rows, offset = [], 0
    for integer_batch, _, context in pending:
        for index in range(offset, offset + len(integer_batch)):
            row = dict(context, **dict(zip(_INTEGER_FIELDS, identities[index], strict=True)),
                       **dict(zip(_FLOAT_FIELDS, values[index], strict=True)))
            for field in ("eligible", "candidate", "applied_clip"):
                row[field] = bool(row[field])
            row["sample_id"] = _sample_id(log, "train", row["epoch"], row["sample_index"])
            row["split"] = "train"
            rows.append(row)
        offset += len(integer_batch)
    _write_rows(log, "avgknn_correction", rows)
    pending.clear()
