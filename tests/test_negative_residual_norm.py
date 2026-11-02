"""CPU checks for negative-side normalization, lagged scales, and global medians."""
from datetime import timedelta
from functools import partial
import ast
import math
import os
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from absl import flags
from ml_collections import config_flags

from flow_grpo.stat_tracking import PerPromptStatTracker
from config.awr_negative_residual_norm import sd3_pickscore


# Execute the trainer's actual definitions without loading SD3 or its GPU runtime.
_trainer_path = Path(__file__).resolve().parents[1] / "scripts/train_awr_sd3_negative_residual_norm.py"
_definitions = [node for node in ast.parse(_trainer_path.read_text()).body
                if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                and node.name in {"compute_weight_factor", "LaggedResidualScale"}]
assert {node.name for node in _definitions} == {"compute_weight_factor", "LaggedResidualScale"}
_namespace = {"torch": torch, "dist": dist, "math": math, "__name__": __name__}
exec(compile(ast.Module(body=_definitions, type_ignores=[]), str(_trainer_path), "exec"), _namespace)
_compute_weight_factor = _namespace["compute_weight_factor"]
LaggedResidualScale = _namespace["LaggedResidualScale"]


_train_config = sd3_pickscore().train
compute_weight_factor = partial(
    _compute_weight_factor,
    offset_coefficient=_train_config.offset_coefficient,
    residual_threshold=_train_config.residual_threshold,
    epsilon=_train_config.weight_factor_epsilon,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_positive_and_zero_keep_original_denominator_and_gradient(dtype):
    current = torch.tensor([[2., -1.], [3., 4.], [1., -2.]],
                           dtype=dtype, requires_grad=True)
    old = torch.zeros_like(current)
    rollout = torch.tensor([[1., 1.], [2., 3.], [0., 0.]], dtype=dtype)
    b = torch.tensor([0., .2, 5.], dtype=dtype)
    w, _, _ = compute_weight_factor(
        current, old, rollout, b, sigma=torch.full((3,), 1e-8),
        ready=torch.ones(3, dtype=torch.bool),
    )
    base = (current.detach().double() - rollout.double()).abs().mean(1, keepdim=True).clamp_min(1e-5)
    torch.testing.assert_close(w, base, rtol=0, atol=0)
    assert not w.requires_grad
    target = old + b[:, None] * (rollout - old)
    new_gradient = torch.autograd.grad(((current - target).square() / w).mean(1).sum(), current)[0]
    original_gradient = torch.autograd.grad(((current - target).square() / base).mean(1).sum(), current)[0]
    torch.testing.assert_close(new_gradient, original_gradient, rtol=0, atol=0)


def test_negative_branches_and_detached_denominator():
    current = torch.tensor([[0., 0.], [3., 3.], [2., 2.]],
                           dtype=torch.float64, requires_grad=True)
    old = torch.zeros_like(current)
    rollout = torch.tensor([[4., 4.], [0., 0.], [1., 1.]], dtype=torch.float64)
    b = torch.tensor([-.5, -.1, -.5], dtype=torch.float64)
    sigma = torch.tensor([1., 1., .01], dtype=torch.float64, requires_grad=True)
    w, rms, diagnostics = compute_weight_factor(
        current, old, rollout, b, sigma=sigma,
        ready=torch.tensor([True, True, False]),
    )
    # Residual floor, offset floor, then missing-reference offset-only.
    expected = torch.tensor([[80. / 11.], [math.sqrt(45.)], [math.sqrt(17.)]], dtype=torch.float64)
    torch.testing.assert_close(w, expected)
    torch.testing.assert_close(rms, torch.tensor([4., 0., 1.], dtype=torch.float64))
    assert not w.requires_grad and not rms.requires_grad
    assert all(not value.requires_grad for value in diagnostics.values())
    target = old + b[:, None] * (rollout - old)
    ((current - target).square() / w).mean(1).sum().backward()
    torch.testing.assert_close(current.grad, (current.detach() - target) / expected)
    assert sigma.grad is None


def test_large_residual_is_attenuated_and_cold_start_has_only_offset():
    current = torch.zeros((2, 2), dtype=torch.float64, requires_grad=True)
    rollout = torch.tensor([[10., 10.], [100., 100.]], dtype=torch.float64)
    b = torch.full((2,), -.5, dtype=torch.float64)
    cold_w, _, _ = compute_weight_factor(current, current.detach(), rollout, b)
    torch.testing.assert_close(cold_w.flatten(), torch.tensor([10., 100.], dtype=torch.float64))
    w, _, _ = compute_weight_factor(current, current.detach(), rollout, b,
                                     sigma=torch.ones(2), ready=torch.ones(2, dtype=torch.bool))
    target = b[:, None] * rollout
    grad = torch.autograd.grad(((current - target).square() / w).mean(1).sum(), current)[0]
    torch.testing.assert_close(grad[1], grad[0] / 10.)


def test_known_zero_reference_is_distinct_from_missing_and_stays_finite():
    current = torch.zeros((2, 1, 2, 2))
    w, _, _ = compute_weight_factor(
        current, current, torch.ones_like(current), torch.full((2,), -.5),
        sigma=torch.zeros(2), ready=torch.tensor([True, False]),
    )
    assert w.shape == (2, 1, 1, 1) and torch.isfinite(w).all()
    torch.testing.assert_close(w.flatten(), torch.tensor([1 / (2.2e-5), 1.], dtype=torch.float64))


def test_negative_denominator_does_not_depend_on_coefficient_magnitude():
    current = torch.ones((3, 2))
    w, _, _ = compute_weight_factor(current, torch.zeros_like(current),
                                     current * 10, torch.tensor([-.5, -.1, -1e-8]),
                                     sigma=torch.ones(3))
    torch.testing.assert_close(w, w[:1].expand_as(w), rtol=0, atol=0)


def test_reference_is_lagged_per_actual_timestep_and_replaced():
    reference = LaggedResidualScale()
    t = torch.tensor([857.25, 857.75, 500.5])
    _, ready = reference.lookup(t)
    assert not ready.any()
    reference.observe(torch.tensor([857.75, 857.25, 857.75, 857.25]),
                      torch.tensor([8., 1., 2., 3.], requires_grad=True))
    assert not reference.lookup(t)[1].any()
    reference.finish_round()
    sigma, ready = reference.lookup(t)
    torch.testing.assert_close(sigma[:2].double(), torch.tensor([2., 5.], dtype=torch.float64))
    assert ready.tolist() == [True, True, False]
    reference.observe(torch.tensor([857.25, 857.25]), torch.tensor([30., 10.]))
    torch.testing.assert_close(reference.lookup(t)[0][:2].double(), sigma[:2].double())
    reference.finish_round()
    sigma, ready = reference.lookup(t)
    assert sigma[0].item() == 20.
    assert ready.tolist() == [True, False, False]
    assert reference.completed_rounds == 2


def test_reference_checkpoint_roundtrip_and_pending_save_rejected(tmp_path):
    reference = LaggedResidualScale()
    reference.observe(torch.tensor([.5, .5, .75]), torch.tensor([1., 3., 0.]))
    reference.finish_round()
    path = tmp_path / "residual_scale.pt"
    torch.save(reference.state_dict(), path)
    restored = LaggedResidualScale()
    restored.load_state_dict(torch.load(path, weights_only=True))
    sigma, ready = restored.lookup(torch.tensor([.75, .5, .6]))
    assert sigma[:2].tolist() == [0., 2.]
    assert ready.tolist() == [True, True, False]
    assert restored.completed_rounds == 1
    restored.observe(torch.tensor([.5]), torch.tensor([10.]))
    with pytest.raises(RuntimeError):
        restored.state_dict()


def _distributed_median_worker(rank, rendezvous, output_dir):
    # This is a local two-process check; avoid dependence on LAN interfaces.
    os.environ["GLOO_SOCKET_IFNAME"] = "lo0" if sys.platform == "darwin" else "lo"
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank,
                            world_size=2, timeout=timedelta(seconds=30))
    try:
        reference = LaggedResidualScale()
        if rank == 0:
            t, r = [10.25, 10.25, 10.75], [0., 100., 4.]
        else:
            t, r = [10.25] * 3 + [10.75] * 3, [10., 12., 14., 8., 9., 12.]
        reference.observe(torch.tensor(t), torch.tensor(r))
        reference.finish_round()
        sigma, ready = reference.lookup(torch.tensor([10.25, 10.75]))
        torch.save((sigma, ready), f"{output_dir}/rank-{rank}.pt")
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="CPU Gloo unavailable")
def test_distributed_global_median_uses_all_values_and_handles_uneven_counts(tmp_path):
    rendezvous = (tmp_path / "gloo-init").as_uri()
    mp.spawn(_distributed_median_worker, args=(rendezvous, str(tmp_path)), nprocs=2, join=True)
    for rank in range(2):
        sigma, ready = torch.load(tmp_path / f"rank-{rank}.pt", weights_only=True)
        # Per-rank median averaging gives 31 and 6.5, both wrong.
        torch.testing.assert_close(sigma.double(), torch.tensor([12., 8.5], dtype=torch.float64))
        assert ready.all()


def test_clipping_applies_to_raw_energy_before_old_upper_bound():
    config = sd3_pickscore()
    assert (config.train.energy_min, config.train.energy_max) == (-.5, 5.)
    tracker = PerPromptStatTracker(adv_clip_max=config.train.adv_clip_max,
                                  energy_min=config.train.energy_min,
                                  energy_max=config.train.energy_max)
    diagnostics = {}
    _, energy, _ = tracker.update(["same"] * 24, np.array([100.] + [0.] * 23),
                                 exp=True, diagnostics=diagnostics)
    assert diagnostics["energy_before_clip"].max() > 5.
    assert energy.max() == 5.  # Clipping an already-clipped A1 coefficient would give 3.
    assert energy.min() == -.5


def test_trainer_passes_raw_coefficient_clip_limits_into_tracker():
    script = Path(__file__).resolve().parents[1] / "scripts/train_awr_sd3_negative_residual_norm.py"
    calls = [node.value for node in ast.walk(ast.parse(script.read_text()))
             if isinstance(node, ast.Assign)
             and any(isinstance(target, ast.Name) and target.id == "stat_tracker" for target in node.targets)
             and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
             and node.value.func.id == "PerPromptStatTracker"]
    assert len(calls) == 1
    keywords = {kw.arg: ast.unparse(kw.value) for kw in calls[0].keywords}
    assert keywords["energy_min"] == "config.train.energy_min"
    assert keywords["energy_max"] == "config.train.energy_max"


def test_command_line_overrides_reach_clipping_and_normalization(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(root)
    script = ast.parse((root / "scripts/train_awr_sd3_negative_residual_norm.py").read_text())
    declaration = next(node for node in ast.walk(script)
                       if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                       and node.func.attr == "DEFINE_config_file")
    # Exercise the trainer's actual default config with the repo's flag parser,
    # without importing SD3 or starting distributed training.
    default_config = ast.literal_eval(declaration.args[1])
    argv = ["test", f"--config={default_config}",
            "--config.train.energy_min=-0.25", "--config.train.energy_max=4",
            "--config.train.offset_coefficient=3", "--config.train.residual_threshold=4",
            "--config.train.weight_factor_epsilon=0.125"]
    flag_values = flags.FlagValues()
    holder = config_flags.DEFINE_config_file(
        "config", default_config, flag_values=flag_values, sys_argv=argv)
    flag_values(argv)
    config = holder.value

    tracker = PerPromptStatTracker(adv_clip_max=config.train.adv_clip_max,
                                  energy_min=config.train.energy_min,
                                  energy_max=config.train.energy_max)
    _, energy, _ = tracker.update(["same"] * 24, np.array([100.] + [0.] * 23), exp=True)
    assert energy.min() == -.25 and energy.max() == 4.

    call = next(node for node in ast.walk(script)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "compute_weight_factor")
    keywords = {kw.arg: ast.unparse(kw.value) for kw in call.keywords}
    assert keywords["offset_coefficient"] == "config.train.offset_coefficient"
    assert keywords["residual_threshold"] == "config.train.residual_threshold"
    assert keywords["epsilon"] == "config.train.weight_factor_epsilon"
    current = torch.tensor([[0., 0.], [2., 2.], [0., 0.]], dtype=torch.float64)
    rollout = torch.tensor([[8., 8.], [0., 0.], [0., 0.]], dtype=torch.float64)
    w, _, _ = _compute_weight_factor(
        current, torch.zeros_like(current), rollout,
        torch.tensor([energy.min(), energy.min(), energy.max()]), sigma=torch.ones(3),
        offset_coefficient=config.train.offset_coefficient,
        residual_threshold=config.train.residual_threshold,
        epsilon=config.train.weight_factor_epsilon,
    )
    # Non-default lambda, offset coefficient and epsilon control three samples.
    torch.testing.assert_close(w.flatten(), torch.tensor([16., math.sqrt(40.), .125], dtype=torch.float64))
