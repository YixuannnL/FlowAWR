"""CPU checks of the real correction-clipping loss; no model downloads or GPUs."""

import ast
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
from absl import flags
from ml_collections import config_flags

from config.awr_correction_clip import sd3_pickscore
from flow_grpo import pickscore_logging
from flow_grpo.stat_tracking import PerPromptStatTracker


ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "scripts/train_awr_sd3_correction_clip.py"
BASE_TRAINER = ROOT / "scripts/train_awr_sd3.py"
LAUNCHER = ROOT / "scripts/launchers/run_flowawr_sd3_pickscore_correction_clip_8gpu.sh"
TREE = ast.parse(TRAINER.read_text())

# Load the actual standalone functions without importing the SD3 runtime.
definitions = [node for node in TREE.body if isinstance(node, ast.FunctionDef)
               and (node.name in {"clip_correction_rms", "validate_correction_clip_config"}
                    or node.name.startswith("_validate"))]
HELPERS = {"torch": torch, "math": math, "np": np}
exec(compile(ast.Module(body=definitions, type_ignores=[]), str(TRAINER), "exec"), HELPERS)
clip_correction_rms = HELPERS["clip_correction_rms"]


def _rms(value):
    return value.square().mean(tuple(range(1, value.ndim)), keepdim=True).sqrt()


def _loss_prefix(path):
    branches = [node for node in ast.walk(ast.parse(path.read_text()))
                if isinstance(node, ast.If)
                and ast.unparse(node.test) == "not config.train.energy_mode"
                and any(isinstance(child, ast.Assign)
                        and any(isinstance(target, ast.Name) and target.id == "target_x0"
                                for target in child.targets)
                        for child in ast.walk(ast.Module(body=node.orelse, type_ignores=[])))]
    assert len(branches) == 1
    prefix = []
    for node in branches[0].orelse:
        if isinstance(node, ast.If) and "loss_mask" in ast.unparse(node.test):
            break
        prefix.append(node)
    return ast.Module(body=prefix, type_ignores=[])


def _run_loss(path=TRAINER, *, threshold=1., enabled=True, timestep=.8):
    config = sd3_pickscore()
    config.train.correction_rms_max = threshold
    config.train.correction_clip_enabled = enabled
    # Positive/negative extremes, a large energy with a small residual, and zero.
    x0 = torch.tensor([[[[3., -3.]]], [[[3., -3.]]],
                       [[[.05, -.05]]], [[[0., 0.]]]])
    current = torch.tensor([[[[.2, .1]]]]).repeat(4, 1, 1, 1)
    t = torch.full((4, 1, 1, 1), timestep)
    forward = (-current / t).requires_grad_()
    namespace = dict(HELPERS, config=config, x0=x0, t_expanded=t,
                     xt=torch.zeros_like(x0), old_prediction=torch.zeros_like(x0),
                     forward_prediction=forward,
                     cur_energies=torch.tensor([.8, -.8, 8., 0.], dtype=torch.float64),
                     loss_terms={})
    exec(compile(_loss_prefix(path), str(path), "exec"), namespace)
    namespace["ori_policy_loss"].sum().backward()
    return namespace


def test_actual_loss_clips_both_sides_per_sample_and_preserves_small_products():
    result = _run_loss()
    expected_target = torch.tensor([[[[1., -1.]]], [[[-1., 1.]]],
                                    [[[.4, -.4]]], [[[0., 0.]]]], dtype=torch.float64)
    torch.testing.assert_close(result["target_x0"], expected_target)
    assert not result["target_x0"].requires_grad
    # b=8 is preserved because the product RMS is only .4: no separate energy cap.
    assert result["cur_energies"][2].item() == 8.
    torch.testing.assert_close(_rms(result["target_x0"]),
                               torch.tensor([1., 1., .4, 0.], dtype=torch.float64).view(4, 1, 1, 1))


def test_equal_x0_corrections_at_different_timesteps_have_the_same_cap():
    low_t, high_t = _run_loss(timestep=.2), _run_loss(timestep=.8)
    torch.testing.assert_close(low_t["target_x0"], high_t["target_x0"], rtol=0, atol=0)
    # Also run another sample alone to exclude dependence on batch companions.
    q = torch.tensor([[3., -3.], [.2, -.2]])
    together, _, _ = clip_correction_rms(q, 1.)
    for i in range(2):
        alone, _, _ = clip_correction_rms(q[i:i + 1], 1.)
        torch.testing.assert_close(alone, together[i:i + 1], rtol=0, atol=0)


def test_original_denominator_and_expected_policy_gradient_are_preserved():
    result = _run_loss()
    current, target = result["x0_prediction"], result["target_x0"]
    denominator = (current.detach().double() - result["x0"].double()).abs().mean(
        (1, 2, 3), keepdim=True).clamp_min(1e-5)
    torch.testing.assert_close(result["weight_factor"], denominator, rtol=0, atol=0)
    assert not result["weight_factor"].requires_grad
    expected = -2 * result["t_expanded"] * (current.detach() - target)
    expected /= current[0].numel() * denominator
    torch.testing.assert_close(result["forward_prediction"].grad, expected.float())


@pytest.mark.parametrize("settings", [{"enabled": False}, {"threshold": 10.}])
def test_disabled_or_untriggered_loss_matches_original_with_the_same_raw_energy(settings):
    baseline, variant = _run_loss(BASE_TRAINER), _run_loss(**settings)
    for key in ("target_x0", "weight_factor", "ori_policy_loss"):
        torch.testing.assert_close(variant[key], baseline[key], rtol=0, atol=0)
    torch.testing.assert_close(variant["forward_prediction"].grad,
                               baseline["forward_prediction"].grad, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_rms_clipping_direction_zero_boundary_and_dtype(dtype):
    q = torch.tensor([[3., 4., 0., 0.], [1., -1., 1., -1.], [0., 0., 0., 0.]],
                     dtype=dtype, requires_grad=True)
    clipped, before, scale = clip_correction_rms(q, 1.)
    expected_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    assert all(v.dtype == expected_dtype and not v.requires_grad for v in (clipped, before, scale))
    torch.testing.assert_close(before.flatten(), torch.tensor([2.5, 1., 0.], dtype=expected_dtype))
    torch.testing.assert_close(scale.flatten(), torch.tensor([.4, 1., 1.], dtype=expected_dtype))
    torch.testing.assert_close(clipped[0], q[0].detach().to(expected_dtype) * .4)
    assert torch.equal(clipped[1:], q[1:].detach().to(expected_dtype))


@pytest.mark.parametrize("dtype,magnitude", [(torch.float16, 2000.),
                                            (torch.float32, 1e20), (torch.float64, 1e200)])
def test_large_finite_correction_does_not_overflow_or_collapse_to_zero(dtype, magnitude):
    q = torch.tensor([[1., -2.]], dtype=dtype) * magnitude
    clipped, rms, scale = clip_correction_rms(q, 1.)
    assert all(torch.isfinite(v).all() for v in (clipped, rms, scale))
    assert scale.item() > 0
    torch.testing.assert_close(_rms(clipped), torch.ones_like(rms))
    torch.testing.assert_close(clipped[0, 1], -2 * clipped[0, 0])


@pytest.mark.parametrize("threshold", [0., -1., float("nan"), float("inf")])
def test_invalid_threshold_is_rejected(threshold):
    config = sd3_pickscore()
    config.train.correction_rms_max = threshold
    with pytest.raises(ValueError):
        HELPERS["validate_correction_clip_config"](config)


def test_real_tracker_constructor_disables_only_scalar_energy_clipping():
    calls = [node for node in ast.walk(TREE) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "PerPromptStatTracker"
             and any(keyword.arg == "energy_min" for keyword in node.keywords)]
    assert len(calls) == 1
    tracker = eval(compile(ast.Expression(calls[0]), str(TRAINER), "eval"),
                   {"PerPromptStatTracker": PerPromptStatTracker, "config": sd3_pickscore(),
                    "np": np, "float": float})
    diagnostics = {}
    _, energies, _ = tracker.update(["same"] * 24, [100.] + [0.] * 23,
                                    exp=True, diagnostics=diagnostics)
    assert energies.max() > 3.
    np.testing.assert_array_equal(energies, diagnostics["energy_before_clip"])
    assert tracker.adv_clip_max == sd3_pickscore().train.adv_clip_max


def test_cli_can_change_threshold_and_disable_clipping(monkeypatch):
    monkeypatch.chdir(ROOT)
    config_path = "config/awr_correction_clip.py:sd3_pickscore"
    argv = ["test", f"--config={config_path}", "--config.train.correction_rms_max=.75",
            "--config.train.correction_clip_enabled=False"]
    flag_values = flags.FlagValues()
    holder = config_flags.DEFINE_config_file("config", config_path,
                                             flag_values=flag_values, sys_argv=argv)
    flag_values(argv)
    config = holder.value
    assert config.train.correction_rms_max == .75
    assert config.train.correction_clip_enabled is False
    HELPERS["validate_correction_clip_config"](config)
    q = torch.tensor([[3., 4.]])
    clipped, _, _ = clip_correction_rms(q, config.train.correction_rms_max,
                                       enabled=config.train.correction_clip_enabled)
    assert torch.equal(clipped, q)
    config.train.correction_clip_enabled = True
    clipped, _, _ = clip_correction_rms(q, config.train.correction_rms_max,
                                       enabled=config.train.correction_clip_enabled)
    torch.testing.assert_close(_rms(clipped), torch.tensor([[.75]]))


def test_actual_raw_logging_call_records_effective_energy_and_reconstructs_loss(tmp_path):
    config = sd3_pickscore()
    config.save_dir = str(tmp_path)
    log = pickscore_logging.create_raw_logger(config, 0, 1)
    result = _run_loss()
    calls = [node for node in ast.walk(TREE) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "log_training_batch"]
    assert len(calls) == 1
    namespace = dict(result, raw_logger=log, log_training_batch=pickscore_logging.log_training_batch,
                     train_sample_batch={"raw_sample_index": torch.arange(4)},
                     t=result["t_expanded"].flatten(), epoch=7, global_step=8, inner_epoch=0,
                     i=0, j_idx=2)
    eval(compile(ast.Expression(calls[0]), str(TRAINER), "eval"), namespace)
    pickscore_logging.flush_training(log)
    rows = [json.loads(line) for line in (log["path"] / "training_terms.jsonl").read_text().splitlines()]
    expected_energy = [1 / 3, -1 / 3, 8., 0.]
    assert len(rows) == 4
    for i, row in enumerate(rows):
        assert row["energy_used"] == pytest.approx(expected_energy[i], rel=1e-6)
        b = row["energy_used"]
        reconstructed = (row["delta_l2"] ** 2 + b * b * row["residual_l2"] ** 2
                         - 2 * b * row["delta_residual_dot"])
        reconstructed /= result["x0"][i].numel() * row["weight_factor"]
        assert reconstructed == pytest.approx(row["policy_loss"], rel=2e-6, abs=2e-6)


def test_launcher_forwards_overrides_without_starting_training(tmp_path):
    fake_root, conda, model, runtime = [tmp_path / name for name in ("flowawr", "conda", "model", "runtime")]
    for path in [fake_root, conda / "bin", model, runtime / "swanlab"]:
        path.mkdir(parents=True)
    (conda / "bin/activate").write_text(": # test activation stub\n")
    captured, torchrun = tmp_path / "args.json", tmp_path / "torchrun"
    torchrun.write_text(f"#!{sys.executable}\nimport json, os, sys\nfrom pathlib import Path\n"
                       "Path(os.environ['CAPTURE_ARGS']).write_text(json.dumps(sys.argv[1:]))\n")
    torchrun.chmod(0o755)
    env = dict(os.environ, FLOWAWR_DIR=str(fake_root), CONDA_ROOT=str(conda), SD3_MODEL=str(model),
               SWANLAB_RUNTIME=str(runtime), TORCHRUN=str(torchrun), LOG_ROOT=str(tmp_path / "logs"),
               SAVE_DIR=str(tmp_path / "save"), CAPTURE_ARGS=str(captured))
    overrides = ["--config.train.correction_rms_max=.75", "--config.train.correction_clip_enabled=False"]
    subprocess.run(["bash", str(LAUNCHER), *overrides], env=env, check=True,
                   capture_output=True, text=True, timeout=15)
    args = json.loads(captured.read_text())
    assert "scripts/train_awr_sd3_correction_clip.py" in args
    assert "--config=config/awr_correction_clip.py:sd3_pickscore" in args
    assert args[-len(overrides):] == overrides
