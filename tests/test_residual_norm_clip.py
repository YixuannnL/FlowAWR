"""CPU checks for independent x0 residual clipping; no SD3 runtime or downloads."""

import ast
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from absl import flags
from ml_collections import config_flags

from config.awr_residual_clip import sd3_pickscore
from flow_grpo import pickscore_logging as base_raw_logging
from flow_grpo.stat_tracking import PerPromptStatTracker


ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "scripts/train_awr_sd3_residual_clip.py"
BASE_TRAINER = ROOT / "scripts/train_awr_sd3.py"
LAUNCHER = ROOT / "scripts/launchers/run_flowawr_sd3_pickscore_residual_clip_8gpu.sh"


# Extract the trainer's real standalone definitions without importing SD3,
# its model libraries, or distributed initialization.
_helper_names = {"_validate_max_rms", "validate_residual_clip_config", "clip_residual_rms"}
_logger_names = {"create_raw_logger", "log_training_batch", "flush_training", "log_update"}
_definitions = [node for node in ast.parse(TRAINER.read_text()).body
                if ((isinstance(node, ast.FunctionDef)
                     and node.name in _helper_names | _logger_names)
                    or (isinstance(node, ast.Assign)
                        and any(isinstance(target, ast.Name)
                                and target.id == "RESIDUAL_CLIP_TRAINING_FIELDS"
                                for target in node.targets)))]
assert {node.name for node in _definitions if isinstance(node, ast.FunctionDef)} == (
    _helper_names | _logger_names)
assert any(isinstance(node, ast.Assign) for node in _definitions)
_namespace = {"__name__": __name__, "torch": torch, "math": math, "json": json,
              "base_raw_logging": base_raw_logging}
exec(compile(ast.Module(body=_definitions, type_ignores=[]), str(TRAINER), "exec"), _namespace)
clip_residual_rms = _namespace["clip_residual_rms"]
validate_residual_clip_config = _namespace["validate_residual_clip_config"]
residual_clip_logging = SimpleNamespace(**{name: _namespace[name] for name in _logger_names})


def test_residual_clip_helpers_remain_inside_the_independent_trainer():
    removed_modules = {"flow_grpo.residual_norm_clip", "flow_grpo.residual_clip_logging"}
    dependencies = set()
    for node in ast.walk(ast.parse(TRAINER.read_text())):
        if isinstance(node, ast.ImportFrom):
            dependencies.add(node.module)
            if node.module == "flow_grpo":
                dependencies.update(f"flow_grpo.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Import):
            dependencies.update(alias.name for alias in node.names)
    assert dependencies.isdisjoint(removed_modules)
    assert not (ROOT / "flow_grpo/residual_norm_clip.py").exists()
    assert not (ROOT / "flow_grpo/residual_clip_logging.py").exists()


def _rms(value):
    return value.square().mean(tuple(range(1, value.ndim)), keepdim=True).sqrt()


def _energy_prefix(path):
    """Run the actual loss construction, without importing the training runtime."""
    tree = ast.parse(path.read_text())
    branches = [node for node in ast.walk(tree)
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
    assert any(isinstance(node, ast.Assign)
               and any(isinstance(target, ast.Name) and target.id == "ori_policy_loss"
                       for target in node.targets) for node in prefix)
    return ast.Module(body=prefix, type_ignores=[])


def _run_loss_prefix(path, *, enabled=True, threshold=0.75):
    config = sd3_pickscore()
    config.train.residual_clip_enabled = enabled
    config.train.residual_rms_max = threshold
    # Identical large residuals deliberately have different b and t.
    x0 = torch.tensor([[[[3., 4.]]], [[[3., 4.]]], [[[.25, -.25]]]])
    t = torch.tensor([.2, .8, .6]).view(3, 1, 1, 1)
    current = torch.tensor([[[[2., 1.]]], [[[2., 1.]]], [[[.125, -.125]]]])
    velocity = (-current / t).requires_grad_()
    namespace = {
        "torch": torch, "clip_residual_rms": clip_residual_rms,
        "config": config, "x0": x0, "t_expanded": t,
        "xt": torch.zeros_like(x0), "old_prediction": torch.zeros_like(x0),
        "forward_prediction": velocity, "cur_energies": torch.tensor([-.5, 5., 0.]),
        "loss_terms": {},
    }
    exec(compile(_energy_prefix(path), str(path), "exec"), namespace)
    namespace["ori_policy_loss"].sum().backward()
    return namespace


def test_rms_clip_is_radial_and_independent_per_sample():
    residual = torch.tensor([[[[3., 4.], [0., 0.]]],
                             [[[.25, -.25], [.25, -.25]]],
                             [[[0., 0.], [0., 0.]]]])
    clipped, before, scale = clip_residual_rms(residual, 0.75)
    assert before.shape == scale.shape == (3, 1, 1, 1)
    # RMS is 2.5 (L2 is 5), so the first sample must use scale 0.3.
    torch.testing.assert_close(before.flatten(), torch.tensor([2.5, .25, 0.]))
    torch.testing.assert_close(scale.flatten(), torch.tensor([.3, 1., 1.]))
    torch.testing.assert_close(clipped[0], residual[0] * .3)
    assert torch.equal(clipped[1:], residual[1:])
    assert torch.all(_rms(clipped) <= .75 + 1e-7)


def test_unclipped_boundary_and_zero_are_exactly_preserved():
    residual = torch.tensor([[.75, -.75], [.125, -.25], [0., 0.]],
                            requires_grad=True)
    clipped, before, scale = clip_residual_rms(residual, .75)
    assert torch.equal(clipped, residual)
    assert torch.equal(scale, torch.ones_like(scale))
    assert before[-1].item() == 0.
    assert all(not value.requires_grad for value in (clipped, before, scale))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_dtype_promotion_and_target_detachment(dtype):
    residual = torch.tensor([[3., 4.], [0., 0.]], dtype=dtype, requires_grad=True)
    clipped, before, scale = clip_residual_rms(residual, .75)
    expected_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    assert all(value.dtype == expected_dtype for value in (clipped, before, scale))
    assert all(not value.requires_grad for value in (clipped, before, scale))
    assert all(value.grad_fn is None for value in (clipped, before, scale))
    expected_rms = torch.tensor([[math.sqrt(12.5)], [0.]], dtype=expected_dtype)
    torch.testing.assert_close(before, expected_rms)
    torch.testing.assert_close(_rms(clipped)[0], torch.tensor([.75], dtype=expected_dtype))
    assert torch.equal(clipped[1], torch.zeros_like(clipped[1]))
    assert scale[1].item() == 1.


def test_fp16_reduction_does_not_overflow_before_clipping():
    residual = torch.full((2, 1, 2, 2), 2000., dtype=torch.float16)
    clipped, before, scale = clip_residual_rms(residual, .75)
    assert all(torch.isfinite(value).all() for value in (clipped, before, scale))
    torch.testing.assert_close(before, torch.full((2, 1, 1, 1), 2000.))
    torch.testing.assert_close(clipped, torch.full((2, 1, 2, 2), .75))


@pytest.mark.parametrize("dtype,magnitude", [(torch.float32, 1e20), (torch.float64, 1e200)])
def test_large_finite_residual_is_clipped_without_overflow_or_erasure(dtype, magnitude):
    residual = torch.tensor([[1., -2.], [1., 1.]], dtype=dtype) * magnitude
    original = residual.clone()
    clipped, before, scale = clip_residual_rms(residual, .75)
    assert torch.equal(residual, original)
    assert all(torch.isfinite(value).all() for value in (clipped, before, scale))
    assert torch.all(scale > 0) and torch.all(clipped != 0)
    expected_before = torch.tensor([[math.sqrt(2.5) * magnitude], [magnitude]], dtype=dtype)
    expected_clipped = torch.tensor([[.75 / math.sqrt(2.5), -1.5 / math.sqrt(2.5)],
                                     [.75, .75]], dtype=dtype)
    torch.testing.assert_close(before, expected_before)
    torch.testing.assert_close(clipped, expected_clipped)
    torch.testing.assert_close(_rms(clipped), torch.full((2, 1), .75, dtype=dtype))


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_residual_remains_detectable(nonfinite):
    clipped, before, _ = clip_residual_rms(torch.tensor([[nonfinite, 1.]]), .75)
    assert not torch.isfinite(before).all()
    assert not torch.isfinite(clipped).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_disabled_clipping_preserves_values_and_still_reports_original_rms(dtype):
    residual = torch.tensor([[3., 4.], [.125, -.25]], dtype=dtype, requires_grad=True)
    clipped, before, scale = clip_residual_rms(residual, .75, enabled=False)
    expected = residual.detach().double() if dtype == torch.float64 else residual.detach().float()
    assert torch.equal(clipped, expected)
    assert torch.equal(scale, torch.ones_like(scale))
    torch.testing.assert_close(before, _rms(expected))
    assert all(not value.requires_grad for value in (clipped, before, scale))


@pytest.mark.parametrize("threshold", [0., -.75, float("nan"), float("inf"), -float("inf")])
def test_invalid_threshold_rejected(threshold):
    with pytest.raises(ValueError):
        clip_residual_rms(torch.ones((2, 4)), threshold)


def test_actual_trainer_clips_unweighted_x0_residual_and_preserves_base_gradient():
    output = _run_loss_prefix(TRAINER)
    residual = output["x0"] - output["pred_x0_old"]
    scale = torch.minimum(torch.ones((3, 1, 1, 1)), .75 / _rms(residual))
    target = output["pred_x0_old"] + output["energies_expanded"] * residual * scale
    torch.testing.assert_close(output["target_x0"], target)
    assert not output["target_x0"].requires_grad
    assert scale[0].item() == scale[1].item()  # Neither b nor t enters the clipping norm.
    assert torch.equal(output["target_x0"][2], output["pred_x0_old"][2])
    denominator = (output["x0_prediction"].detach().double() - output["x0"].double()).abs().mean(
        (1, 2, 3), keepdim=True).clamp_min(1e-5)
    torch.testing.assert_close(output["weight_factor"], denominator, rtol=0, atol=0)
    assert not output["weight_factor"].requires_grad
    expected_gradient = -2 * output["t_expanded"] * (output["x0_prediction"].detach() - target)
    expected_gradient = expected_gradient / (residual[0].numel() * denominator)
    torch.testing.assert_close(output["forward_prediction"].grad, expected_gradient.float())


def test_disabled_trainer_matches_original_target_loss_and_gradient_exactly():
    baseline = _run_loss_prefix(BASE_TRAINER)
    disabled = _run_loss_prefix(TRAINER, enabled=False)
    for key in ["target_x0", "weight_factor", "ori_policy_loss"]:
        torch.testing.assert_close(disabled[key], baseline[key], rtol=0, atol=0)
    torch.testing.assert_close(disabled["forward_prediction"].grad,
                               baseline["forward_prediction"].grad, rtol=0, atol=0)


def test_original_weight_factor_expression_is_unchanged_and_detached():
    expressions = []
    for path in (BASE_TRAINER, TRAINER):
        prefix = _energy_prefix(path)
        assignments = [node for node in ast.walk(prefix) if isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Name) and target.id == "weight_factor"
                               for target in node.targets)]
        assert len(assignments) == 1
        assignment = assignments[0]
        expressions.append(ast.dump(assignment.value, include_attributes=False))
        assert any(isinstance(node, ast.With)
                   and any(ast.unparse(item.context_expr) == "torch.no_grad()" for item in node.items)
                   and assignment in list(ast.walk(node)) for node in ast.walk(prefix))
    assert expressions[0] == expressions[1]


def test_cli_overrides_reach_real_config_and_residual_helper(monkeypatch):
    monkeypatch.chdir(ROOT)
    default_config = "config/awr_residual_clip.py:sd3_pickscore"
    argv = ["test", f"--config={default_config}",
            "--config.train.residual_rms_max=0.8",
            "--config.train.residual_clip_enabled=False",
            "--config.train.energy_min=-0.25", "--config.train.energy_max=4"]
    flag_values = flags.FlagValues()
    holder = config_flags.DEFINE_config_file("config", default_config,
                                             flag_values=flag_values, sys_argv=argv)
    flag_values(argv)
    config = holder.value
    assert config.train.residual_rms_max == .8
    assert config.train.residual_clip_enabled is False
    assert (config.train.energy_min, config.train.energy_max) == (-.25, 4.)
    residual = torch.tensor([[3., 4.]])
    clipped, _, scale = clip_residual_rms(residual, config.train.residual_rms_max,
                                         enabled=config.train.residual_clip_enabled)
    assert torch.equal(clipped, residual) and scale.item() == 1.
    config.train.residual_clip_enabled = True
    clipped, _, _ = clip_residual_rms(residual, config.train.residual_rms_max,
                                     enabled=config.train.residual_clip_enabled)
    torch.testing.assert_close(_rms(clipped), torch.tensor([[.8]]))


def test_energy_keeps_its_independent_clip_limits():
    config = sd3_pickscore()
    assert config.train.residual_rms_max == .75
    assert config.train.residual_clip_enabled is True
    assert (config.train.energy_min, config.train.energy_max) == (-.5, 5.)
    tracker = PerPromptStatTracker(adv_clip_max=config.train.adv_clip_max,
                                  energy_min=config.train.energy_min,
                                  energy_max=config.train.energy_max)
    diagnostics = {}
    _, energy, _ = tracker.update(["same prompt"] * 24, [100.] + [0.] * 23,
                                  exp=True, diagnostics=diagnostics)
    assert diagnostics["energy_before_clip"].max() > 5.
    assert energy.max() == 5. and energy.min() == -.5
    calls = [node for node in ast.walk(ast.parse(TRAINER.read_text()))
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
             and node.func.id == "PerPromptStatTracker"
             and any(keyword.arg == "energy_min" for keyword in node.keywords)]
    assert len(calls) == 1
    kwargs = {keyword.arg: ast.unparse(keyword.value) for keyword in calls[0].keywords}
    assert kwargs["energy_min"] == "config.train.energy_min"
    assert kwargs["energy_max"] == "config.train.energy_max"


def test_launcher_forwards_cli_overrides_without_starting_training(tmp_path):
    fake_root = tmp_path / "flowawr"
    conda = tmp_path / "conda"
    model = tmp_path / "model"
    runtime = tmp_path / "runtime"
    for path in [fake_root, conda / "bin", model, runtime / "swanlab"]:
        path.mkdir(parents=True)
    (conda / "bin/activate").write_text(": # isolated test activation stub\n")
    captured = tmp_path / "torchrun-args.json"
    torchrun = tmp_path / "torchrun"
    torchrun.write_text(f"#!{sys.executable}\nimport json, os, sys\n"
                       "from pathlib import Path\n"
                       "Path(os.environ['CAPTURE_ARGS']).write_text(json.dumps(sys.argv[1:]))\n")
    torchrun.chmod(0o755)
    env = dict(os.environ, FLOWAWR_DIR=str(fake_root), CONDA_ROOT=str(conda),
               SD3_MODEL=str(model), SWANLAB_RUNTIME=str(runtime), TORCHRUN=str(torchrun),
               LOG_ROOT=str(tmp_path / "logs"), SAVE_DIR=str(tmp_path / "save"),
               CAPTURE_ARGS=str(captured))
    overrides = ["--config.train.residual_rms_max=0.8",
                 "--config.train.residual_clip_enabled=False",
                 "--config.train.energy_min=-0.25"]
    subprocess.run(["bash", str(LAUNCHER), *overrides], env=env, check=True,
                   capture_output=True, text=True, timeout=15)
    args = json.loads(captured.read_text())
    assert "scripts/train_awr_sd3_residual_clip.py" in args
    assert "--config=config/awr_residual_clip.py:sd3_pickscore" in args
    assert args[-len(overrides):] == overrides
    assert "--nproc_per_node=8" in args


@pytest.mark.parametrize("changes", [
    {"energy_mode": False},
    {"residual_rms_max": float("nan")},
    {"residual_rms_max": float("inf")},
    {"residual_rms_max": 0.},
    {"energy_min": 5., "energy_max": 5.},
    {"energy_min": float("nan")},
])
def test_invalid_training_configuration_fails_before_distributed_setup(changes):
    config = sd3_pickscore()
    for name, value in changes.items():
        setattr(config.train, name, value)
    with pytest.raises(ValueError):
        validate_residual_clip_config(config)
    main = next(node for node in ast.parse(TRAINER.read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name == "main")
    calls = {node.func.id: node.lineno for node in ast.walk(main)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
             and node.func.id in {"validate_residual_clip_config", "setup_distributed"}}
    assert calls["validate_residual_clip_config"] < calls["setup_distributed"]


def test_raw_logs_keep_original_terms_and_reconstruct_clipped_loss(tmp_path):
    config = sd3_pickscore()
    config.save_dir = str(tmp_path / "run")
    log = residual_clip_logging.create_raw_logger(config, rank=0, world_size=1)
    schema = json.loads((log["path"].parent / "residual_clip_schema.json").read_text())
    assert schema["schema_version"] == 2
    assert schema["variant"] == "unweighted_x0_residual_rms_clip"
    assert "residual_l2_after_clip" in schema["policy_loss_reconstruction"]
    assert {"energy_used", "energy_effective", "residual_clip_scale"}.issubset(schema["training_fields"])

    outputs = [_run_loss_prefix(TRAINER, enabled=enabled) for enabled in (True, False)]
    for index, output in enumerate(outputs):
        residual_clip_logging.log_training_batch(
            log, sample_index=torch.tensor([11, 12, 13]),
            t=output["t_expanded"].flatten(), energy_used=output["cur_energies"],
            x0=output["x0"], pred_x0_old=output["pred_x0_old"],
            pred_x0_current=output["x0_prediction"], policy_loss=output["ori_policy_loss"],
            weight_factor=output["weight_factor"], epoch=7, global_step=8,
            inner_epoch=0, microbatch=index, timestep_index=index,
            residual_used=output["residual_used"],
            residual_rms_before=output["residual_rms_before"],
            residual_clip_scale=output["residual_clip_scale"],
            residual_rms_max=.75, residual_clip_enabled=index == 0,
        )
    training_path = log["path"] / "training_terms.jsonl"
    assert not training_path.exists()  # Records are buffered until the update.
    assert len(log["training"]) == 2
    update = dict(epoch=7, global_step=8, inner_epoch=0,
                  grad_norm_before_clip=torch.tensor(.4), max_grad_norm=1.,
                  scaler_scale=65536., optimizer_step_skipped=False)
    residual_clip_logging.log_update(log, **update)
    assert log["training"] == []
    rows = [json.loads(line) for line in training_path.read_text().splitlines()]
    assert len(rows) == 6

    for index, row in enumerate(rows):
        batch, sample = divmod(index, 3)
        output = outputs[batch]
        raw = (output["x0"] - output["pred_x0_old"])[sample].flatten()
        used = output["residual_used"][sample].flatten()
        delta = (output["x0_prediction"].detach() - output["pred_x0_old"])[sample].flatten()
        assert row["sample_index"] == 11 + sample
        assert row["microbatch"] == row["timestep_index"] == batch
        assert row["training_terms_schema_version"] == 2
        assert row["latent_numel"] == raw.numel()
        assert row["energy_used"] == output["cur_energies"][sample].item()
        assert row["residual_l2"] == raw.norm().item()
        assert row["delta_residual_dot"] == torch.dot(delta, raw).item()
        assert row["residual_l2_after_clip"] == used.norm().item()
        assert row["delta_residual_dot_after_clip"] == torch.dot(delta, used).item()
        assert row["residual_rms_before"] == output["residual_rms_before"][sample].item()
        assert row["residual_rms_after"] == pytest.approx(used.norm().item() / math.sqrt(raw.numel()))
        assert row["residual_clipped"] == (batch == 0 and sample < 2)
        assert row["residual_clip_enabled"] == (batch == 0)
        assert row["energy_effective"] == pytest.approx(row["energy_used"] * row["residual_clip_scale"])
        assert row["policy_loss"] == output["ori_policy_loss"][sample].item()

        b = row["energy_used"]
        reconstructed = (row["delta_l2"] ** 2 + b ** 2 * row["residual_l2_after_clip"] ** 2
                         - 2 * b * row["delta_residual_dot_after_clip"])
        reconstructed /= row["latent_numel"] * row["weight_factor"]
        effective = row["energy_effective"]
        reconstructed_effective = (row["delta_l2"] ** 2 + effective ** 2 * row["residual_l2"] ** 2
                                   - 2 * effective * row["delta_residual_dot"])
        reconstructed_effective /= row["latent_numel"] * row["weight_factor"]
        assert reconstructed == pytest.approx(row["policy_loss"], rel=2e-6, abs=2e-6)
        assert reconstructed_effective == pytest.approx(row["policy_loss"], rel=2e-6, abs=2e-6)

    # Repeated flush/update cannot duplicate rows or route extended columns
    # through the legacy eight-column writer.
    frozen = training_path.read_bytes()
    residual_clip_logging.flush_training(log)
    residual_clip_logging.log_update(log, **{**update, "global_step": 9})
    assert training_path.read_bytes() == frozen
    updates = [json.loads(line) for line in (log["path"] / "updates.jsonl").read_text().splitlines()]
    assert [row["global_step"] for row in updates] == [8, 9]
    assert all(row["grad_norm_before_clip"] == pytest.approx(.4) for row in updates)
