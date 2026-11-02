"""A4.7 CPU behavior checks using the actual trainer loss and raw logger."""
import ast
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from absl import flags
from ml_collections import config_flags

from config.awr_negative_residual_clip import sd3_pickscore
from config.awr_residual_clip import sd3_pickscore as a44_config
from flow_grpo import pickscore_logging as base_raw_logging
from flow_grpo.stat_tracking import PerPromptStatTracker

ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "scripts/train_awr_sd3_negative_residual_clip.py"
ORIGINAL = ROOT / "scripts/train_awr_sd3.py"
LAUNCHER = ROOT / "scripts/launchers/run_flowawr_sd3_pickscore_negative_residual_clip_8gpu.sh"
NAMES = {"_validate_max_rms", "validate_negative_residual_clip_config", "clip_negative_residual_rms",
         "create_raw_logger", "log_training_batch", "flush_training", "log_update"}
tree = ast.parse(TRAINER.read_text())
nodes = [n for n in tree.body if
         (isinstance(n, ast.FunctionDef) and n.name in NAMES) or
         (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and
           t.id == "RESIDUAL_CLIP_TRAINING_FIELDS" for t in n.targets))]
HELPERS = dict(torch=torch, math=math, json=json, base_raw_logging=base_raw_logging)
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(TRAINER), "exec"), HELPERS)
clip = HELPERS["clip_negative_residual_rms"]
validate = HELPERS["validate_negative_residual_clip_config"]


def rms(x):
    return x.square().mean(tuple(range(1, x.ndim)), keepdim=True).sqrt()


def loss_prefix(path):
    branches = [n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n, ast.If)
                and ast.unparse(n.test) == "not config.train.energy_mode"
                and "target_x0" in ast.unparse(ast.Module(body=n.orelse, type_ignores=[]))]
    assert len(branches) == 1
    out = []
    for n in branches[0].orelse:
        if isinstance(n, ast.If) and "loss_mask" in ast.unparse(n.test):
            break
        out.append(n)
    return ast.Module(body=out, type_ignores=[])


def run_loss(path=TRAINER, enabled=True):
    config = sd3_pickscore()
    config.train.residual_clip_enabled = enabled
    x0 = torch.tensor([[[[3., -4.]]], [[[3., -4.]]], [[[3., -4.]]], [[[.2, -.2]]]])
    t = torch.tensor([.2, .8, .6, .4]).view(-1, 1, 1, 1)
    current = torch.tensor([[[[.2, .1]]]]).repeat(4, 1, 1, 1)
    velocity = (-current / t).requires_grad_()
    ns = dict(HELPERS, config=config, x0=x0, t_expanded=t, xt=torch.zeros_like(x0),
              old_prediction=torch.zeros_like(x0), forward_prediction=velocity,
              cur_energies=torch.tensor([-.5, 5., 0., -.1], dtype=torch.float64), loss_terms={})
    exec(compile(loss_prefix(path), str(path), "exec"), ns)
    ns["ori_policy_loss"].sum().backward()
    return ns


def test_only_negative_samples_clip_and_norm_excludes_energy_magnitude():
    r = torch.tensor([[3., -4.], [3., -4.], [3., -4.], [3., -4.], [.6, -.6], [0., 0.]])
    e = torch.tensor([-.1, -.5, 5., 0., -.2, -.5])
    used, before, scale = clip(r, e, .6)
    torch.testing.assert_close(rms(used[:2]), torch.full((2, 1), .6))
    assert scale[0].item() == scale[1].item()
    assert torch.equal(used[2:], r[2:]) and torch.equal(scale[2:], torch.ones_like(scale[2:]))
    torch.testing.assert_close(rms(e[:2, None] * used[:2]), torch.tensor([[.06], [.3]]))
    torch.testing.assert_close(before, rms(r))
    assert all(not z.requires_grad for z in (used, before, scale))


@pytest.mark.parametrize("dtype,magnitude", [(torch.float16, 2000.), (torch.bfloat16, 2000.),
                                             (torch.float32, 1e20), (torch.float64, 1e200)])
def test_large_finite_residual_preserves_positive_side_and_negative_direction(dtype, magnitude):
    r = torch.tensor([[1., -2.], [1., -2.], [1., -2.]], dtype=dtype) * magnitude
    original = r.clone()
    used, before, scale = clip(r, torch.tensor([-.5, 5., 0.]), .6)
    expected_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    assert used.dtype == before.dtype == scale.dtype == expected_dtype
    assert torch.equal(r, original)
    assert all(torch.isfinite(z).all() for z in (used, before, scale))
    assert torch.equal(used[1:], r[1:].to(expected_dtype))
    torch.testing.assert_close(rms(used[:1]), torch.tensor([[.6]], dtype=expected_dtype))
    assert torch.equal(torch.sign(used[0]), torch.sign(r[0]))


def test_actual_loss_selects_negative_side_and_preserves_original_denominator_gradient():
    out = run_loss()
    assert not out["target_x0"].requires_grad
    torch.testing.assert_close(rms(out["residual_used"][:1]), torch.tensor([[[[.6]]]]))
    assert torch.equal(out["residual_used"][1:], out["x0"][1:])
    torch.testing.assert_close(out["target_x0"], out["pred_x0_old"] +
                               out["cur_energies"].view(-1, 1, 1, 1) * out["residual_used"])
    denominator = (out["x0_prediction"].detach().double() - out["x0"].double()).abs().mean(
        (1, 2, 3), keepdim=True).clamp_min(1e-5)
    assert not out["weight_factor"].requires_grad
    torch.testing.assert_close(out["weight_factor"], denominator, rtol=0, atol=0)
    expected = -2 * out["t_expanded"] * (out["x0_prediction"].detach() - out["target_x0"])
    expected /= out["x0"][0].numel() * denominator
    torch.testing.assert_close(out["forward_prediction"].grad, expected.float())
    base = run_loss(ORIGINAL)
    for key in ("target_x0", "ori_policy_loss"):
        torch.testing.assert_close(out[key][1:], base[key][1:], rtol=0, atol=0)
    torch.testing.assert_close(out["forward_prediction"].grad[1:],
                               base["forward_prediction"].grad[1:], rtol=0, atol=0)


def test_disabled_actual_loss_matches_original_exactly():
    out, base = run_loss(enabled=False), run_loss(ORIGINAL)
    for key in ("target_x0", "weight_factor", "ori_policy_loss"):
        torch.testing.assert_close(out[key], base[key], rtol=0, atol=0)
    torch.testing.assert_close(out["forward_prediction"].grad, base["forward_prediction"].grad, rtol=0, atol=0)


def test_config_matches_a44_except_requested_norm_threshold_and_identity():
    config, base = sd3_pickscore(), a44_config()
    validate(config)
    assert config.train.residual_rms_max == .6 and config.train.residual_clip_enabled is True
    assert (config.train.energy_min, config.train.energy_max) == (-.5, 5.)
    assert config.run_name == "flowawr_sd3_pickscore_negative_residual_clip"
    config.train.residual_rms_max = base.train.residual_rms_max
    config.run_name, config.save_dir = base.run_name, base.save_dir
    assert config.to_dict() == base.to_dict()


def test_independent_energy_clip_preserves_raw_sign_for_side_selection():
    c = sd3_pickscore()
    tracker = PerPromptStatTracker(adv_clip_max=c.train.adv_clip_max,
                                  energy_min=c.train.energy_min, energy_max=c.train.energy_max)
    d = {}
    _, used, _ = tracker.update(["same"] * 24, [100.] + [0.] * 23, exp=True, diagnostics=d)
    raw = torch.from_numpy(d["energy_before_clip"])
    used = torch.from_numpy(used)
    assert raw.max().item() > 5 and raw.min().item() < -.5
    assert used.max().item() == 5 and used.min().item() == -.5
    assert torch.equal(raw < 0, used < 0)
    r = torch.ones((24, 3))
    actual = clip(r, used, .6)[0]
    expected = clip(r, raw, .6)[0]
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("changes", [{"energy_min": 0.}, {"energy_max": 0.},
                                      {"energy_min": float("nan")}, {"residual_rms_max": 0.},
                                      {"residual_rms_max": float("inf")}, {"energy_mode": False}])
def test_configuration_rejects_sign_ambiguity_before_distributed_setup(changes):
    c = sd3_pickscore()
    for k, v in changes.items():setattr(c.train, k, v)
    with pytest.raises(ValueError):validate(c)
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    calls = {n.func.id:n.lineno for n in ast.walk(main) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id in
             {"validate_negative_residual_clip_config", "setup_distributed"}}
    assert calls["validate_negative_residual_clip_config"] < calls["setup_distributed"]


def test_cli_overrides_reach_real_config_and_helper(monkeypatch):
    monkeypatch.chdir(ROOT)
    path = "config/awr_negative_residual_clip.py:sd3_pickscore"
    argv = ["test", f"--config={path}", "--config.train.residual_rms_max=.4"]
    fv = flags.FlagValues()
    holder = config_flags.DEFINE_config_file("config", path, flag_values=fv, sys_argv=argv)
    fv(argv)
    validate(holder.value)
    used, _, _ = clip(torch.tensor([[2., -2.], [2., -2.]]), torch.tensor([-.5, 5.]),
                      holder.value.train.residual_rms_max)
    torch.testing.assert_close(rms(used), torch.tensor([[.4], [2.]]))


def test_actual_raw_logs_distinguish_side_and_reconstruct_policy_loss(tmp_path):
    c = sd3_pickscore(); c.save_dir = str(tmp_path)
    log = HELPERS["create_raw_logger"](c, rank=0, world_size=1)
    schema = json.loads((log["path"].parent / "residual_clip_schema.json").read_text())
    assert schema["variant"] == "negative_unweighted_x0_residual_rms_clip"
    assert schema["residual_clip_side"] == "negative_only"
    out = run_loss()
    HELPERS["log_training_batch"](log, sample_index=torch.arange(4), t=out["t_expanded"].flatten(),
        energy_used=out["cur_energies"], x0=out["x0"], pred_x0_old=out["pred_x0_old"],
        pred_x0_current=out["x0_prediction"], policy_loss=out["ori_policy_loss"],
        weight_factor=out["weight_factor"], epoch=0, global_step=0, inner_epoch=0,
        microbatch=0, timestep_index=0, residual_used=out["residual_used"],
        residual_rms_before=out["residual_rms_before"], residual_clip_scale=out["residual_clip_scale"],
        residual_rms_max=.6, residual_clip_enabled=True)
    HELPERS["log_update"](log, epoch=0, global_step=0, inner_epoch=0,
        grad_norm_before_clip=torch.tensor(.4), max_grad_norm=1., scaler_scale=65536., optimizer_step_skipped=False)
    rows = list(map(json.loads, (log["path"] / "training_terms.jsonl").read_text().splitlines()))
    assert len(rows)==4
    for i,r in enumerate(rows):
        assert r["residual_clip_side"] == "negative_only"
        assert r["residual_clip_eligible"] == (i in (0,3))
        assert r["residual_clipped"] == (i==0)
        assert r["energy_used"] == out["cur_energies"][i].item()
        if i in (1,2):assert r["residual_l2"] == r["residual_l2_after_clip"] and r["residual_clip_scale"] == 1.
        b = r["energy_effective"]
        reconstructed = (r["delta_l2"]**2 + b*b*r["residual_l2"]**2 - 2*b*r["delta_residual_dot"])
        reconstructed /= r["latent_numel"] * r["weight_factor"]
        assert reconstructed == pytest.approx(r["policy_loss"], rel=2e-6, abs=2e-6)


def test_launcher_forwards_cli_without_starting_training(tmp_path):
    root, conda, model, runtime = [tmp_path/n for n in ("root","conda","model","runtime")]
    for p in (root, conda/"bin", model, runtime/"swanlab"):p.mkdir(parents=True)
    (conda/"bin/activate").write_text(":\n")
    capture = tmp_path/"args.json"; runner=tmp_path/"torchrun"
    runner.write_text(f"#!{sys.executable}\nimport json,os,sys\nfrom pathlib import Path\nPath(os.environ['CAPTURE_ARGS']).write_text(json.dumps(sys.argv[1:]))\n")
    runner.chmod(0o755)
    env=dict(os.environ,FLOWAWR_DIR=str(root),CONDA_ROOT=str(conda),SD3_MODEL=str(model),
             SWANLAB_RUNTIME=str(runtime),TORCHRUN=str(runner),CAPTURE_ARGS=str(capture))
    arg="--config.train.residual_rms_max=.4"
    subprocess.run(["bash",str(LAUNCHER),arg],env=env,check=True,capture_output=True,text=True,timeout=15)
    args=json.loads(capture.read_text())
    assert "scripts/train_awr_sd3_negative_residual_clip.py" in args
    assert "--config=config/awr_negative_residual_clip.py:sd3_pickscore" in args
    assert "--nproc_per_node=8" in args and args[-1]==arg
