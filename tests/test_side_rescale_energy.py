"""CPU-only numerical and integration checks for separate-side group rescaling.

Run with the existing results environment; no model, CUDA, or training import:
    /tmp/flowawr-results-venv/bin/python -m unittest discover -s tests -p test_side_rescale_energy.py
"""

import ast
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from flow_grpo.stat_tracking import PerPromptStatTracker as OriginalTracker
from flow_grpo.stat_tracking_side_rescale import PerPromptStatTracker


ROOT = Path(__file__).resolve().parents[1]
NEW_TRAINER = ROOT / "scripts/train_awr_sd3_side_rescale.py"
BASE_TRAINER = ROOT / "scripts/train_awr_sd3.py"
TRACKER = ROOT / "flow_grpo/stat_tracking_side_rescale.py"
RAW_FIELDS = {"advantage_raw", "advantage_clipped", "softmax_prob", "energy_before_clip", "energy_used"}


def run_rescaling_block(values, *, energy_min, energy_max):
    """Execute the actual inline clipping replacement without constructing rewards."""
    tree = ast.parse(TRACKER.read_text())
    branches = [n for n in ast.walk(tree) if isinstance(n, ast.If)
                and isinstance(n.test, ast.Name) and n.test.id == "exp"
                and any(isinstance(item, ast.Assign)
                        and any(isinstance(target, ast.Name) and target.id == "energies_clipped"
                                for target in item.targets)
                        for item in n.body)]
    if len(branches) != 1:
        raise AssertionError("Expected one actual energy rescaling block")
    body = branches[0].body
    stop = next(i for i, node in enumerate(body)
                if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Subscript)
                        and ast.unparse(target) == "energies[prompts == prompt]"
                        for target in node.targets))
    start = max(i for i, node in enumerate(body[:stop])
                if isinstance(node, ast.If) and ast.unparse(node.test) == "diagnostics is not None") + 1
    namespace = {"np": np, "self": SimpleNamespace(energy_min=energy_min, energy_max=energy_max),
                 "energies_reranged": np.asarray(values, dtype=np.float64)}
    exec(compile(ast.Module(body=body[start:stop], type_ignores=[]), str(TRACKER), "exec"), namespace)
    return namespace["energies_clipped"]


def expected_group(values, lower, upper):
    """Scalar column-by-column reference, independent of the vectorized tracker."""
    result = np.array(values, copy=True)
    columns = result.reshape(len(result), -1)
    for column in columns.T:
        positive = column > 0
        negative = column < 0
        if positive.any() and column[positive].max() > upper:
            column[positive] *= upper / column[positive].max()
        if negative.any() and column[negative].min() < lower:
            column[negative] *= lower / column[negative].min()
    return result


class SideRescaleMathTests(unittest.TestCase):
    def test_two_sides_use_different_scales_and_preserve_internal_ratios(self):
        original = np.array([-2., -1., -.5, 0., 1., 2., 4., 8.])
        saved = original.copy()
        actual = run_rescaling_block(original, energy_min=-.5, energy_max=3.)
        np.testing.assert_array_equal(actual, [-.5, -.25, -.125, 0., .375, .75, 1.5, 3.])
        np.testing.assert_array_equal(original, saved)
        self.assertFalse(np.array_equal(actual, np.clip(original, -.5, 3.)))
        np.testing.assert_array_equal(actual[original > 0] / original[original > 0], [.375] * 4)
        np.testing.assert_array_equal(actual[original < 0] / original[original < 0], [.25] * 3)

    def test_one_side_trigger_does_not_shrink_the_other_side(self):
        for values, lower, upper, expected in [
            ([-.4, -.2, 1., 4.], -.5, 2., [-.4, -.2, .5, 2.]),
            ([-2., -1., .2, .4], -.5, 1., [-.5, -.25, .2, .4]),
        ]:
            with self.subTest(values=values):
                actual = run_rescaling_block(values, energy_min=lower, energy_max=upper)
                np.testing.assert_array_equal(actual, expected)

    def test_below_or_at_bounds_never_amplifies(self):
        for values in [[-.5, -.1, 0., .3, 3.], [-.1, .1], [0., 0.]]:
            with self.subTest(values=values):
                actual = run_rescaling_block(values, energy_min=-.5, energy_max=3.)
                np.testing.assert_array_equal(actual, values)

    def test_absent_sides_remain_finite(self):
        for values, expected in [([0., 1., 4.], [0., .5, 2.]),
                                 ([-4., -1., 0.], [-2., -.5, 0.]), ([0., 0.], [0., 0.])]:
            with self.subTest(values=values):
                actual = run_rescaling_block(values, energy_min=-2., energy_max=2.)
                self.assertTrue(np.isfinite(actual).all())
                np.testing.assert_array_equal(actual, expected)

    def test_columns_have_independent_group_extrema(self):
        values = np.array([[8., .5, 0.], [4., 1., 0.], [-2., -4., 0.], [-1., -2., 0.]])
        actual = run_rescaling_block(values, energy_min=-1., energy_max=2.)
        np.testing.assert_array_equal(actual, [[2., .5, 0.], [1., 1., 0.], [-1., -1., 0.], [-.5, -.5, 0.]])

    def test_permutation_equivariance_and_sign_preservation(self):
        rng = np.random.default_rng(42)
        values = rng.normal(size=(47, 5)) * 7
        permutation = rng.permutation(len(values))
        actual = run_rescaling_block(values, energy_min=-.4, energy_max=1.7)
        permuted = run_rescaling_block(values[permutation], energy_min=-.4, energy_max=1.7)
        np.testing.assert_array_equal(permuted, actual[permutation])
        np.testing.assert_array_equal(np.sign(actual), np.sign(values))
        self.assertTrue((np.abs(actual) <= np.abs(values)).all())
        self.assertTrue((actual >= -.4 - 1e-15).all())
        self.assertTrue((actual <= 1.7 + 1e-15).all())

    def test_independent_scales_do_not_rebalance_the_group(self):
        values = np.array([-2., -1., 1., 2.])
        self.assertEqual(values.sum(), 0.)
        actual = run_rescaling_block(values, energy_min=-.5, energy_max=1.)
        np.testing.assert_array_equal(actual, [-.5, -.25, .5, 1.])
        self.assertNotEqual(actual.sum(), 0.)

    def test_invalid_bounds_fail_when_rescaling(self):
        for lower, upper in [(0., 3.), (1., 3.), (-3., 0.), (-3., -1.),
                             (float("nan"), 3.), (-3., float("nan")),
                             (-float("inf"), 3.), (-3., float("inf"))]:
            with self.subTest(lower=lower, upper=upper):
                with self.assertRaises(ValueError):
                    run_rescaling_block([-.1, .1], energy_min=lower, energy_max=upper)


class TrackerIntegrationTests(unittest.TestCase):
    def test_original_advantage_softmax_and_stdev_are_unchanged(self):
        prompts = np.array(["a", "b", "a", "b", "a", "b", "a", "b"])
        rewards = np.array([[-2., -.2], [1., 4.], [0., 1.], [2., -1.],
                            [1., 2.], [3., 0.], [8., 9.], [12., 1.]])
        for hard_gating in [False, True]:
            for global_std in [False, True]:
                with self.subTest(hard_gating=hard_gating, global_std=global_std):
                    kwargs = dict(global_std=global_std, adv_clip_max=5., temperature=.7)
                    base = OriginalTracker(**kwargs)
                    new = PerPromptStatTracker(**kwargs, energy_min=-.2, energy_max=.6)
                    old_details, new_details = {}, {}
                    old_adv, _, old_stdev = base.update(prompts, rewards, exp=True, hard_gating=hard_gating, diagnostics=old_details)
                    new_adv, used, new_stdev = new.update(prompts, rewards, exp=True, hard_gating=hard_gating, diagnostics=new_details)
                    np.testing.assert_array_equal(new_adv, old_adv)
                    np.testing.assert_array_equal(new_stdev, old_stdev)
                    for field in ["advantage_raw", "advantage_clipped", "softmax_prob", "energy_before_clip"]:
                        np.testing.assert_array_equal(new_details[field], old_details[field])
                    np.testing.assert_array_equal(new_details["energy_used"], used)
                    for group in np.unique(prompts):
                        mask = prompts == group
                        expected = expected_group(old_details["energy_before_clip"][mask], -.2, .6)
                        np.testing.assert_array_equal(used[mask], expected)
                    self.assertEqual(set(new_details), RAW_FIELDS)
                    for field in new_details.values():
                        self.assertEqual(field.shape, rewards.shape)

    def test_wide_limits_match_the_original_unclipped_energy_exactly(self):
        prompts = ["p"] * 4
        rewards = np.array([[-1., 2.], [0., -3.], [2., 0.], [6., 5.]])
        for hard_gating in [False, True]:
            old = OriginalTracker(energy_min=-1e6, energy_max=1e6)
            new = PerPromptStatTracker(energy_min=-1e6, energy_max=1e6)
            for old_value, new_value in zip(old.update(prompts, rewards, exp=True, hard_gating=hard_gating),
                                            new.update(prompts, rewards, exp=True, hard_gating=hard_gating)):
                np.testing.assert_array_equal(new_value, old_value)

    def test_prompt_groups_are_independent_and_follow_input_order(self):
        prompts = np.array(["a", "b", "a", "b", "a", "b", "a", "b"])
        rewards = np.array([0., -3., .1, -2., .2, -1., 4., 5.])[:, None]
        kwargs = dict(global_std=False, energy_min=-.25, energy_max=.75)
        together = PerPromptStatTracker(**kwargs).update(prompts, rewards, exp=True)
        for prompt in ["a", "b"]:
            mask = prompts == prompt
            separate = PerPromptStatTracker(**kwargs).update(prompts[mask], rewards[mask], exp=True)
            for combined_field, separate_field in zip(together, separate):
                np.testing.assert_array_equal(combined_field[mask], separate_field)
        order = np.array([7, 3, 1, 6, 0, 4, 2, 5])
        shuffled = PerPromptStatTracker(**kwargs).update(prompts[order], rewards[order], exp=True)
        for original_field, shuffled_field in zip(together, shuffled):
            np.testing.assert_allclose(original_field[order], shuffled_field, rtol=1e-14, atol=1e-14)

    def test_equal_rewards_give_zero_energy_and_nonenergy_path_is_preserved(self):
        prompts = ["same"] * 4
        rewards = np.full((4, 2), 2.)
        new = PerPromptStatTracker(energy_min=-.5, energy_max=3.)
        details = {}
        _, energy, _ = new.update(prompts, rewards, exp=True, diagnostics=details)
        np.testing.assert_array_equal(energy, np.zeros_like(rewards))
        values = np.array([[-2.], [0.], [1.], [8.]])
        base, new = OriginalTracker(adv_clip_max=5.), PerPromptStatTracker(adv_clip_max=5., energy_min=-.5, energy_max=3.)
        for factor in [1., 2.]:
            np.testing.assert_array_equal(new.update(prompts, values * factor), base.update(prompts, values * factor))
            # Production clears the prompt tracker after each rollout.
            new.clear()
            base.clear()
        new.clear()
        base.clear()
        self.assertEqual(new.stats, base.stats)

    def test_existing_raw_logger_records_before_and_after_energy(self):
        # These are the unmodified production JSONL functions. Their imports are
        # pure Python; extracting them avoids importing the unrelated CUDA logger.
        logger_path = ROOT / "flow_grpo/pickscore_logging.py"
        names = {"_write_rows", "_sample_id", "_prompt_ids", "log_weight_batch"}
        functions = [n for n in ast.parse(logger_path.read_text()).body
                     if isinstance(n, ast.FunctionDef) and n.name in names]
        self.assertEqual({n.name for n in functions}, names)
        namespace = {"json": json, "hashlib": hashlib, "math": math}
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(logger_path), "exec"), namespace)
        prompts = ["same"] * 4
        rewards = np.array([[-2.], [0.], [1.], [8.]])
        details = {}
        _, used, _ = PerPromptStatTracker(energy_min=-.2, energy_max=.5).update(prompts, rewards, exp=True, diagnostics=details)
        with tempfile.TemporaryDirectory() as directory:
            log = dict(path=Path(directory), rank=1, run_id="test", attempt_id="test-attempt", prompts=set())
            namespace["log_weight_batch"](log, epoch=2, global_step=3, group_keys=prompts,
                                          diagnostics=details, samples_per_rank=2)
            rows = [json.loads(line) for line in (Path(directory) / "weights.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 2)
        for local_index, row in enumerate(rows):
            source_index = 2 + local_index
            self.assertEqual(row["energy_used"], float(used[source_index, 0]))
            for key in RAW_FIELDS:
                self.assertEqual(row[key], float(details[key][source_index, 0]))
            self.assertEqual(row["sample_index"], local_index)
            self.assertEqual(row["rank"], 1)


class ConfigurationAndWiringTests(unittest.TestCase):
    def test_config_preserves_scientific_hyperparameters(self):
        from config.awr import sd3_pickscore as baseline_config
        from config.awr_side_rescale import sd3_pickscore
        original = baseline_config().to_dict()
        variant = sd3_pickscore().to_dict()
        self.assertEqual(variant["train"].pop("energy_min"), -3.)
        self.assertEqual(variant["train"].pop("energy_max"), 3.)
        self.assertEqual(variant, original)

    def test_config_flags_accept_bound_overrides(self):
        program = '''
import json, sys
from absl import flags
from ml_collections import config_flags
config_flags.DEFINE_config_file("config", lock_config=True)
flags.FLAGS(sys.argv)
c = flags.FLAGS.config
print(json.dumps({"lower":c.train.energy_min,"upper":c.train.energy_max,"lr":c.train.learning_rate}))
'''
        result = subprocess.run([sys.executable, "-c", program,
                                 "--config=config/awr_side_rescale.py:sd3_pickscore",
                                 "--config.train.energy_min=-0.4", "--config.train.energy_max=5.0"], cwd=ROOT, check=True,
                                text=True, capture_output=True)
        actual = json.loads(result.stdout)
        self.assertEqual(actual["lower"], -.4)
        self.assertEqual(actual["upper"], 5.)

    def test_trainer_reads_bounds_from_config_and_keeps_loss_construction(self):
        original = ast.parse(BASE_TRAINER.read_text())
        variant = ast.parse(NEW_TRAINER.read_text())
        imports = [n.module for n in variant.body if isinstance(n, ast.ImportFrom)]
        self.assertIn("flow_grpo.stat_tracking_side_rescale", imports)
        self.assertNotIn("flow_grpo.stat_tracking", imports)
        tracker_calls = [n for n in ast.walk(variant) if isinstance(n, ast.Call)
                         and isinstance(n.func, ast.Name) and n.func.id == "PerPromptStatTracker"]
        self.assertGreaterEqual(len(tracker_calls), 1)
        for call in tracker_calls:
            args = {kw.arg: ast.unparse(kw.value) for kw in call.keywords}
            self.assertEqual(args["energy_min"], "config.train.energy_min")
            self.assertEqual(args["energy_max"], "config.train.energy_max")
        # Check the unchanged loss itself, not a separate reimplementation.
        def loss_branch(tree):
            nodes = [n for n in ast.walk(tree) if isinstance(n, ast.If)
                     and ast.unparse(n.test) == "not config.train.energy_mode"
                     and any(isinstance(child, ast.Assign)
                             and any(isinstance(target, ast.Name) and target.id == "target_x0"
                                     for target in child.targets)
                             for child in ast.walk(ast.Module(body=n.orelse, type_ignores=[])))]
            self.assertEqual(len(nodes), 1)
            # The original abs(e)>=2.99 statistic is deliberately removed: it
            # cannot measure how many samples were rescaled under variable bounds.
            nodes[0].orelse = [node for node in nodes[0].orelse
                              if not (isinstance(node, ast.Assign)
                                      and any(isinstance(target, ast.Subscript)
                                              and ast.unparse(target) == "loss_terms['energy_clip_rate']"
                                              for target in node.targets))]
            return ast.dump(nodes[0], include_attributes=False)
        self.assertEqual(loss_branch(variant), loss_branch(original))


if __name__ == "__main__":
    unittest.main()
