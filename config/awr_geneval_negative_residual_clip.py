"""GenEval with independent energy clipping and negative-only residual clipping.

Use the existing scripts/train_awr_sd3_negative_residual_clip.py trainer.
Defaults follow A1.6.1's PickScore-style normalization, with A4.4's x0-space
unweighted residual RMS threshold applied only to negative-energy samples.
All experiment overrides live here and accept --config.<field>=<value>.
"""
import importlib.util
import os


awr_path = os.path.join(os.path.dirname(__file__), "awr.py")
spec = importlib.util.spec_from_file_location("awr", awr_path)
awr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(awr)


def get_config(name):
    return globals()[name]()


def sd3_geneval():
    config = awr.sd3_geneval()
    config.run_name = "flowawr_sd3_geneval_negative_residual_clip"
    config.save_dir = f"{config.logdir}/sd3/geneval_negative_residual_clip"
    # Match the actual A1.6.1 checkpoint cadence (base.py defaults to 30).
    config.save_freq = 10

    # False: K * softmax - 1 (A1.6.1). True: official GenEval normalization.
    config.train.hard_gating = False
    config.train.energy_min = -0.5
    config.train.energy_max = 5.0
    config.train.residual_clip_enabled = True
    # RMS(x0 - pred_x0_old), per image and actual timestep; no energy or /t.
    # The trainer selects energy < 0; positive and zero samples are unchanged.
    config.train.residual_rms_max = 0.75
    return config
