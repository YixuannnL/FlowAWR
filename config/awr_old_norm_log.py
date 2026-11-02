"""A1.8: original A1 PickScore baseline, 1000 updates, extra old-norm logs."""

import importlib.util
import os

_path = os.path.join(os.path.dirname(__file__), "awr.py")
_spec = importlib.util.spec_from_file_location("a1_original_awr_config", _path)
_baseline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_baseline)


def sd3_pickscore():
    config = _baseline.sd3_pickscore()
    # A1's 8-GPU configuration accumulates exactly one update per epoch.
    config.num_epochs = 1000
    config.run_name = "flowawr_sd3_pickscore_old_norm_a1_8"
    config.save_dir = f"{config.logdir}/sd3/pickscore_old_norm_a1_8"
    return config


def get_config(name):
    if name != "sd3_pickscore":
        raise ValueError(f"Unknown A1 old-norm logging configuration: {name}")
    return sd3_pickscore()
