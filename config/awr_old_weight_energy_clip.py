"""Old-policy denominator with configurable energy clipping.

Defaults reproduce A4.14's [-3, 3] interval. Supply experiment-specific bounds
and timestep overrides with --config.*; the trainer contains no fixed bounds.
"""

import importlib.util
from pathlib import Path


_path = Path(__file__).with_name("awr_old_weight.py")
_spec = importlib.util.spec_from_file_location("old_weight_baseline", _path)
_baseline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_baseline)


def sd3_pickscore():
    config = _baseline.sd3_pickscore()
    config.train.energy_min = -3.0
    config.train.energy_max = 3.0
    config.run_name += "_energy_clip"
    config.save_dir += "_energy_clip"
    return config


def get_config(name):
    if name != "sd3_pickscore":
        raise ValueError(f"Unknown old-policy energy-clip configuration: {name}")
    return sd3_pickscore()
