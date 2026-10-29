"""A2.5 protocol with a per-group negative energy clipping bound.

Use scripts/train_awr_sd3_dynamic_energy_clip.py. For the actual softmax input
z, groups with logmeanexp(z) - mean(z) >= log(2) use [-0.5, 3]; other groups
use [-0.7, 3]. All other settings come from the existing energy-clip config.
"""

import importlib.util
import os

import numpy as np


base_path = os.path.join(os.path.dirname(__file__), "awr_energy_clip.py")
spec = importlib.util.spec_from_file_location("awr_dynamic_energy_clip_base", base_path)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def get_config(name):
    return globals()[name]()


def sd3_pickscore_dynamic_energy_clip():
    config = base.sd3_pickscore_eclip_m0p5_p5()
    config.train.energy_min = -0.7
    config.train.energy_max = 3.0
    config.train.energy_protected_min = -0.5
    config.train.energy_B_threshold = np.log(2.0)
    return config
