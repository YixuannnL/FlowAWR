"""A2.3-derived config with a deterministic negative energy clipping schedule.

Use scripts/train_awr_sd3_energy_clip_cosine.py. Defaults: -0.5 through step 800,
cosine to -0.4 at step 1200, then hold -0.4. The positive bound stays +5.
Override train.energy_min, train.energy_min_final, train.energy_max,
train.energy_min_cosine_start_step and train.energy_min_cosine_end_step
via --config.train.<field>=<value>; set --config.run_name for each experiment.
All sampler, reward, optimizer and model settings come from the A2.3 config.
"""

import importlib.util
import os


base_path = os.path.join(os.path.dirname(__file__), "awr_energy_clip.py")
spec = importlib.util.spec_from_file_location("awr_energy_clip_cosine_base", base_path)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def get_config(name):
    return globals()[name]()


def sd3_pickscore_eclip_cosine():
    config = base.sd3_pickscore_eclip_m0p5_p5()
    # energy_min remains the initial bound in the saved config. The trainer
    # computes the active bound from global_step without mutating this config.
    config.train.energy_min_final = -0.4
    config.train.energy_min_cosine_start_step = 800
    config.train.energy_min_cosine_end_step = 1200
    config.run_name = "flowawr_sd3_pickscore_eclip_cosine_m0p5_m0p4_s800_e1200_p5"
    config.save_dir = os.path.join(config.logdir, "sd3", "pickscore_eclip_cosine")
    return config
