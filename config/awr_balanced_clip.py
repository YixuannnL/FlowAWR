"""Independent PickScore config: clip K*p-1 to [-0.5, 3], then balance.

Use with scripts/train_awr_sd3_balanced_clip.py. Existing reward, advantage,
softmax, sampler, optimizer and loss settings come from awr.sd3_pickscore().
"""

import importlib.util
import os


awr_path = os.path.join(os.path.dirname(__file__), "awr.py")
spec = importlib.util.spec_from_file_location("awr_balanced_clip_base", awr_path)
awr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(awr)


def get_config(name):
    return globals()[name]()


def sd3_pickscore_balanced_clip_m0p5_p3():
    config = awr.sd3_pickscore()
    config.train.energy_min = -0.5
    config.train.energy_max = 3.0
    config.run_name = "flowawr_sd3_pickscore_balanced_clip_m0p5_p3"
    config.save_dir = os.path.join(config.logdir, "sd3", "pickscore_balanced_clip_m0p5_p3")
    config.log_raw_data = True
    return config
