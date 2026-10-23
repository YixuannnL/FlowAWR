"""Energy-clip variants of the FlowAWR SD3 PickScore config.

Derived from config/awr.py:sd3_pickscore (main @ c46f74e) without modifying it.
The only additions are train.energy_min / train.energy_max, which the official
PerPromptStatTracker already accepts (defaults -3 / 3) but the official config
and training script never set. Use with scripts/train_awr_sd3_energy_clip.py.
awr.py is loaded by file path, the same way awr.py loads base.py, so this file
works wherever awr.py works.
"""
import importlib.util
import os

awr_path = os.path.join(os.path.dirname(__file__), "awr.py")
spec = importlib.util.spec_from_file_location("awr", awr_path)
awr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(awr)


def get_config(name):
    return globals()[name]()


def sd3_pickscore_eclip_m0p5_p5():
    config = awr.sd3_pickscore()
    config.train.energy_min = -0.5
    config.train.energy_max = 5.0
    return config
