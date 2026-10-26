import importlib.util
import os


awr_path = os.path.join(os.path.dirname(__file__), "awr.py")
spec = importlib.util.spec_from_file_location("awr_offset_norm_base", awr_path)
awr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(awr)


def get_config(name):
    return globals()[name]()


def sd3_pickscore_offset_norm():
    config = awr.sd3_pickscore()
    config.run_name = "flowawr_sd3_pickscore_offset_norm"
    config.save_dir = os.path.join(config.logdir, "sd3", "pickscore_offset_norm")
    return config
