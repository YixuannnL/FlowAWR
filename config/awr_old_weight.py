"""A4.14/A4.15: baseline PickScore/GenEval with the old-policy denominator.

All training hyperparameters come from the corresponding A1 preset and remain
overridable with --config.*. Only run/output identities are changed here.
"""

import importlib.util
from pathlib import Path


_path = Path(__file__).with_name("awr.py")
_spec = importlib.util.spec_from_file_location("old_weight_awr_baseline", _path)
_baseline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_baseline)


def _with_identity(config):
    config.run_name += "_old_weight"
    config.save_dir += "_old_weight"
    return config


def sd3_pickscore():
    return _with_identity(_baseline.sd3_pickscore())


def sd3_geneval():
    return _with_identity(_baseline.sd3_geneval())


def get_config(name):
    if name not in {"sd3_pickscore", "sd3_geneval"}:
        raise ValueError(f"Unknown old-policy weight configuration: {name}")
    return globals()[name]()
