"""Independent A4.4 variants with fixed empirical sign/actual-t RMS cap tables.

Tables were selected against A1 and checked on held-out A2.3.1/A4.4 records.
The trainer refuses an empty calibration ID or an uncalibrated actual t value.
"""

import importlib.util
import os

_path = os.path.join(os.path.dirname(__file__), "awr_residual_clip.py")
_spec = importlib.util.spec_from_file_location("frozen_a44_tclip_config", _path)
_baseline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_baseline)

_T_GRID = (
    0.008928571827709675, 0.2780488133430481, 0.46487605571746826,
    0.6021506190299988, 0.7072785496711731, 0.7903683185577393,
    0.8576923608779907, 0.9133490324020386, 0.9601293802261353, 1.0,
)
_CALIBRATIONS = {
    "tight": {
        "positive": (0.015, 0.30, 0.49, 0.65, 0.73, 0.78, 0.84, 0.90, 1.00, 1.04),
        "negative": (0.015, 0.30, 0.49, 0.65, 0.73, 0.78, 0.84, 0.90, 1.00, 1.04),
        "id": "a1_a231_a44_signed_t_cap200_20261102_v1",
    },
    "loose": {
        "positive": (0.015, 0.32, 0.50, 0.65, 0.75, 0.85, 0.92, 1.00, 1.10, 1.20),
        "negative": (0.015, 0.32, 0.50, 0.65, 0.75, 0.85, 0.92, 1.00, 1.10, 1.20),
        "id": "a1_a231_a44_signed_t_cap219_20261102_v1",
    },
}


def _get_tclip_config(profile):
    config = _baseline.sd3_pickscore()
    calibration = _CALIBRATIONS[profile]
    config.num_epochs = 100000
    config.run_name = f"flowawr_sd3_pickscore_tclip_{profile}"
    config.save_dir = f"{config.logdir}/sd3/pickscore_tclip_{profile}"
    # Scalar A4.4 cap is replaced by an explicit table, not silently retained.
    del config.train.residual_rms_max
    config.train.residual_clip_profile = profile
    config.train.residual_clip_calibration_id = calibration["id"]
    config.train.residual_t_grid = list(_T_GRID)
    config.train.residual_rms_positive = list(calibration["positive"])
    config.train.residual_rms_negative = list(calibration["negative"])
    return config


def sd3_pickscore_tclip_tight():
    return _get_tclip_config("tight")


def sd3_pickscore_tclip_loose():
    return _get_tclip_config("loose")


def get_config(name):
    variants = {
        "sd3_pickscore_tclip_tight": sd3_pickscore_tclip_tight,
        "sd3_pickscore_tclip_loose": sd3_pickscore_tclip_loose,
    }
    if name not in variants:
        raise ValueError(f"Unknown timestep residual clipping configuration: {name}")
    return variants[name]()
