"""Small helper to load `configs/defaults.ini` into a Python dict.

Usage:
    from configs.config_utils import load_defaults
    defaults = load_defaults()  # returns dict with UPPERCASE keys

This parser expects a section named [defaults] and supports Python-style
literals for lists/tuples/dicts (it uses ast.literal_eval). Values that are
plain numbers will be returned as int/float. All keys are uppercased.
"""
import ast
import configparser
import os
import multiprocessing
from typing import Any, Dict, Optional

# I am testing!

def load_config_file(path: Optional[str] = None) -> Dict[str, Any]:
    """Load defaults from the project's defaults.ini and return a dict.

    Args:
        path: Optional path to an INI file. If None, uses the `defaults.ini`
              located next to this module.

    Returns:
        Dict mapping UPPERCASE option names to Python values.
    """
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "defaults.ini")

    cp = configparser.ConfigParser()
    # Use read_file to get helpful errors if the file is missing/corrupt
    if not os.path.exists(path):
        raise FileNotFoundError(f"file not found at {path}")

    with open(path, "r", encoding="utf-8") as f:
        cp.read_file(f)

    if "defaults" not in cp:
        # If there is no [defaults] section, fail fast as requested by the user.
        # This mirrors configparser's semantics for missing sections.
        raise configparser.NoSectionError("defaults")

    out: Dict[str, Any] = {}
    for key, raw_value in cp["defaults"].items():
        v = raw_value.strip()
        # Try ast.literal_eval first (handles numbers, tuples, lists, dicts)
        parsed: Any
        try:
            parsed = ast.literal_eval(v)
        except Exception:
            # Fallback: try plain numeric conversion, else keep string
            try:
                if "." in v:
                    parsed = float(v)
                else:
                    parsed = int(v)
            except Exception:
                parsed = v
        out[key.upper()] = parsed

    # Special handling for NUM_WORKERS
    val = out.get("NUM_WORKERS")
    if isinstance(val, str) and val.lower() == "best":
        out["NUM_WORKERS"] = _get_best_num_workers(max_limit=16)
    else:
        out["NUM_WORKERS"] = _validate_positive_int(val, "NUM_WORKERS")


    # Special handling for MANUAL_SUNCAM_PARS
    sun_az = out.get("MANUAL_SUN_AZ")
    sun_el = out.get("MANUAL_SUN_EL")
    cam_az = out.get("MANUAL_CAM_AZ")
    cam_el = out.get("MANUAL_CAM_EL")
    cam_dist = out.get("MANUAL_CAM_DIST")
    manual_suncam = list(zip(sun_az, sun_el, cam_az, cam_el, cam_dist))
    out["MANUAL_SUNCAM_PARS"] = manual_suncam

    # Return validated config
    output = validate(out)
    return output


def _get_best_num_workers(max_limit=None):
    try:
        # HPC: brug kerner allokeret til job
        n_cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        # Fallback til alle kerner
        n_cpus = multiprocessing.cpu_count()

    num_workers = max(1, n_cpus - 1)
    if max_limit:
        num_workers = min(num_workers, max_limit)
    return num_workers


def validate(input):
        # Validate certain entries
    input["EPOCHS"] = _validate_positive_int(input["EPOCHS"], "EPOCHS")
    input["MANUAL_SUNCAM_PARS"] = _validate_suncam_pars(input["MANUAL_SUNCAM_PARS"])
    input["FLUID_TRAIN_DEMS"] = _validate_positive_int(input["FLUID_TRAIN_DEMS"], "FLUID_TRAIN_DEMS")
    input["FLUID_VAL_DEMS"] = _validate_positive_int(input["FLUID_VAL_DEMS"], "FLUID_VAL_DEMS")
    input["FLUID_TEST_DEMS"] = _validate_positive_int(input["FLUID_TEST_DEMS"], "FLUID_TEST_DEMS")
    input["IMAGES_PER_DEM"] = _validate_positive_int(input["IMAGES_PER_DEM"], "IMAGES_PER_DEM")
    input["DEM_SIZE"] = _validate_positive_int(input["DEM_SIZE"], "DEM_SIZE")
    input["IMAGES_PER_DEM"] = _validate_positive_int(input["IMAGES_PER_DEM"], "IMAGES_PER_DEM")
    input["IMAGE_H"] = _validate_positive_int(input["IMAGE_H"], "IMAGE_H")
    input["IMAGE_W"] = _validate_positive_int(input["IMAGE_W"], "IMAGE_W")
    input["FOCAL_LENGTH"] = _validate_positive_float(input["FOCAL_LENGTH"], "FOCAL_LENGTH")
    input["N_CRATERS"] = _validate_feature(input["N_CRATERS"], "N_CRATERS")
    input["N_RIDGES"] = _validate_feature(input["N_RIDGES"], "N_RIDGES")
    input["N_HILLS"] = _validate_feature(input["N_HILLS"], "N_HILLS")
    input["CRATER_RADIUS_RANGE"] = _validate_range(input["CRATER_RADIUS_RANGE"], "CRATER_RADIUS_RANGE")
    input["CRATER_DEPTH_RANGE"] = _validate_range(input["CRATER_DEPTH_RANGE"], "CRATER_DEPTH_RANGE")
    input["RIDGE_LENGTH_RANGE"] = _validate_range(input["RIDGE_LENGTH_RANGE"], "RIDGE_LENGTH_RANGE")
    input["RIDGE_WIDTH_RANGE"] = _validate_range(input["RIDGE_WIDTH_RANGE"], "RIDGE_WIDTH_RANGE")
    input["RIDGE_HEIGHT_RANGE"] = _validate_range(input["RIDGE_HEIGHT_RANGE"], "RIDGE_HEIGHT_RANGE")
    input["HILL_HEIGHT_RANGE"] = _validate_range(input["HILL_HEIGHT_RANGE"], "HILL_HEIGHT_RANGE")
    input["HILL_SIGMA_RANGE"] = _validate_range(input["HILL_SIGMA_RANGE"], "HILL_SIGMA_RANGE")
    input["MANUAL_SUN_AZ_PM"] = _validate_positive_int(input["MANUAL_SUN_AZ_PM"], "MANUAL_SUN_AZ_PM")
    input["MANUAL_SUN_EL_PM"] = _validate_positive_int(input["MANUAL_SUN_EL_PM"], "MANUAL_SUN_EL_PM")
    input["MANUAL_CAM_AZ_PM"] = _validate_positive_int(input["MANUAL_CAM_AZ_PM"], "MANUAL_CAM_AZ_PM")
    input["MANUAL_CAM_EL_PM"] = _validate_positive_int(input["MANUAL_CAM_EL_PM"], "MANUAL_CAM_EL_PM")
    input["MANUAL_CAM_DIST_PM"] = _validate_positive_int(input["MANUAL_CAM_DIST_PM"], "MANUAL_CAM_DIST_PM")

    return input

def create_folder_structure(config):
    """Create folder structure for dataset storage.

    Args:
        run_dir: Directory where dataset folders will be created.
    """
    # Create main directory for all runs if it doesn't exist
    sup_dir = config["SUP_DIR"]
    sup_dir = os.path.join(sup_dir)
    os.makedirs(sup_dir, exist_ok=True)

    # Create subdirectory for this particular run
    run_dir = os.path.join(sup_dir, config["RUN_DIR"])
    os.makedirs(run_dir, exist_ok=True)

    # Create val/test subdirectories
    subdirs = ['val', 'test']
    subdirs = [os.path.join(run_dir, subdir) for subdir in subdirs]
    for subdir in subdirs:
        os.makedirs(subdir, exist_ok=True)

    return run_dir, subdirs[0], subdirs[1]


def _validate_range(r, name):
    if not isinstance(r, (list, tuple)) or len(r) != 2:
        raise TypeError(f"{name} must be a length-2 sequence (min, max); got {r!r}")
    try:
        a = float(r[0])
        b = float(r[1])
    except Exception:
        raise TypeError(f"{name} entries must be numeric; got {r!r}")
    if a < 0 or b < 0:
        raise ValueError(f"{name} values must be non-negative; got {a}, {b}")
    return a, b

def _validate_feature(x, name):
    if not isinstance(x, int):
        raise TypeError(f"{name} must be an integer; got {type(x).__name__}: {x!r}")
    if x < 0:
        raise ValueError(f"{name} must be non-negative; got {x}")
    return x

def _validate_suncam_pars(manual_suncam_pars):
    if not isinstance(manual_suncam_pars, (list, tuple)) or len(manual_suncam_pars) == 0:
        raise TypeError(f"MANUAL_SUNCAM_PARS must be a non-empty list/tuple of 5-tuples, got {type(manual_suncam_pars).__name__}: {manual_suncam_pars!r}")
    for i, entry in enumerate(manual_suncam_pars):
        if not (isinstance(entry, (list, tuple)) and len(entry) == 5):
            raise TypeError(f"Entry {i} of MANUAL_SUNCAM_PARS must be a 5-tuple (sun_az, sun_el, cam_az, cam_el, cam_dist); got {entry!r}")
    return manual_suncam_pars

def _validate_positive_float(x, name):
    try:
        val = float(x)
    except Exception:
        raise TypeError(f"{name} must be a float; got {type(x).__name__}: {x!r}")
    if val < 0:
        raise ValueError(f"{name} must be positive; got {val}")
    return val

def _validate_positive_int(x, name):
    try:
        val = int(x)
    except Exception:
        raise TypeError(f"{name} must be an integer; got {type(x).__name__}: {x!r}")
    if val < 0:
        raise ValueError(f"{name} must be positive; got {val}")
    return val

if __name__ == "__main__":
    # Quick smoke test / demo
    try:
        d = load_config_file()
        for k in sorted(d.keys()):
            print(k, "=", d[k])
    except Exception as e:
        print("Failed to load defaults.ini:", e)


__all__ = ["load_config_file", "create_folder_structure"]
