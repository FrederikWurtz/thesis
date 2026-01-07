import numpy as np
import torch as torch
import os
from multiprocessing import Pool
from tqdm import tqdm

from master.render.dem_utils import DEM
from master.render.hapke_model import HapkeModel
from master.render.camera import Camera
from master.render.renderer import Renderer

def generate_and_return_data(config: dict = None):
    """
    Generate a synthetic DEM and render five camera images + reflectance maps.

    This is the top-level data generator used by the project. It creates a
    synthetic DEM (via ``_generate_synthetic_dem``), constructs the rendering
    pipeline (``DEM``, ``HapkeModel``, ``Camera``, ``Renderer``) and produces
    five image / reflectance-map pairs using the project's suncam sampling
    helper.

    Parameters
    ----------
    config : dict-like
        Configuration dictionary. Required keys used by this function include
        (names are the ones expected in the project defaults):
        - ``DEM_SIZE`` (int or (H, W))
        - ``IMAGE_HEIGHT``, ``IMAGE_WIDTH`` (int)
        - ``FOCAL_LENGTH`` (float)
        - ``MANUAL_SUNCAM_PARS`` (bool)
        - manual suncam parameter ranges: ``MANUAL_SUN_AZ_PM``, ``MANUAL_SUN_EL_PM``,
          ``MANUAL_CAM_AZ_PM``, ``MANUAL_CAM_EL_PM``, ``MANUAL_CAM_DIST_PM``
        - feature placement ranges / counts: ``N_CRATERS``, ``N_RIDGES``, ``N_HILLS``,
          ``CRATER_DEPTH_RANGE``, ``CRATER_RADIUS_RANGE``, ``RIDGE_HEIGHT_RANGE``,
          ``RIDGE_LENGTH_RANGE``, ``RIDGE_WIDTH_RANGE``, ``HILL_HEIGHT_RANGE``,
          ``HILL_SIGMA_RANGE``

    Returns
    -------
    images, reflectance_maps, dem_np, metas
        images : list of numpy.ndarray
            Five camera-sampled images, each shaped (IMAGE_HEIGHT, IMAGE_WIDTH).
        reflectance_maps : list of numpy.ndarray
            Five reflectance maps at DEM resolution (H_dem, W_dem).
        dem_np : numpy.ndarray
            Generated DEM as a 2D array (H_dem, W_dem), dtype float32.
        metas : list
            A list of five metadata tuples/lists describing sun/camera params
            used for each render.

    Raises
    ------
    ValueError
        If ``config`` is None.

    Notes
    -----
    - The function runs the generation and rendering inside ``torch.no_grad()``
      to avoid tracking gradients.
    - This docstring focuses on the API; implementation details are in the
      helper functions (see ``_generate_synthetic_dem`` and
      ``_get_5_sets_of_suncam_values``).
    """

    if config is None:
        raise ValueError("config must be provided")

    # unpack once to avoid repeated dict lookups and make locals explicit
    dem_size = config['DEM_SIZE']
    images_per_dem = config['IMAGES_PER_DEM']
    image_h = config['IMAGE_H']
    image_w = config['IMAGE_W']
    focal_length = config['FOCAL_LENGTH']
    manual_suncam_pars = config['MANUAL_SUNCAM_PARS']
    n_craters = config['N_CRATERS']
    n_ridges = config['N_RIDGES']
    n_hills = config['N_HILLS']
    crater_depth_range = config['CRATER_DEPTH_RANGE']
    crater_radius_range = config['CRATER_RADIUS_RANGE']
    ridge_height_range = config['RIDGE_HEIGHT_RANGE']
    ridge_length_range = config['RIDGE_LENGTH_RANGE']
    ridge_width_range = config['RIDGE_WIDTH_RANGE']
    hill_height_range = config['HILL_HEIGHT_RANGE']
    hill_sigma_range = config['HILL_SIGMA_RANGE']
    manual_sun_az_pm = config['MANUAL_SUN_AZ_PM']
    manual_sun_el_pm = config['MANUAL_SUN_EL_PM']
    manual_cam_az_pm = config['MANUAL_CAM_AZ_PM']
    manual_cam_el_pm = config['MANUAL_CAM_EL_PM']
    manual_cam_dist_pm = config['MANUAL_CAM_DIST_PM']


    n_craters, n_ridges, n_hills = _get_random_n_features(n_craters_max=n_craters, 
                                                            n_ridges_max=n_ridges, 
                                                            n_hills_max=n_hills)

    dem_np = _generate_synthetic_dem(size=dem_size, 
                                    n_craters=n_craters, 
                                    n_ridges=n_ridges, 
                                    n_hills=n_hills,
                                    crater_depth_range=crater_depth_range,
                                    crater_radius_range=crater_radius_range,
                                    ridge_height_range=ridge_height_range,
                                    ridge_length_range=ridge_length_range,
                                    ridge_width_range=ridge_width_range,
                                    hill_height_range=hill_height_range,
                                    hill_sigma_range=hill_sigma_range)
        
    with torch.no_grad():
        dem_obj = DEM(dem_np, cellsize=1, x0=0, y0=0)
        hapke = HapkeModel(w=0.6, B0=0.4, h=0.1, phase_fun="hg", xi=0.1)
        camera = Camera(image_width=image_w,
                        image_height=image_h,
                        focal_length=focal_length)
        renderer = Renderer(dem_obj, hapke, camera)

        reflectance_maps = []
        images = []
        metas = []

        # Use the project's standard suncam variation function
        sets_of_params = _get_sets_of_suncam_values(manual_suncam_pars=manual_suncam_pars,
                                                          manual_sun_az_pm=manual_sun_az_pm,
                                                          manual_sun_el_pm=manual_sun_el_pm,
                                                          manual_cam_az_pm=manual_cam_az_pm,
                                                          manual_cam_el_pm=manual_cam_el_pm,
                                                          manual_cam_dist_pm=manual_cam_dist_pm,
                                                          images_per_dem=images_per_dem)
        # Render images + reflectance maps
        for i in range(images_per_dem):
            params = sets_of_params[i]
            img, reflectance_map = _render_single_image(renderer=renderer, params=params, image_w=image_w, image_h=image_h)
            reflectance_maps.append(reflectance_map)
            images.append(img)
            metas.append(list(params))
    return images, reflectance_maps, dem_np, metas

def generate_and_save_data(path: str = None, config: dict = None):
    """
    Generate data (via ``generate_and_return_data``) and save to a compressed
    NumPy ``.npz`` file.

    Parameters
    ----------
    path : str or pathlib.Path
        Destination file path for the compressed ``.npz`` archive. If a
        ``Path`` is passed it will be used directly.
    config : dict-like
        Configuration dictionary forwarded to ``generate_and_return_data``.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If ``config`` or ``path`` is not provided.
    """

    if config is None:
        raise ValueError("config must be provided")
    if path is None:
        raise ValueError("path must be provided")

    images, reflectance_maps, dem_np, metas = generate_and_return_data(config=config)

    np.savez_compressed(path,
                        dem=dem_np,
                        data=np.stack(images, axis=0),
                        reflectance_maps=np.stack(reflectance_maps, axis=0),
                        meta=np.array(metas, dtype=np.float32))
    
def generate_and_return_worker_friendly(config):
    # args is a (path, config) tuple so this worker remains picklable and simple
    try:
        images, reflectance_maps, dem_np, metas = generate_and_return_data(config=config)
    except Exception:
        import traceback
        traceback.print_exc()
        return None
    return images, reflectance_maps, dem_np, metas

def generate_and_save_worker_friendly(args):
    """Worker-friendly generator: instantiate local HapkeModel/Camera/Renderer and save a .npz.
    Wrapper around `generate_and_save_data` that takes a single argument tuple for use in
    multiprocessing.Pool workers.
    """
    # args is a (path, config) tuple so this worker remains picklable and simple
    path, config = args
    try:
        generate_and_save_data(path=path, config=config)
    except Exception:
        import traceback
        traceback.print_exc()
        return False
    return True


def generate_and_save_data_pooled(config: dict = None,
                                  images_dir: str = None,
                                  n_dems: int = None):
    if config is None:
        raise ValueError("config must be provided")
    if images_dir is None:
        raise ValueError("images_dir must be provided")
    if n_dems is None:
        raise ValueError("n_dems must be provided")

    print(f"\n{'='*60}")
    print(f"Creating {n_dems} DEMs in parallel, using {config['NUM_WORKERS']} workers")
    print(f"with {config['IMAGES_PER_DEM']} images, saving in folder '{images_dir}'")
    print(f"{'='*60}\n")

    all_args = []
    for dem_idx in range(n_dems):
        filename = os.path.join(
            images_dir,
            f"dataset_{dem_idx:04d}.npz"
        )
        all_args.append((filename, config))

    n_batches = (n_dems + config["BATCH_SIZE"] - 1) // config["BATCH_SIZE"]
    n_batches = (n_dems + config["BATCH_SIZE"] - 1) // config["BATCH_SIZE"]
    # accumulate successes across all batches so final summary reports the
    # total number of successfully created DEMs (not just the last batch)
    n_success_total = 0
    for batch_idx in tqdm(range(n_batches), desc="Generating DEMs in batches", position=0, dynamic_ncols=True, leave=False):
        batch_start = batch_idx * config["BATCH_SIZE"]
        batch_end = min((batch_idx + 1) * config["BATCH_SIZE"], n_dems)
        batch_args = all_args[batch_start:batch_end]
        with Pool(processes=config["NUM_WORKERS"]) as pool:
            results = list(tqdm(
                pool.imap(generate_and_save_worker_friendly, batch_args),
                total=len(batch_args),
                desc=f"Batch {batch_idx+1}/{n_batches}",
                leave=False,
                position=1,
                dynamic_ncols=True,

            ))
        n_success = sum(1 for r in results if r)
        n_success_total += n_success
        del results

    tqdm.write(f"Finished generating dataset. Successfully created {n_success_total}/{len(all_args)} DEMs.\n")



def _get_sets_of_suncam_values(manual_suncam_pars=None, manual_sun_az_pm=None, manual_sun_el_pm=None,
                                manual_cam_az_pm=None, manual_cam_el_pm=None, manual_cam_dist_pm=None,
                                images_per_dem=None):
    """Get 5 sets of sun/camera parameters with small random variations."""
    #fail if none is provided
    if manual_suncam_pars is None:
        raise ValueError("manual_suncam_pars must be provided")

    random_seed = np.random.randint(0, images_per_dem)
    suncam_values = []
    for i in range(images_per_dem):
        rotated_idx = (i + (random_seed or 0)) % len(manual_suncam_pars)
        sun_az, sun_el, cam_az, cam_el, cam_dist = manual_suncam_pars[rotated_idx]
        sun_az += np.random.uniform(-manual_sun_az_pm, manual_sun_az_pm)
        sun_el += np.random.uniform(-manual_sun_el_pm, manual_sun_el_pm)
        cam_az += np.random.uniform(-manual_cam_az_pm, manual_cam_az_pm)
        cam_el += np.random.uniform(-manual_cam_el_pm, manual_cam_el_pm)
        cam_dist += np.random.uniform(-manual_cam_dist_pm, manual_cam_dist_pm)
        suncam_values.append((sun_az, sun_el, cam_az, cam_el, cam_dist))
    return suncam_values

def _get_random_n_features(n_craters_max=None, n_ridges_max=None, n_hills_max=None):
    """Get the number of features in a feature map."""
    n_craters = int(np.random.randint(1, n_craters_max)) if n_craters_max > 0 else 0
    n_ridges = int(np.random.randint(1, n_ridges_max)) if n_ridges_max > 0 else 0
    n_hills = int(np.random.randint(1, n_hills_max)) if n_hills_max > 0 else 0
    return n_craters, n_ridges, n_hills

def _generate_synthetic_dem(size=None, n_craters=None, n_ridges=None, n_hills=None, 
                            crater_depth_range=None, crater_radius_range=None, 
                            ridge_height_range=None, ridge_length_range=None, 
                            ridge_width_range=None, hill_height_range=None, 
                            hill_sigma_range=None):
    """
    Simple synthetic DEM generator.

    Places craters, ridges and Gaussian hills at random positions using
    the global parameter ranges defined in this file. This version is
    intentionally simple (allows overlap) and is faster/easier to reason
    about than the complex placement used in generate_synthetic_dem_complex().

    Args:
        size: output DEM size in pixels
        n_craters: number of craters to place
        n_ridges: number of ridges to place
        n_hills: number of Gaussian hills to place
        seed: optional random seed for reproducibility
        flat_bottom: if True, do not add base undulation
    Returns:
        dem: (size, size) float32 numpy array with elevations (meters)
    """

    dem = np.zeros((size, size), dtype=np.float32)
    yy, xx = np.ogrid[:size, :size]

    # ---- Craters (simple random placement) ----
    for _ in range(n_craters):
        # sample integer radius in pixels
        min_r = max(1, int(size * crater_radius_range[0]))
        max_r = max(min_r + 1, int(size * crater_radius_range[1]))
        r = np.random.randint(min_r, max_r)
        cy = np.random.randint(r, size - r)
        cx = np.random.randint(r, size - r)
        depth = np.random.uniform(crater_depth_range[0], crater_depth_range[1])

        dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
        bowl_mask = dist < r
        crater = np.zeros_like(dem)
        if np.any(bowl_mask):
            crater[bowl_mask] = -depth * (1 - (dist[bowl_mask] / r)**2)
        # add a simple rim
        rim_mask = (dist > r*0.9) & (dist < r*1.05)
        if np.any(rim_mask):
            rim_height = depth * 0.2
            crater[rim_mask] += rim_height * (1 - ((dist[rim_mask] - r*0.9) / (r*0.15))**2)

        dem += crater

    # ---- Ridges (random placement, same profile logic as complex generator) ----
    for _ in range(n_ridges):
        # sample ridge parameters using same ranges as the complex generator
        angle = np.random.uniform(0, 2*np.pi)
        ridge_length = np.random.randint(int(size * ridge_length_range[0]), max(int(size * ridge_length_range[1]), 1))
        ridge_width = np.random.randint(int(size * ridge_width_range[0]), max(int(size * ridge_width_range[1]), 1))
        ridge_height_val = np.random.uniform(ridge_height_range[0], ridge_height_range[1])

        # choose a random centerpoint with a small margin to keep segment inside
        max_width = int(size * ridge_width_range[1])
        margin = max(max_width + 10, 1)
        if margin >= size // 2:
            margin = max(1, size // 10)
        x0 = np.random.randint(margin, size - margin)
        y0 = np.random.randint(margin, size - margin)
        x1 = x0 + ridge_length * np.cos(angle)
        y1 = y0 + ridge_length * np.sin(angle)

        # project grid points onto the segment and compute distance field
        dx = xx - x0
        dy = yy - y0
        # use ridge_length for normalization (same as complex code)
        t = (dx * np.cos(angle) + dy * np.sin(angle)) / float(max(1, ridge_length))
        t = np.clip(t, 0, 1)
        closest_x = x0 + t * ridge_length * np.cos(angle)
        closest_y = y0 + t * ridge_length * np.sin(angle)
        dist = np.sqrt((xx - closest_x)**2 + (yy - closest_y)**2)

        transition_width = ridge_width * 0.8
        full_ridge_mask = dist < (ridge_width + transition_width)
        ridge = np.zeros_like(dem)
        if np.any(full_ridge_mask):
            d = dist[full_ridge_mask]
            sigma_core = ridge_width / 2.5
            # edge_value where gaussian would evaluate at ridge_width
            edge_value = np.exp(-(ridge_width**2) / (2 * sigma_core**2))
            t2 = (d - ridge_width) / transition_width
            t2 = np.clip(t2, 0, 1)
            transition_factor = (1 - t2)**3
            ridge_height = np.where(
                d <= ridge_width,
                ridge_height_val * np.exp(-(d**2) / (2 * sigma_core**2)),
                ridge_height_val * edge_value * transition_factor
            )
            ridge[full_ridge_mask] = ridge_height
        dem += ridge

    # ---- Gaussian hills ----
    for _ in range(n_hills):
        sigma = np.random.uniform(size * hill_sigma_range[0], size * hill_sigma_range[1])
        height = np.random.uniform(hill_height_range[0], hill_height_range[1])
        cy = np.random.randint(0, size)
        cx = np.random.randint(0, size)
        dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
        hill = height * np.exp(-(dist**2) / (2 * sigma**2))
        dem += hill

    return dem

def _render_single_image(renderer=None, params=None, image_w=None, image_h=None):
    """
    Render a single camera image + reflectance map from a DEM object using
    an instance of `Renderer`.

    suncam_params: tuple or list of (sun_az_deg, sun_el_deg, cam_az_deg, cam_el_deg, cam_distance)

    Returns (image, reflectance_map) as numpy arrays (H_img, W_img) and
    (H_dem, W_dem) respectively.
    """
    sun_az, sun_el, cam_az, cam_el, cam_dist = params
    # Most Renderer implementations in this repo provide a method named
    # `render_camera_image` or `render_shading`. We call `render_camera_image`
    # which should return the camera-sampled image and a reflectance map.
    result = renderer.render_camera_image(
        sun_az_deg=sun_az,
        sun_el_deg=sun_el,
        camera_az_deg=cam_az,
        camera_el_deg=cam_el,
        camera_distance_from_center=cam_dist,
        img_width=image_w,
        img_height=image_h,
    )

    img, refl_map = result
    return img, refl_map

__all__ = ["generate_and_return_data", 
           "generate_and_save_data", 
           "generate_and_return_worker_friendly", 
           "generate_and_save_worker_friendly"]
