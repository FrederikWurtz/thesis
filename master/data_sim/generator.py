import numpy as np
import torch as torch
import os
from multiprocessing import Pool
from tqdm import tqdm

from master.render.dem_utils import DEM
from master.render.hapke_model import HapkeModel
from master.render.camera import Camera
from master.render.renderer import Renderer

from master.lro_data_sim.lro_generator import generate_and_return_lro_data, generate_and_save_lro_data

import cProfile
import pstats
import io
import time

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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        # Convert DEM to torch tensor on GPU in a single operation
        dem_tensor = torch.from_numpy(dem_np).to(device=device, dtype=torch.float32)

        dem_obj = DEM(dem_tensor, cellsize=1, x0=0, y0=0)
        hapke = HapkeModel(w=0.6, B0=0.4, h=0.1, phase_fun="hg", xi=0.1)
        camera = Camera(image_width=image_w,
                        image_height=image_h,
                        focal_length=focal_length,
                        device=device)
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

def generate_and_return_data_CPU(config: dict = None):
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
        # Convert DEM to torch tensor on CPU, not GPU
        dem_tensor = torch.from_numpy(dem_np).to(dtype=torch.float32)
        device = torch.device('cpu')

        dem_obj = DEM(dem_tensor, cellsize=1, x0=0, y0=0)
        hapke = HapkeModel(w=0.6, B0=0.4, h=0.1, phase_fun="hg", xi=0.1)
        camera = Camera(image_width=image_w,
                        image_height=image_h,
                        focal_length=focal_length,
                        device=device)
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

    # Convert to PyTorch tensors and save
    data_dict = {
        'dem': torch.from_numpy(dem_np),
        'data': torch.stack(images),
        'reflectance_maps': torch.stack(reflectance_maps),
        'meta': torch.tensor(metas, dtype=torch.float32)
    }

    torch.save(data_dict, path)
    
def generate_and_return_worker_friendly(config):
    # args is a (path, config) tuple so this worker remains picklable and simple
    try:
        if config["USE_LRO_DEMS"]:
            images, reflectance_maps, dem_np, metas = generate_and_return_lro_data(config=config)
        else:
            images, reflectance_maps, dem_np, metas = generate_and_return_data(config=config)
        # Explicit cleanup
        import gc
        gc.collect()
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
        if config["USE_LRO_DEMS"]:
            generate_and_save_lro_data(config=config, save_path=path)
        else:
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


def _multiprocessing_worker(args):
    """Worker function for multiprocessing - must be at module level to be picklable."""
    dem_idx, gpu_id, images_dir, config = args
    
    # Set GPU for this process
    torch.cuda.set_device(gpu_id)
    
    filename = os.path.join(images_dir, f"dataset_{dem_idx:04d}.pt")
    try:
        if config["USE_LRO_DEMS"]:
            generate_and_save_lro_data(config=config, save_path=filename)
        else:
            generate_and_save_data(path=filename, config=config)
        return True
    except Exception:
        import traceback
        traceback.print_exc()
        return False


def _generate_with_multiprocessing(n_dems, n_gpus, max_workers, images_dir, config):
    """Multiprocessing approach - better for larger DEMs where compute dominates."""
    
    import torch.multiprocessing as mp
    
    # Force spawn method for CUDA compatibility
    mp.set_start_method('spawn', force=True)

    # Create work items
    tasks = [
        (i, i % n_gpus, images_dir, config)
        for i in range(n_dems)
    ]
    
    # Use process pool with spawn context
    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=max_workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(_multiprocessing_worker, tasks, chunksize=1),
            total=n_dems,
            desc="Generating DEMs"
        ))
    
    return results


def _generate_with_threading_optimized(n_dems, n_gpus, max_workers, images_dir, config):
    """Optimized threading with minimal overhead."""
    from concurrent.futures import ThreadPoolExecutor
    from queue import Queue
    import threading
    
    save_queue = Queue(maxsize=16)
    
    def gpu_worker(args):
        dem_idx, gpu_id = args
        filename = os.path.join(images_dir, f"dataset_{dem_idx:04d}.pt")
        torch.cuda.set_device(gpu_id)
        
        try:
            if config["USE_LRO_DEMS"]:
                images, reflectance_maps, dem_tensor, metas = generate_and_return_lro_data(config=config, device=f"cuda:{gpu_id}")
                data_dict = {
                    'dem': dem_tensor,
                    'data': torch.stack(images),
                    'reflectance_maps': torch.stack(reflectance_maps),
                    'meta': torch.tensor(metas, dtype=torch.float32)
                }

            else:
                images, reflectance_maps, dem_np, metas = generate_and_return_data(config=config)
                data_dict = {
                    'dem': torch.from_numpy(dem_np),
                    'data': torch.stack(images).cpu(),
                    'reflectance_maps': torch.stack(reflectance_maps).cpu(),
                    'meta': torch.tensor(metas, dtype=torch.float32)
                }
            
            
            save_queue.put((filename, data_dict))
            return True
        except Exception:
            import traceback
            traceback.print_exc()
            return False
    
    def saver_worker():
        while True:
            item = save_queue.get()
            if item is None:
                break
            filename, data_dict = item
            try:
                torch.save(data_dict, filename)
            except Exception:
                import traceback
                traceback.print_exc()
            finally:
                save_queue.task_done()
    
    n_saver_threads = 4
    saver_threads = []
    for _ in range(n_saver_threads):
        t = threading.Thread(target=saver_worker, daemon=True)
        t.start()
        saver_threads.append(t)
    
    tasks = [(i, i % n_gpus) for i in range(n_dems)]
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(
            executor.map(gpu_worker, tasks),
            total=n_dems,
            desc="Generating DEMs"
        ))
    
    save_queue.join()
    
    for _ in range(n_saver_threads):
        save_queue.put(None)
    for t in saver_threads:
        t.join()
    
    return results


def generate_and_save_data_pooled_multi_gpu(config: dict = None,
                                  images_dir: str = None,
                                  n_dems: int = None):
    """
    Optimized multi-GPU data generation with automatic tuning.
    """
    import time
    
    if config is None:
        raise ValueError("config must be provided")
    if images_dir is None:
        raise ValueError("images_dir must be provided")
    if n_dems is None:
        raise ValueError("n_dems must be provided")

    dem_size = config.get('DEM_SIZE', 512)
    
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        
        # Optimal configuration based on DEM size
        if dem_size < 512:
            workers_per_gpu = 8
            use_multiprocessing = False
        elif dem_size < 1024:
            workers_per_gpu = 4
            use_multiprocessing = False
        else:
            workers_per_gpu = 8
            use_multiprocessing = True
        
        max_workers = n_gpus * workers_per_gpu
        
        print(f"\n{'='*60}")
        print(f"Creating {n_dems} DEMs (size: {dem_size}x{dem_size})")
        print(f"Using {max_workers} workers across {n_gpus} GPU(s)")
        print(f"Workers per GPU: {workers_per_gpu}")
        print(f"Mode: {'Multiprocessing' if use_multiprocessing else 'Threading'}")
        print(f"Images per DEM: {config['IMAGES_PER_DEM']}")
        print(f"{'='*60}\n")
        
        t_start = time.perf_counter()
        
        if use_multiprocessing:
            results = _generate_with_multiprocessing(
                n_dems, n_gpus, max_workers, images_dir, config
            )
        else:
            results = _generate_with_threading_optimized(
                n_dems, n_gpus, max_workers, images_dir, config
            )
        
        t_end = time.perf_counter()
        elapsed = t_end - t_start
        
        n_success = sum(results)
        
        print(f"\n{'='*60}")
        print(f"Generation Complete")
        print(f"{'='*60}")
        print(f"Successfully created: {n_success}/{n_dems} DEMs")
        print(f"Total time: {elapsed:.2f} seconds")
        print(f"Time per DEM: {elapsed/n_dems:.2f} seconds")
        print(f"Throughput: {n_dems/elapsed:.2f} DEMs/second")
        print(f"{'='*60}\n")
        
    else:
        # CPU fallback
        print(f"\n{'='*60}")
        print(f"Creating {n_dems} DEMs using CPU")
        print(f"Using {config['NUM_WORKERS']} workers")
        print(f"{'='*60}\n")
        
        t_start = time.perf_counter()
        
        all_args = []
        for dem_idx in range(n_dems):
            filename = os.path.join(images_dir, f"dataset_{dem_idx:04d}.pt")
            all_args.append((filename, config))
        
        # Use spawn method for consistency
        import multiprocessing as mp
        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=config["NUM_WORKERS"]) as pool:
            results = list(tqdm(
                pool.imap_unordered(generate_and_save_worker_friendly, all_args, chunksize=1),
                total=n_dems,
                desc="Generating DEMs"
            ))
        
        t_end = time.perf_counter()
        elapsed = t_end - t_start
        
        n_success = sum(1 for r in results if r)
        
        print(f"\n{'='*60}")
        print(f"Successfully created: {n_success}/{n_dems} DEMs")
        print(f"Total time: {elapsed:.2f} seconds")
        print(f"Time per DEM: {elapsed/n_dems:.2f} seconds")
        print(f"Throughput: {n_dems/elapsed:.2f} DEMs/second")
        print(f"{'='*60}\n")


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
        # Clamp sun elevation to avoid tan(90°) = inf in shadow map computation
        sun_el = np.clip(sun_el, -89.9, 89.9)
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
