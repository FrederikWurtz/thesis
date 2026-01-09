import numpy as np
import torch as torch
import os
from multiprocessing import Pool
from tqdm import tqdm

from LRO_data.functions import detrend_2d, extract_local_dem_subset
from master.render.dem_utils import DEM
from master.render.hapke_model import HapkeModel
from master.render.camera import Camera
from master.render.renderer import Renderer

from master.data_sim.generator import _get_sets_of_suncam_values, _render_single_image

import torch
import torch.nn.functional as F



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


    


    dem_np = generate_and_return_lro_dem(
                                        CENTER_LAT_DEG=config.get('CENTER_LAT_DEG', -45.0),
                                        CENTER_LON_DEG=config.get('CENTER_LON_DEG', 30.0),
                                        BOX_RADIUS_M=config.get('BOX_RADIUS_M', 20_000.0),
                                        dem_path=config.get('DEM_PATH', None),
                                        desired_res_m=100.0,
                                        desired_pixel_size=dem_size,
                                        verbose=False
                                        )
    


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



def generate_and_return_lro_dem(CENTER_LAT_DEG=None, CENTER_LON_DEG=None, BOX_RADIUS_M=None, dem_path=None
                                desired_res_m=100.0, desired_pixel_size= (512, 512), verbose=False):

    dem_path = "/Users/au644271/Desktop/local_python/LRO_data_sandbox/Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014.tif"

    dem_array, metadata = extract_local_dem_subset(
                                                    dem_path=dem_path,
                                                    center_lat_deg=CENTER_LAT_DEG,
                                                    center_lon_deg=CENTER_LON_DEG,
                                                    box_radius_m=BOX_RADIUS_M,
                                                    res_m=100.0,
                                                    local_proj_type="stere",
                                                    verbose = False
                                                )
    
    data_input = np.ma.masked_equal(dem_array, metadata['nodata'])

    # Detrend data
    data_detrended = data_input - np.nanmean(data_input)
    data_detrended = detrend_2d(data_detrended, method='linear')

    # Mask nodata values (< -10000) and find global min/max for consistent colorscale
    nodata_threshold = -10000
    masked_data = []
    masked = np.ma.masked_where(data_detrended < nodata_threshold, data_detrended)
    # maybe find better thing to handle nodata values later

    # Resample to desired pixel size using PyTorch (can use GPU)
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

    desired_pixel_size = (desired_pixel_size[0], desired_pixel_size[1])  # (H, W)
    dem_resampled = resample_dem_torch(masked.filled(0), desired_pixel_size, device=device)

    return dem_resampled

def resample_dem_torch(dem_array, desired_pixel_size, device='cpu'):
    """
    Resample using PyTorch (can use GPU).
    """
    # Convert to torch tensor: (1, 1, H, W) for batch and channel dims
    dem_tensor = torch.from_numpy(dem_array).unsqueeze(0).unsqueeze(0).float()
    dem_tensor = dem_tensor.to(device)

    # Interpolate
    resampled = F.interpolate(dem_tensor, 
                                size=desired_pixel_size,
                                mode='bicubic',  # or 'bilinear', 'nearest'
                                align_corners=True)

    # Remove batch and channel dims
    return resampled.squeeze().cpu().numpy()
    
def get
    
import cv2

# Alternative version of generate_and_return_lro_dem with resampling
def generate_and_return_lro_dem_alt(CENTER_LAT_DEG=None, CENTER_LON_DEG=None, BOX_RADIUS_M=None, 
                                dem_path=None, desired_res_m=100.0, 
                                desired_pixel_size=(512, 512), verbose=False):

    dem_path = "/Users/au644271/Desktop/local_python/LRO_data_sandbox/Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014.tif"

    dem_array, metadata = extract_local_dem_subset(
                                                    dem_path=dem_path,
                                                    center_lat_deg=CENTER_LAT_DEG,
                                                    center_lon_deg=CENTER_LON_DEG,
                                                    box_radius_m=BOX_RADIUS_M,
                                                    res_m=desired_res_m,
                                                    local_proj_type="stere",
                                                    verbose=verbose
                                                )
    
    # Resample to desired pixel size
    if dem_array.shape != desired_pixel_size:
        # Mask nodata before resampling
        nodata = metadata.get('nodata', -9999)
        mask = dem_array != nodata
        
        # Resample DEM and mask separately
        dem_resampled = cv2.resize(dem_array, 
                                   (desired_pixel_size[1], desired_pixel_size[0]),
                                   interpolation=cv2.INTER_CUBIC)
        mask_resampled = cv2.resize(mask.astype(np.uint8), 
                                    (desired_pixel_size[1], desired_pixel_size[0]),
                                    interpolation=cv2.INTER_NEAREST) > 0.5
        
        # Restore nodata values
        dem_resampled[~mask_resampled] = nodata
        dem_array = dem_resampled
    
    return dem_array, metadata