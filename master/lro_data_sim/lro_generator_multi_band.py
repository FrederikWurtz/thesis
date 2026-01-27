import numpy as np
import torch as torch
import os
from multiprocessing import Pool
from tqdm import tqdm

import cv2

from master.lro_data_sim.lro_data_utils_multi_band import detrend_2d, extract_local_subset_all_bands
from master.render.dem_utils import DEM
from master.render.hapke_model import FullHapkeModel
from master.render.camera import Camera
from master.render.renderer import Renderer


import torch
import torch.nn.functional as F

def generate_and_save_lro_data_multi_band(config: dict = None, save_path: str = None, device: str = "cpu"):

    images, reflectance_maps, dem_tensor, metas, w_tensor, theta_bar_tensor, lro_meta = generate_and_return_lro_data_multi_band(config, device=device)
    lro_meta = torch.tensor(lro_meta, dtype=torch.float32)    

    data_dict = {
        'dem': dem_tensor,
        'data': torch.stack(images),
        'reflectance_maps': torch.stack(reflectance_maps),
        'meta': torch.tensor(metas, dtype=torch.float32),
        'w': w_tensor,
        'theta_bar': theta_bar_tensor,
        'lro_meta': lro_meta
    }
        
    torch.save(data_dict, save_path)

def generate_and_return_lro_data_multi_band(config: dict = None, device: str = "cpu"):
    dem_tensor, w_tensor, theta_bar_tensor, lro_meta = generate_and_return_lro_multi_band(config, device=device)

    with torch.no_grad():
        # Convert DEM to torch tensor on GPU in a single operation
        dem_tensor = dem_tensor.to(device=device, dtype=torch.float32)
        w_tensor = w_tensor.to(device=device, dtype=torch.float32)
        theta_bar_tensor = theta_bar_tensor.to(device=device, dtype=torch.float32)

        dem_obj = DEM(dem_tensor, cellsize=1, x0=0, y0=0)
        hapke = FullHapkeModel(w=w_tensor, theta_bar=theta_bar_tensor)
        camera = Camera(image_width=config["IMAGE_W"],
                        image_height=config["IMAGE_H"],
                        focal_length=config["FOCAL_LENGTH"],
                        device=device)
        
        renderer = Renderer(dem_obj, hapke, camera)

        reflectance_maps = []
        images = []
        metas = []
        
        # Import internal functions for parameter handling and rendering - has to be done here to avoid circular imports
        from master.data_sim.generator import _get_sets_of_suncam_values, _render_single_image
        
        
        sets_of_params = _get_sets_of_suncam_values(manual_suncam_pars=config["MANUAL_SUNCAM_PARS"],
                                                          manual_sun_az_pm=config["MANUAL_SUN_AZ_PM"],
                                                          manual_sun_el_pm=config["MANUAL_SUN_EL_PM"],
                                                          manual_cam_az_pm=config["MANUAL_CAM_AZ_PM"],
                                                          manual_cam_el_pm=config["MANUAL_CAM_EL_PM"],
                                                          manual_cam_dist_pm=config["MANUAL_CAM_DIST_PM"],
                                                          images_per_dem=config["IMAGES_PER_DEM"])
        # Render images + reflectance maps
        for i in range(config["IMAGES_PER_DEM"]):
            params = sets_of_params[i]
            img, reflectance_map = _render_single_image(renderer=renderer, params=params, image_w=config["IMAGE_W"], image_h=config["IMAGE_H"])
            reflectance_maps.append(reflectance_map)
            images.append(img)
            metas.append(list(params))
    
    return images, reflectance_maps, dem_tensor, metas, w_tensor, theta_bar_tensor, lro_meta


def generate_and_return_lro_multi_band(config: dict = None, device: str = "cpu"):

    dem_path = "master/lro_data_sim/Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014_with_test_bands.tif"

    lat, lon, box_radius, height_norm = get_lat_lon_radius_height(config)    
    all_bands_array, metadata = extract_local_subset_all_bands(
                                                    dem_path=dem_path,
                                                    center_lat_deg=lat,
                                                    center_lon_deg=lon,
                                                    box_radius_m=box_radius,
                                                    res_m=config['SAMPLE_RES_M'],
                                                    local_proj_type="stere",
                                                    verbose = False
                                                )
    dem_array = all_bands_array[0, :, :]  # First band is DEM
    w_array = all_bands_array[1, :, :]    # Second band is albedo (w)
    theta_bar_array = all_bands_array[2, :, :]  # Third band is theta_bar (for roughness)
    
    dem_input = np.ma.masked_equal(dem_array, metadata['nodata'])

    # Detrend data
    dem_detrended = dem_input - np.nanmean(dem_input)
    dem_detrended = detrend_2d(dem_detrended, method='linear')
    # Mask nodata values (< -10000)
    nodata_threshold = -10000
    masked_data = []
    masked = np.ma.masked_where(dem_detrended < nodata_threshold, dem_detrended)
    # maybe find better thing to handle nodata values later

    # Resample to desired pixel size using PyTorch (can use GPU)

    desired_pixel_size = (config["LRO_DEM_SIZE"], config["LRO_DEM_SIZE"])  # (H, W)
    dem_resampled = resample_dem_torch(masked.filled(0), desired_pixel_size, device=device)
    w_resampled = resample_dem_torch(w_array.filled(0), desired_pixel_size, device=device)
    theta_bar_resampled = resample_dem_torch(theta_bar_array.filled(0), desired_pixel_size, device=device)

    # normalize heights
    dem_normalized = dem_resampled / torch.max(torch.abs(dem_resampled)) * height_norm

    return dem_normalized, w_resampled, theta_bar_resampled, [lat, lon, box_radius, height_norm]

def resample_dem_torch(dem_array, desired_pixel_size, device=None):
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
    return resampled.squeeze()
    
def get_lat_lon_radius_height(config: dict):
    if config["USE_SEPARATE_VALTEST_PARS"] == False:
        center_lat_deg = config['CENTER_LAT_DEG']
        center_lon_deg = config['CENTER_LON_DEG']
        box_radius_m = config['BOX_RADIUS_M']
        lat_deg_pm = config['CENTER_LAT_DEG_PM']
        lon_deg_pm = config['CENTER_LON_DEG_PM']
        box_radius_m_pm = config['BOX_RADIUS_M_PM']
        height_normalization = config['HEIGHT_NORMALIZATION']
        height_normalization_pm = config['HEIGHT_NORMALIZATION_PM']

        if config["STOCHASTIC"]:
            center_lat_deg += np.random.uniform(-lat_deg_pm, lat_deg_pm)
            center_lon_deg += np.random.uniform(-lon_deg_pm, lon_deg_pm)
            box_radius_m += np.random.uniform(-box_radius_m_pm, box_radius_m_pm)
            height_normalization += np.random.uniform(-height_normalization_pm, height_normalization_pm)
    else:
        # use separate val/test parameters, including stochasticity if enabled
        # print("Using separate val/test parameters for LRO DEM generation")
        center_lat_deg = config['CENTER_LAT_DEG_VALTEST']
        center_lon_deg = config['CENTER_LON_DEG_VALTEST']
        box_radius_m = config['BOX_RADIUS_M_VALTEST']
        height_normalization = config['HEIGHT_NORMALIZATION_VALTEST']
        lat_deg_pm = config['CENTER_LAT_DEG_PM_VALTEST']
        lon_deg_pm = config['CENTER_LON_DEG_PM_VALTEST']
        box_radius_m_pm = config['BOX_RADIUS_M_PM_VALTEST']
        height_normalization_pm = config['HEIGHT_NORMALIZATION_PM_VALTEST']

        if config["STOCHASTIC"]:
            center_lat_deg += np.random.uniform(-lat_deg_pm, lat_deg_pm)
            center_lon_deg += np.random.uniform(-lon_deg_pm, lon_deg_pm)
            box_radius_m += np.random.uniform(-box_radius_m_pm, box_radius_m_pm)
            height_normalization += np.random.uniform(-height_normalization_pm, height_normalization_pm)

    # Clamp latitude to valid range
    center_lat_deg = np.clip(center_lat_deg, -90.0, 90.0)

    # Wrap longituded using modulus, such that -180 < lon <= 180
    center_lon_deg = ((center_lon_deg + 180) % 360) - 180

    return center_lat_deg, center_lon_deg, box_radius_m, height_normalization