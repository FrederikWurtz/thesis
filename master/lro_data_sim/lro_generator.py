import numpy as np
import torch as torch
import os
from multiprocessing import Pool
from tqdm import tqdm

import cv2

from LRO_data.functions import detrend_2d, extract_local_dem_subset
from master.render.dem_utils import DEM
from master.render.hapke_model import HapkeModel
from master.render.camera import Camera
from master.render.renderer import Renderer

import torch
import torch.nn.functional as F


def generate_and_save_lro_data(config: dict = None, save_path: str = None):

    images, reflectance_maps, dem_tensor, metas = generate_and_return_lro_data(config)

    if not torch.is_tensor(dem_tensor):
        dem_tensor = torch.from_numpy(dem_tensor)

    data_dict = {
        'dem': dem_tensor,
        'data': torch.stack(images),
        'reflectance_maps': torch.stack(reflectance_maps),
        'meta': torch.tensor(metas, dtype=torch.float32)
    }
        
    torch.save(data_dict, save_path)


def generate_and_return_lro_data(config: dict = None, device: str = "cpu"):

    dem_tensor = generate_and_return_lro_dem(config)

    with torch.no_grad():
        # Convert DEM to torch tensor on GPU in a single operation
        dem_tensor = dem_tensor.to(device=device, dtype=torch.float32)

        dem_obj = DEM(dem_tensor, cellsize=1, x0=0, y0=0)
        hapke = HapkeModel(w=0.6, B0=0.4, h=0.1, phase_fun="hg", xi=0.1)
        camera = Camera(image_width=config["IMAGE_W"],
                        image_height=config["IMAGE_H"],
                        focal_length=config["FOCAL_LENGTH"],
                        device=device)
        
        renderer = Renderer(dem_obj, hapke, camera)

        reflectance_maps = []
        images = []
        metas = []

        # Use the project's standard suncam variation function
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
            
    return images, reflectance_maps, dem_tensor, metas


def generate_and_return_lro_dem(config: dict = None):

    dem_path = "/Users/au644271/Desktop/local_python/LRO_data_sandbox/Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014.tif"

    lat, lon, box_radius = get_lat_lon_radius(config)    
    dem_array, metadata = extract_local_dem_subset(
                                                    dem_path=dem_path,
                                                    center_lat_deg=lat,
                                                    center_lon_deg=lon,
                                                    box_radius_m=box_radius,
                                                    res_m=config['SAMPLE_RES_M'],
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

    desired_pixel_size = (config["LRO_DEM_SIZE"], config["LRO_DEM_SIZE"])  # (H, W)
    dem_resampled = resample_dem_torch(masked.filled(0), desired_pixel_size, device=device)

    # normalize heights
    height_normalisation_pm = config["HEIGHT_NORMALISATION_PM"]
    dem_normalized = dem_resampled / torch.max(torch.abs(dem_resampled)) * height_normalisation_pm

    return dem_normalized


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
    
def get_lat_lon_radius(config: dict):
    center_lat_deg = config['CENTER_LAT_DEG']
    center_lon_deg = config['CENTER_LON_DEG']
    box_radius_m = config['BOX_RADIUS_M']
    lat_deg_pm = config['CENTER_LAT_DEG_PM']
    lon_deg_pm = config['CENTER_LON_DEG_PM']
    box_radius_m_pm = config['BOX_RADIUS_M_PM']

    if config["STOCHASTIC"]:
        center_lat_deg += np.random.uniform(-lat_deg_pm, lat_deg_pm)
        center_lon_deg += np.random.uniform(-lon_deg_pm, lon_deg_pm)
        box_radius_m += np.random.uniform(-box_radius_m_pm, box_radius_m_pm)

    return center_lat_deg, center_lon_deg, box_radius_m
    


# Alternative version of generate_and_return_lro_dem with resampling
def generate_and_return_lro_dem_alt(config: dict = None):

    dem_path = "/Users/au644271/Desktop/local_python/LRO_data_sandbox/Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014.tif"

    lat, lon, box_radius = get_lat_lon_radius(config)

    dem_array, metadata = extract_local_dem_subset(
                                                    dem_path=dem_path,
                                                    center_lat_deg=lat,
                                                    center_lon_deg=lon,
                                                    box_radius_m=box_radius,
                                                    res_m=config['SAMPLE_RES_M'],
                                                    local_proj_type="stere",
                                                    verbose=False
                                                )
    
    # Resample to desired pixel size
    if dem_array.shape != config["LRO_DEM_SIZE"]:
        # Mask nodata before resampling
        nodata = metadata.get('nodata', -9999)
        mask = dem_array != nodata
        
        # Resample DEM and mask separately
        dem_resampled = cv2.resize(dem_array, 
                                   (config["LRO_DEM_SIZE"], config["LRO_DEM_SIZE"]), # (W, H)
                                   interpolation=cv2.INTER_CUBIC)
        mask_resampled = cv2.resize(mask.astype(np.uint8), 
                                    (config["LRO_DEM_SIZE"], config["LRO_DEM_SIZE"]),
                                    interpolation=cv2.INTER_NEAREST) > 0.5
        
        # Restore nodata values
        dem_resampled[~mask_resampled] = nodata
        dem_array = dem_resampled
    
    return dem_array, metadata