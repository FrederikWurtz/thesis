# Imports and parameters
import numpy as np
import matplotlib.pyplot as plt
import torch as torch
import math as math
import torch.nn.functional as F

import os

# Project utilities (same folder)
from master.data_sim.generator import _generate_synthetic_dem, _render_single_image
from master.render.dem_utils import DEM
from master.render.hapke_model import HapkeModel
from master.render.camera import Camera
from master.render.renderer import Renderer
from scipy.ndimage import label


def main():
    # Example parameters (tweak as needed)
    DEM_SIZE = 512
    IMAGE_W = 128
    IMAGE_H = 128
    SUN_AZ = 90.0   # degrees
    SUN_EL = 20.0   # degrees
    CAM_AZ = 180.0  # degrees
    CAM_EL = 80.0   # degrees
    CAM_DIST = 2000.0


    # Create a synthetic DEM and show it
    # We use the project's generator so the DEM looks like what training uses
    np.random.seed(20)
    dem_size = 512
    n_craters = 0
    n_ridges = 1
    n_hills = 2
    crater_depth_range = (20, 50)
    crater_radius_range = (30, 80)
    ridge_height_range = (30, 40)
    ridge_length_range = (0.05, 0.1)
    ridge_width_range = (0.05, 0.1)
    hill_height_range = (30, 40)
    hill_sigma_range = (0.5, 0.1)

    dem = _generate_synthetic_dem(size=dem_size, 
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
    print('DEM shape:', dem.shape)

    # # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), 'test_results')
    # os.makedirs(output_dir, exist_ok=True)

    # plt.figure(figsize=(6,6))
    # plt.imshow(dem, cmap='terrain', origin='lower')
    # plt.title('Synthetic DEM')
    # plt.colorbar(label='elevation (m)')
    # plt.savefig(os.path.join(output_dir, 'synthetic_dem.png'), dpi=150, bbox_inches='tight')
    # plt.close()
    # print(f"✓ Saved: synthetic_dem.png")


    # Render reflectance map and camera image using renderer
    # Build objects used by render_single_image


    DEM_SIZE = 512
    IMAGE_W = 128*3
    IMAGE_H = 128*3
    FOCAL_LENGTH = 2000.0
    SUN_AZ = 0   # degrees
    SUN_EL = 10.0   # degrees
    CAM_AZ = 180  # degrees
    CAM_EL = 60.0   # degrees
    CAM_DIST = 3000.0

    dem_obj = DEM(dem, cellsize=1, x0=0, y0=0)
    hapke = HapkeModel(w=0.6, B0=0.4, h=0.1, phase_fun='hg', xi=0.1)
    camera = Camera(image_width=IMAGE_W, image_height=IMAGE_H, focal_length=FOCAL_LENGTH, device='cpu')
    renderer = Renderer(dem_obj, hapke, camera)


    params = (SUN_AZ, SUN_EL, CAM_AZ, CAM_EL, CAM_DIST)

    # render_single_image will compute its own shadow map internally as implemented in the project
    img, reflectance_map = _render_single_image(renderer=renderer, params=params, image_w=IMAGE_W, image_h=IMAGE_H)

    print('Image shape:', img.shape)
    print('Reflectance map shape:', reflectance_map.shape)

    # Plot reflectance map and camera image
    plt.figure(figsize=(18,6))

    plt.subplot(1,3,1)
    plt.imshow(dem, cmap='terrain', origin='lower')
    plt.title('Synthetic DEM')
    plt.colorbar(label='elevation (m)')

    plt.subplot(1,3,2)
    plt.imshow(reflectance_map, cmap='gray', origin='lower')
    plt.title('Reflectance map (with shadows applied)')
    plt.colorbar()

    plt.subplot(1,3,3)
    plt.imshow(img, cmap='gray', origin='lower')
    plt.title('Camera image')

    plt.suptitle(f"Sun az={SUN_AZ:.1f}°, el={SUN_EL:.1f}°")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'rendering_az{SUN_AZ:.0f}_el{SUN_EL:.0f}.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: rendering_az{SUN_AZ:.0f}_el{SUN_EL:.0f}.png")
    
    print("\n" + "="*80)
    print(f"All plots saved to {output_dir}")
    print("="*80)

if __name__ == '__main__':
    main()
