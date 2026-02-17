"""
FM Solver Test Script - Step-by-Step Verification

This script tests the FMSolver with comprehensive debug output and saves
all figures for manual verification. Each major step is clearly separated
with console output and corresponding figure(s).

Figures are saved to: master/FM_solver/figures/
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch

# Project imports
from master.validate.plotting_new import ax_format
from master.data_sim.generator import _render_single_image
from master.render.dem_utils import DEM
from master.render.hapke_model import HapkeModel, LambertianModel
from master.render.camera import Camera
from master.render.renderer import Renderer
from master.lro_data_sim.lro_generator import generate_and_return_lro_dem
from master.configs.config_utils import load_config_file
from master.FM_solver.solver import FMSolver
from master.FM_solver.solver_utils import validate_normal_field, compute_angular_error, remove_outer_n_pixels


def main():
    # Create figures directory
    FIGURES_DIR = os.path.join(os.path.dirname(__file__), 'figures')
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print(f"Figures will be saved to: {FIGURES_DIR}")
    print("=" * 80)

    # =============================================================================
    # STEP 1: Setup and Configuration
    # =============================================================================
    print("\n" + "=" * 80)
    print("STEP 1: Setup and Configuration")
    print("=" * 80)

    config = load_config_file() 

    # Simulation parameters
    LRO_DEM_SIZE = 512
    DEM_SIZE = LRO_DEM_SIZE  # Size of DEM to use for testing (will be downsampled from LRO data)
    IMAGE_W = 512
    IMAGE_H = 512
    FOCAL_LENGTH = 100000.0
    SUN_AZ = 0       # degrees (North)
    SUN_AZ_2 = 90    # degrees (East)
    SUN_EL = 75.0    # degrees
    CAM_AZ = 0       # degrees
    CAM_EL = 90.0    # degrees (overhead)
    CAM_DIST = 100000.0
    HEIGHT_NORMALIZATION = 10.0  # Meters per unit in DEM (for scaling)
    # Random seed for reproducibility
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    #overwrite config with these test values
    for key, value in [("DEM_SIZE", DEM_SIZE), ("IMAGE_W", IMAGE_W), ("IMAGE_H", IMAGE_H), ("FOCAL_LENGTH", FOCAL_LENGTH),
                     ("SUN_AZ", SUN_AZ), ("SUN_AZ_2", SUN_AZ_2), ("SUN_EL", SUN_EL), 
                     ("CAM_AZ", CAM_AZ), ("CAM_EL", CAM_EL), ("CAM_DIST", CAM_DIST), ("SEED", SEED), ("HEIGHT_NORMALIZATION", HEIGHT_NORMALIZATION), ("LRO_DEM_SIZE", LRO_DEM_SIZE)]:
        config[key] = value


    print(f"DEM Size: {config['DEM_SIZE']}x{config['DEM_SIZE']}")
    print(f"Image Size: {IMAGE_W}x{IMAGE_H}")
    print(f"Sun Azimuth 1: {SUN_AZ}° (North)")
    print(f"Sun Azimuth 2: {SUN_AZ_2}° (East)")
    print(f"Sun Elevation: {SUN_EL}°")
    print(f"Camera: Overhead (el={CAM_EL}°)")

    # =============================================================================
    # STEP 2: Generate DEM from LRO Data
    # =============================================================================
    print("\n" + "=" * 80)
    print("STEP 2: Generate DEM from LRO Data")
    print("=" * 80)

    file_location = '/Users/au644271/Desktop/local_python/master/lro_data_sim/Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014.tif'

    print(f"Loading LRO data from: {file_location}")
    dem_tensor, lro_meta = generate_and_return_lro_dem(config, file_location=file_location)


    device = 'cpu'
    dem = dem_tensor.to(device=device, dtype=torch.float32)

    print(f"DEM shape: {dem.shape}")
    print(f"DEM elevation range: [{dem.min():.2f}, {dem.max():.2f}]")
    print(f"DEM mean elevation: {dem.mean():.2f}")

    # Save DEM figure
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(dem.cpu().numpy(), cmap='terrain', origin='lower')
    ax.set_title('Ground Truth DEM from LRO Data', pad=15, fontsize=14)
    fig.colorbar(im, ax=ax, label='Elevation (m)')
    ax_format(ax)
    fig.tight_layout()
    save_path = os.path.join(FIGURES_DIR, '01_ground_truth_dem.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved: {save_path}")

    # =============================================================================
    # STEP 3: Setup Renderer
    # =============================================================================
    print("\n" + "=" * 80)
    print("STEP 3: Setup Renderer (DEM, Hapke Model, Camera)")
    print("=" * 80)

    cellsize = 1  # meters per pixel
    dem_obj = DEM(dem, cellsize=cellsize, x0=0, y0=0)
    # hapke = HapkeModel(w=1, B0=0.4, h=0.1, phase_fun='hg', xi=0.1)
    reflection_model = LambertianModel(w=1.0)  # Use Lambertian for initial testing
    camera = Camera(image_width=IMAGE_W, image_height=IMAGE_H, focal_length=FOCAL_LENGTH, device='cpu')
    renderer = Renderer(dem_obj, reflection_model, camera)

    print(f"DEM object: cellsize={dem_obj.cellsize}, origin=({dem_obj.x0}, {dem_obj.y0})")
    print(f"Reflection model: {reflection_model}")
    print(f"Camera: {IMAGE_W}x{IMAGE_H}, focal_length={FOCAL_LENGTH}")
    print("Renderer ready")

    # =============================================================================
    # STEP 4: Render First Reflectance Map (Sun Azimuth = 0°, North)
    # =============================================================================
    print("\n" + "=" * 80)
    print("STEP 4: Render First Reflectance Map (Sun from North)")
    print("=" * 80)

    params = (SUN_AZ, SUN_EL, CAM_AZ, CAM_EL, CAM_DIST)
    print(f"Rendering with sun az={SUN_AZ}°, el={SUN_EL}°")

    img, reflectance_map = _render_single_image(
        renderer=renderer, 
        params=params, 
        image_w=IMAGE_W, 
        image_h=IMAGE_H
    )

    img = remove_outer_n_pixels(img, n=5, debug=True)
    reflectance_map = remove_outer_n_pixels(reflectance_map, n=5, debug=True)

    print(f"Image shape: {img.shape}")
    print(f"Reflectance map shape: {reflectance_map.shape}")
    print(f"Reflectance range: [{reflectance_map.min():.4f}, {reflectance_map.max():.4f}]")

    # Save first reflectance map
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    im = ax.imshow(reflectance_map.cpu().numpy(), cmap='gray', origin='lower')
    ax.set_title(f'Reflectance Map (Sun az={SUN_AZ}°, North)', pad=15, fontsize=12)
    fig.colorbar(im, ax=ax, label='Reflectance')
    ax_format(ax)

    ax = axes[1]
    im = ax.imshow(img.cpu().numpy(), cmap='gray', origin='lower')
    ax.set_title('Camera Image with Shadows', pad=15, fontsize=12)
    fig.colorbar(im, ax=ax)
    ax_format(ax)

    fig.suptitle(f"First Illumination: Sun az={SUN_AZ}°, el={SUN_EL}°", fontsize=14)
    fig.tight_layout()
    save_path = os.path.join(FIGURES_DIR, '02_reflectance_map_1_north.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved: {save_path}")

    # =============================================================================
    # STEP 5: Render Second Reflectance Map (Sun Azimuth = 90°, East)
    # =============================================================================
    print("\n" + "=" * 80)
    print("STEP 5: Render Second Reflectance Map (Sun from East)")
    print("=" * 80)

    params_2 = (SUN_AZ_2, SUN_EL, CAM_AZ, CAM_EL, CAM_DIST)
    print(f"Rendering with sun az={SUN_AZ_2}°, el={SUN_EL}°")

    img_2, reflectance_map_2 = _render_single_image(
        renderer=renderer, 
        params=params_2, 
        image_w=IMAGE_W, 
        image_h=IMAGE_H
    )

    img_2 = remove_outer_n_pixels(img_2, n=5, debug=False)
    reflectance_map_2 = remove_outer_n_pixels(reflectance_map_2, n=5, debug=False)
    dem = remove_outer_n_pixels(dem, n=5, debug=False)

    print(f"Image shape: {img_2.shape}")
    print(f"Reflectance map 2 shape: {reflectance_map_2.shape}")
    print(f"Reflectance map 2 range: [{reflectance_map_2.min():.4f}, {reflectance_map_2.max():.4f}]")
    
    #compute and print correlation between the two reflectance maps (should be positive but not perfect due to different shadowing)
    corr = np.corrcoef(reflectance_map.cpu().numpy().flatten(), reflectance_map_2.cpu().numpy().flatten())[0, 1]
    print(f"Correlation between reflectance maps: {corr:.4f}")

    # Save comparison of both reflectance maps
    reflectance_np = reflectance_map.detach().cpu().numpy()
    reflectance_np_2 = reflectance_map_2.detach().cpu().numpy()
    vmax = np.max([reflectance_np.max(), reflectance_np_2.max()])
    vmin = np.min([reflectance_np.min(), reflectance_np_2.min()])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    im = ax.imshow(reflectance_np, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    ax.set_title(f'Sun az={SUN_AZ}° (North)', pad=15, fontsize=12)
    fig.colorbar(im, ax=ax, label='Reflectance')
    ax_format(ax)

    ax = axes[1]
    im = ax.imshow(reflectance_np_2, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
    ax.set_title(f'Sun az={SUN_AZ_2}° (East)', pad=15, fontsize=12)
    fig.colorbar(im, ax=ax, label='Reflectance')
    ax_format(ax)

    fig.suptitle("Effect of Changing Sun Azimuth on Reflectance", fontsize=14)
    fig.tight_layout()
    save_path = os.path.join(FIGURES_DIR, '03_reflectance_maps_comparison.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved: {save_path}")

    # =============================================================================
    # STEP 6: Initialize FMSolver with Debug Mode
    # =============================================================================
    print("\n" + "=" * 80)
    print("STEP 6: Initialize FMSolver (with DEBUG enabled)")
    print("=" * 80)

    print("\nInitializing solver with comprehensive debug output...")
    print("-" * 80)

    # Med dette får jeg -7.9%, altså en forbedring i RMSE og MAE
    # solver = FMSolver(
    #     reflectance_maps=[reflectance_map, reflectance_map_2],
    #     sun_azs=[SUN_AZ, SUN_AZ_2],
    #     sun_el=SUN_EL,
    #     gt_dem=dem,
    #     scale_down_factor=40,
    #     sigma_smooth=5,
    #     max_slope_prior=10,
    #     noise_fraction_brightness=0.1,
    #     sigma_data=0.01,
    #     sigma_m=20.0,
    #     pixel_size=cellsize,
    #     debug=True  # Enable comprehensive debug output
    # ) 
    
    
    solver = FMSolver(
        reflectance_maps=[reflectance_map, reflectance_map_2],
        sun_azs=[SUN_AZ, SUN_AZ_2],
        sun_el=SUN_EL,
        gt_dem=dem,
        scale_down_factor=40,
        sigma_smooth=5,
        max_slope_prior=10,
        noise_fraction_brightness=0.1,
        sigma_data=0.01,
        sigma_m=20.0,
        pixel_size=cellsize,
        debug=True  # Enable comprehensive debug output
    )

    print("-" * 80)
    print("Solver initialized successfully")

    # Save prior DEM comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # find max and min vals to use consistent color scales across all three DEMs
    vmin = min(solver.gt_dem.min(), solver.dem_lower.min(), solver.dem_prior.min())
    vmax = max(solver.gt_dem.max(), solver.dem_lower.max(), solver.dem_prior.max())

    axes[0].imshow(solver.gt_dem, cmap='terrain', origin='lower', vmin=vmin, vmax=vmax)
    fig.colorbar(axes[0].images[0], ax=axes[0], label='Elevation (m)')
    axes[0].set_title('Ground Truth DEM', pad=15, fontsize=12)
    ax_format(axes[0])

    axes[1].imshow(solver.dem_lower, cmap='terrain', origin='lower', vmin=vmin, vmax=vmax)
    fig.colorbar(axes[1].images[0], ax=axes[1], label='Elevation (m)')
    axes[1].set_title('Downsampled DEM (40x)', pad=15, fontsize=12)
    ax_format(axes[1])

    # add colorbar to final subplot, but use the same vmin/vmax for consistent interpretation of elevation values    
    im = axes[2].imshow(solver.dem_prior, cmap='terrain', origin='lower', vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=axes[2], label='Elevation (m)')
    axes[2].set_title('Prior DEM (Upsampled + Smoothed)', pad=15, fontsize=12)
    ax_format(axes[2])

    fig.suptitle("DEM Processing for Prior", fontsize=14)
    fig.tight_layout()
    save_path = os.path.join(FIGURES_DIR, '04_dem_prior_processing.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n✓ Saved: {save_path}")

    # =============================================================================
    # STEP 7: Estimate Normals and Albedo
    # =============================================================================
    print("\n" + "=" * 80)
    print("STEP 7: Estimate Surface Normals and Albedo")
    print("=" * 80)

    print("\nEstimating normals with iterative MAP solver...")
    print("-" * 80)

    normals_estimated, albedo_estimated = solver.estimate_normal_and_albedo(n_iters=5)

    print("-" * 80)
    print("Normal and albedo estimation complete")

    # Compute ground truth normals for comparison
    normals_gt = solver.compute_normals_from_dem(solver.gt_dem)
    
    n_est = normals_estimated
    n_gt = normals_gt
    for c, name in enumerate(["nx", "ny", "nz"]):
        est_c = n_est[..., c].ravel()
        gt_c  = n_gt[..., c].ravel()
        corr = np.corrcoef(est_c, gt_c)[0, 1]
        rel_err = np.linalg.norm(est_c - gt_c) / (np.linalg.norm(gt_c) + 1e-12)
        print(f"{name}: corr(est,gt) = {corr:.4f}, rel_err = {rel_err:.4f}")


    # Extract components
    gt_x = normals_gt[..., 0]
    gt_y = normals_gt[..., 1]
    gt_z = normals_gt[..., 2]

    solved_x = normals_estimated[..., 0]
    solved_y = normals_estimated[..., 1]
    solved_z = normals_estimated[..., 2]

    diff_x = solved_x - gt_x
    diff_y = solved_y - gt_y

    # Compute angular errors
    angular_errors = compute_angular_error(normals_estimated, normals_gt)

    print(f"\nNormal Comparison Statistics:")
    print(f"  Angular error mean: {angular_errors.mean():.2f}°")
    print(f"  Angular error median: {np.median(angular_errors):.2f}°")
    print(f"  Angular error std: {angular_errors.std():.2f}°")
    print(f"  Angular error max: {angular_errors.max():.2f}°")
    print(f"  Angular error percentiles:")
    print(f"    50th: {np.percentile(angular_errors, 50):.2f}°")
    print(f"    75th: {np.percentile(angular_errors, 75):.2f}°")
    print(f"    90th: {np.percentile(angular_errors, 90):.2f}°")
    print(f"    95th: {np.percentile(angular_errors, 95):.2f}°")
    print(f"    99th: {np.percentile(angular_errors, 99):.2f}°")

    # Save normal vector comparison WITH CONSISTENT COLOR SCALES
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Compute consistent color scales for X and Y components
    vmin_x = min(gt_x.min(), solved_x.min())
    vmax_x = max(gt_x.max(), solved_x.max())
    vmin_y = min(gt_y.min(), solved_y.min())
    vmax_y = max(gt_y.max(), solved_y.max())
    
    # Symmetric scale for differences
    vmax_diff_x = max(abs(diff_x.min()), abs(diff_x.max()))
    vmax_diff_y = max(abs(diff_y.min()), abs(diff_y.max()))

    im = axes[0, 0].imshow(gt_x, cmap='RdBu_r', origin='lower', vmin=vmin_x, vmax=vmax_x)
    axes[0, 0].set_title('GT Normal X', pad=15, fontsize=11)
    fig.colorbar(im, ax=axes[0, 0])
    ax_format(axes[0, 0])

    im = axes[0, 1].imshow(solved_x, cmap='RdBu_r', origin='lower', vmin=vmin_x, vmax=vmax_x)
    axes[0, 1].set_title('Estimated Normal X', pad=15, fontsize=11)
    fig.colorbar(im, ax=axes[0, 1])
    ax_format(axes[0, 1])

    im = axes[0, 2].imshow(diff_x, cmap='bwr', origin='lower', vmin=-vmax_diff_x, vmax=vmax_diff_x)
    axes[0, 2].set_title(f'Difference X\n(RMSE: {np.sqrt(np.mean(diff_x**2)):.4f})', pad=15, fontsize=11)
    fig.colorbar(im, ax=axes[0, 2])
    ax_format(axes[0, 2])

    im = axes[1, 0].imshow(gt_y, cmap='RdBu_r', origin='lower', vmin=vmin_y, vmax=vmax_y)
    axes[1, 0].set_title('GT Normal Y', pad=15, fontsize=11)
    fig.colorbar(im, ax=axes[1, 0])
    ax_format(axes[1, 0])

    im = axes[1, 1].imshow(solved_y, cmap='RdBu_r', origin='lower', vmin=vmin_y, vmax=vmax_y)
    axes[1, 1].set_title('Estimated Normal Y', pad=15, fontsize=11)
    fig.colorbar(im, ax=axes[1, 1])
    ax_format(axes[1, 1])

    im = axes[1, 2].imshow(diff_y, cmap='bwr', origin='lower', vmin=-vmax_diff_y, vmax=vmax_diff_y)
    axes[1, 2].set_title(f'Difference Y\n(RMSE: {np.sqrt(np.mean(diff_y**2)):.4f})', pad=15, fontsize=11)
    fig.colorbar(im, ax=axes[1, 2])
    ax_format(axes[1, 2])

    fig.suptitle(f"Normal Vector Comparison (Mean Angular Error: {angular_errors.mean():.2f}°)", fontsize=14)
    fig.tight_layout()
    save_path = os.path.join(FIGURES_DIR, '05_normal_vectors_comparison.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved: {save_path}")
    

    nx = normals_estimated[..., 0]
    ny = normals_estimated[..., 1]

    print("nx: min, max, mean:", nx.min(), nx.max(), nx.mean())
    print("ny: min, max, mean:", ny.min(), ny.max(), ny.mean())

    corr = np.corrcoef(nx.ravel(), ny.ravel())[0, 1]
    rel_diff = np.linalg.norm(nx - ny) / (np.linalg.norm(nx) + 1e-12)

    print("corr(nx, ny) =", corr)
    print("relative ||nx - ny|| =", rel_diff)



    # # =============================================================================
    # # STEP 7b: RGB Normal Map Visualization
    # # =============================================================================
    # print("\nGenerating RGB normal map visualization...")
    
    # # Convert normals to RGB (map [-1,1] to [0,1])
    # normals_gt_rgb = (normals_gt + 1) / 2
    # normals_estimated_rgb = (normals_estimated + 1) / 2
    # normals_diff_rgb = np.abs(normals_gt_rgb - normals_estimated_rgb)
    
    # fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # axes[0].imshow(normals_gt_rgb, origin='lower')
    # axes[0].set_title('GT Normals (RGB)\nR=X(East), G=Y(North), B=Z(Up)', pad=15, fontsize=12)
    # ax_format(axes[0])
    
    # axes[1].imshow(normals_estimated_rgb, origin='lower')
    # axes[1].set_title('Estimated Normals (RGB)\nR=X(East), G=Y(North), B=Z(Up)', pad=15, fontsize=12)
    # ax_format(axes[1])
    
    # im = axes[2].imshow(normals_diff_rgb, origin='lower', vmin=0, vmax=0.3)
    # axes[2].set_title(f'Absolute Difference (RGB)\nMean Diff: {normals_diff_rgb.mean():.4f}', pad=15, fontsize=12)
    # fig.colorbar(im, ax=axes[2], label='Color Difference')
    # ax_format(axes[2])
    
    # fig.suptitle('RGB Normal Map Comparison - Color Shows Surface Orientation', fontsize=14)
    # fig.tight_layout()
    # save_path = os.path.join(FIGURES_DIR, '05b_normal_rgb_comparison.png')
    # fig.savefig(save_path, dpi=150, bbox_inches='tight')
    # plt.close(fig)
    # print(f"✓ Saved: {save_path}")

    # =============================================================================
    # STEP 7c: Synthetic Reflectance Validation
    # =============================================================================
    print("\nValidating normals with synthetic reflectance...")
    
    # Compute sun direction vectors (matching solver convention)
    sun_vecs = solver.sun_normals  # Sun from North
    
    # Compute synthetic reflectance from estimated normals (Lambertian: R = n·s)
    R_synth_1 = np.maximum(0, np.sum(normals_estimated * sun_vecs[0], axis=-1))/np.pi  # divide by pi to match LambertianModel scaling
    R_synth_2 = np.maximum(0, np.sum(normals_estimated * sun_vecs[1], axis=-1))/np.pi  # Sun from East
    
    # Get actual reflectance maps (convert from tensors)
    # Note: With w=π in LambertianModel, renderer outputs R = (w/π)*cos(θ) = cos(θ)
    # which already matches the solver's forward model R = albedo * (n·s)
    R_actual_1 = reflectance_map.cpu().numpy()
    R_actual_2 = reflectance_map_2.cpu().numpy()
    
    # Compute residuals
    residual_1 = R_actual_1 - R_synth_1
    residual_2 = R_actual_2 - R_synth_2
    
    rmse_1 = np.sqrt(np.mean(residual_1**2))
    rmse_2 = np.sqrt(np.mean(residual_2**2))
    
    print(f"  Illumination 1 (North):")
    print(f"    RMSE: {rmse_1:.4f}")
    print(f"    Relative RMSE: {100*rmse_1/R_actual_1.mean():.2f}%")
    print(f"  Illumination 2 (East):")
    print(f"    RMSE: {rmse_2:.4f}")
    print(f"    Relative RMSE: {100*rmse_2/R_actual_2.mean():.2f}%")
    
    # Plot synthetic vs actual reflectance
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Determine common color scale for each illumination
    vmin_1 = min(R_actual_1.min(), R_synth_1.min())
    vmax_1 = max(R_actual_1.max(), R_synth_1.max())
    vmin_2 = min(R_actual_2.min(), R_synth_2.min())
    vmax_2 = max(R_actual_2.max(), R_synth_2.max())
    
    # Illumination 1 (North)
    im = axes[0, 0].imshow(R_actual_1, cmap='gray', origin='lower', vmin=vmin_1, vmax=vmax_1)
    axes[0, 0].set_title('Actual Reflectance\n(Sun North)', pad=15, fontsize=11)
    fig.colorbar(im, ax=axes[0, 0])
    ax_format(axes[0, 0])
    
    im = axes[0, 1].imshow(R_synth_1, cmap='gray', origin='lower', vmin=vmin_1, vmax=vmax_1)
    axes[0, 1].set_title('Synthetic Reflectance\n(from Estimated Normals)', pad=15, fontsize=11)
    fig.colorbar(im, ax=axes[0, 1])
    ax_format(axes[0, 1])
    
    vmax_res_1 = max(abs(residual_1.min()), abs(residual_1.max()))
    im = axes[0, 2].imshow(residual_1, cmap='bwr', origin='lower', vmin=-vmax_res_1, vmax=vmax_res_1)
    axes[0, 2].set_title(f'Residual\nRMSE: {rmse_1:.4f} ({100*rmse_1/R_actual_1.mean():.1f}\%)', pad=15, fontsize=11)
    fig.colorbar(im, ax=axes[0, 2])
    ax_format(axes[0, 2])
    
    # Illumination 2 (East)
    im = axes[1, 0].imshow(R_actual_2, cmap='gray', origin='lower', vmin=vmin_2, vmax=vmax_2)
    axes[1, 0].set_title('Actual Reflectance\n(Sun East)', pad=15, fontsize=11)
    fig.colorbar(im, ax=axes[1, 0])
    ax_format(axes[1, 0])
    
    im = axes[1, 1].imshow(R_synth_2, cmap='gray', origin='lower', vmin=vmin_2, vmax=vmax_2)
    axes[1, 1].set_title('Synthetic Reflectance\n(from Estimated Normals)', pad=15, fontsize=11)
    fig.colorbar(im, ax=axes[1, 1])
    ax_format(axes[1, 1])
    
    vmax_res_2 = max(abs(residual_2.min()), abs(residual_2.max()))
    im = axes[1, 2].imshow(residual_2, cmap='bwr', origin='lower', vmin=-vmax_res_2, vmax=vmax_res_2)
    axes[1, 2].set_title(f'Residual\nRMSE: {rmse_2:.4f} ({100*rmse_2/R_actual_2.mean():.1f}\%)', pad=15, fontsize=11)
    fig.colorbar(im, ax=axes[1, 2])
    ax_format(axes[1, 2])
    
    fig.suptitle('Synthetic Reflectance Validation - Low Residuals Confirm Correct Normals', fontsize=14)
    fig.tight_layout()
    save_path = os.path.join(FIGURES_DIR, '05c_synthetic_reflectance_validation.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved: {save_path}")

    # Save albedo figure
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(albedo_estimated, cmap='gray', origin='lower')
    ax.set_title('Estimated Albedo', pad=15, fontsize=14)
    fig.colorbar(im, ax=ax, label='Albedo')
    ax_format(ax)
    fig.tight_layout()
    save_path = os.path.join(FIGURES_DIR, '06_estimated_albedo.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved: {save_path}")

    print(f"\nAlbedo Statistics:")
    print(f"  Min: {albedo_estimated.min():.4f}")
    print(f"  Max: {albedo_estimated.max():.4f}")
    print(f"  Mean: {albedo_estimated.mean():.4f}")
    print(f"  Std: {albedo_estimated.std():.4f}")
    
    # Check for correlation with reflectance (should be low if albedo is constant)
    corr_1 = np.corrcoef(albedo_estimated.flatten(), reflectance_map.cpu().numpy().flatten())[0, 1]
    corr_2 = np.corrcoef(albedo_estimated.flatten(), reflectance_map_2.cpu().numpy().flatten())[0, 1]
    print(f"  Correlation with Reflectance Map 1 (North): {corr_1:.4f} (R² = {corr_1**2:.4f})")
    print(f"  Correlation with Reflectance Map 2 (East): {corr_2:.4f} (R² = {corr_2**2:.4f})")
    
    # Analyze prior normals for bias
    print(f"\n  Prior Normal Statistics:")
    normals_prior = solver.compute_normals_from_dem(solver.dem_prior)
    prior_x = normals_prior[..., 0]
    prior_y = normals_prior[..., 1]
    print(f"    Prior Normal X: mean={prior_x.mean():.4f}, std={prior_x.std():.4f}")
    print(f"    Prior Normal Y: mean={prior_y.mean():.4f}, std={prior_y.std():.4f}")
    print(f"    GT Normal X: mean={gt_x.mean():.4f}, std={gt_x.std():.4f}")
    print(f"    GT Normal Y: mean={gt_y.mean():.4f}, std={gt_y.std():.4f}")
    print(f"    Estimated Normal X: mean={solved_x.mean():.4f}, std={solved_x.std():.4f}")
    print(f"    Estimated Normal Y: mean={solved_y.mean():.4f}, std={solved_y.std():.4f}")

    # Quick unit consistency check for gradients with cellsize
    print("\nChecking gradient scaling with cellsize...")
    dz_dy_gt = np.gradient(solver.gt_dem, solver.dy, axis=0)
    dz_dx_gt = np.gradient(solver.gt_dem, solver.dx, axis=1)
    dz_dy_prior = np.gradient(solver.dem_prior, solver.dy, axis=0)
    dz_dx_prior = np.gradient(solver.dem_prior, solver.dx, axis=1)

    mean_gt_dx = np.mean(np.abs(dz_dx_gt))
    mean_gt_dy = np.mean(np.abs(dz_dy_gt))
    mean_prior_dx = np.mean(np.abs(dz_dx_prior))
    mean_prior_dy = np.mean(np.abs(dz_dy_prior))

    print(f"  Mean |dz/dx| GT:    {mean_gt_dx:.6f}")
    print(f"  Mean |dz/dx| Prior: {mean_prior_dx:.6f}")
    print(f"  Mean |dz/dy| GT:    {mean_gt_dy:.6f}")
    print(f"  Mean |dz/dy| Prior: {mean_prior_dy:.6f}")

    ratio_x = mean_prior_dx / (mean_gt_dx + 1e-12)
    ratio_y = mean_prior_dy / (mean_gt_dy + 1e-12)
    print(f"  Scale ratio (prior/gt): x={ratio_x:.3f}, y={ratio_y:.3f}")
    if ratio_x > 4 or ratio_x < 0.25 or ratio_y > 4 or ratio_y < 0.25:
        print("  WARNING: Prior gradient scale differs significantly from GT; check cellsize usage.")

    # =============================================================================
    # STEP 7d: Albedo-Reflectance Correlation Analysis
    # =============================================================================
    print("\nAnalyzing albedo-reflectance correlations...")
    
    # Create scatter plots to visualize correlations
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Scatter: Albedo vs Reflectance 1
    ax = axes[0]
    refl_1_np = reflectance_map.cpu().numpy().flatten()
    albedo_flat = albedo_estimated.flatten()
    ax.scatter(refl_1_np, albedo_flat, alpha=0.1, s=1, c='blue')
    ax.set_xlabel('Reflectance Map 1 (North)', fontsize=11)
    ax.set_ylabel('Estimated Albedo', fontsize=11)
    ax.set_title(f'Albedo vs Reflectance 1\nCorr={corr_1:.4f}, R²={corr_1**2:.4f}', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Scatter: Albedo vs Reflectance 2
    ax = axes[1]
    refl_2_np = reflectance_map_2.cpu().numpy().flatten()
    ax.scatter(refl_2_np, albedo_flat, alpha=0.1, s=1, c='green')
    ax.set_xlabel('Reflectance Map 2 (East)', fontsize=11)
    ax.set_ylabel('Estimated Albedo', fontsize=11)
    ax.set_title(f'Albedo vs Reflectance 2\nCorr={corr_2:.4f}, R²={corr_2**2:.4f}', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Spatial map: Albedo deviation from mean
    ax = axes[2]
    albedo_deviation = albedo_estimated - albedo_estimated.mean()
    im = ax.imshow(albedo_deviation, cmap='RdBu_r', origin='lower', 
                   vmin=-3*albedo_estimated.std(), vmax=3*albedo_estimated.std())
    ax.set_title(f'Albedo Deviation from Mean\n(Std={albedo_estimated.std():.4f})', fontsize=12)
    fig.colorbar(im, ax=ax, label='Deviation')
    ax_format(ax)
    
    fig.suptitle('Albedo Correlation Analysis - Investigating Non-Zero Correlation with North Illumination', fontsize=14)
    fig.tight_layout()
    save_path = os.path.join(FIGURES_DIR, '06b_albedo_correlation_analysis.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved: {save_path}")
    
    # Correlation with surface slopes (gradients)
    slope_magnitude = np.sqrt(dz_dx_gt**2 + dz_dy_gt**2)
    
    corr_slope = np.corrcoef(albedo_flat, slope_magnitude.flatten())[0, 1]
    corr_dz_dy = np.corrcoef(albedo_flat, dz_dy_gt.flatten())[0, 1]
    corr_dz_dx = np.corrcoef(albedo_flat, dz_dx_gt.flatten())[0, 1]
    
    print(f"\n  Correlation with terrain properties:")
    print(f"    Slope magnitude: {corr_slope:.4f}")
    print(f"    dz/dy (North gradient): {corr_dz_dy:.4f}")
    print(f"    dz/dx (East gradient): {corr_dz_dx:.4f}")

    # =============================================================================
    # STEP 8: Compute DEM Update via Sylvester Equation
    # =============================================================================
    print("\n" + "=" * 80)
    print("STEP 8: Compute DEM Update via Sylvester Equation")
    print("=" * 80)

    print("\nConstructing and solving Sylvester equation...")
    print("-" * 80)

    # Use tuned regularization for stable updates
    M_update = solver.compute_model_update()

    print("-" * 80)
    print("DEM update computed successfully")

    print(f"\nUpdate Statistics:")
    print(f"  Shape: {M_update.shape}")
    print(f"  Range: [{M_update.min():.4f}, {M_update.max():.4f}]")
    print(f"  Mean: {M_update.mean():.4f}")
    print(f"  Std: {M_update.std():.4f}")

    dem_initial = solver.dem_prior

    # Compute updated DEM (additive update applied to full prior; interior filled, border zero)
    M_updated = solver.dem_prior + M_update

    # Save DEM update visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))

    im = axes[0, 0].imshow(solver.dem_prior, cmap='terrain', origin='lower')
    axes[0, 0].set_title('Prior DEM', pad=15, fontsize=12)
    fig.colorbar(im, ax=axes[0, 0], label='Elevation (m)')
    ax_format(axes[0, 0])

    im = axes[0, 1].imshow(M_update, cmap='bwr', origin='lower')
    axes[0, 1].set_title(f'DEM Update\n(Full, border zero)', pad=15, fontsize=12)
    fig.colorbar(im, ax=axes[0, 1], label='Update (m)')
    ax_format(axes[0, 1])

    im = axes[1, 0].imshow(M_updated, cmap='terrain', origin='lower')
    axes[1, 0].set_title('Updated DEM\n(Prior + Update)', pad=15, fontsize=12)
    fig.colorbar(im, ax=axes[1, 0], label='Elevation (m)')
    ax_format(axes[1, 0])

    # Compare with ground truth interior
    gt_interior = solver.gt_dem[1:-1, 1:-1]
    im = axes[1, 1].imshow(gt_interior, cmap='terrain', origin='lower')
    axes[1, 1].set_title('Ground Truth DEM\n(Interior for Comparison)', pad=15, fontsize=12)
    fig.colorbar(im, ax=axes[1, 1], label='Elevation (m)')
    ax_format(axes[1, 1])

    fig.suptitle("DEM Update Process", fontsize=14)
    fig.tight_layout()
    save_path = os.path.join(FIGURES_DIR, '07_dem_update_process.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved: {save_path}")

    # =============================================================================
    # STEP 9: Final Comparison and Error Analysis
    # =============================================================================
    print("\n" + "=" * 80)
    print("STEP 9: Final Comparison and Error Analysis")
    print("=" * 80)

    # Compute errors
    error_prior = gt_interior - solver.dem_prior[1:-1, 1:-1]
    error_updated = gt_interior - M_updated[1:-1, 1:-1]

    rmse_prior = np.sqrt(np.mean(error_prior**2))
    rmse_updated = np.sqrt(np.mean(error_updated**2))
    mae_prior = np.mean(np.abs(error_prior))
    mae_updated = np.mean(np.abs(error_updated))

    print(f"\nDEM Error Statistics:")
    print(f"  Prior DEM:")
    print(f"    RMSE: {rmse_prior:.4f} m")
    print(f"    MAE:  {mae_prior:.4f} m")
    print(f"  Updated DEM:")
    print(f"    RMSE: {rmse_updated:.4f} m")
    print(f"    MAE:  {mae_updated:.4f} m")
    print(f"  Improvement:")
    print(f"    RMSE change: {rmse_updated - rmse_prior:+.4f} m ({100*(rmse_updated-rmse_prior)/rmse_prior:+.1f}%)")
    print(f"    MAE change:  {mae_updated - mae_prior:+.4f} m ({100*(mae_updated-mae_prior)/mae_prior:+.1f}%)")

    # Save error comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    im = axes[0].imshow(error_prior, cmap='bwr', origin='lower')
    axes[0].set_title(f'Prior DEM Error\nRMSE: {rmse_prior:.4f} m', pad=15, fontsize=12)
    fig.colorbar(im, ax=axes[0], label='Error (m)')
    ax_format(axes[0])

    im = axes[1].imshow(error_updated, cmap='bwr', origin='lower')
    axes[1].set_title(f'Updated DEM Error\nRMSE: {rmse_updated:.4f} m', pad=15, fontsize=12)
    fig.colorbar(im, ax=axes[1], label='Error (m)')
    ax_format(axes[1])

    im = axes[2].imshow(error_updated - error_prior, cmap='bwr', origin='lower')
    axes[2].set_title('Error Change\n(Updated - Prior)', pad=15, fontsize=12)
    fig.colorbar(im, ax=axes[2], label='Change (m)')
    ax_format(axes[2])

    fig.suptitle("DEM Error Analysis", fontsize=14)
    fig.tight_layout()
    save_path = os.path.join(FIGURES_DIR, '08_dem_error_analysis.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved: {save_path}")


    # now iterate a few more times to see if we can get further improvement
    print("\nIterating additional DEM updates to observe convergence...")
    n_additional_iters = 100
    rmse_history = [rmse_prior, rmse_updated]
    mae_history = [mae_prior, mae_updated]
    
    M_update_total = M_update.copy()  # Keep track of total update applied
    
    solver.debug = False  # Disable debug for faster iterations
    
    for i in range(n_additional_iters):
        print(f"\nIteration {i+1}/{n_additional_iters}...")
        solver.dem_prior = M_updated  # Update prior for next iteration
        M_update = solver.compute_model_update()
        M_update_total += M_update  # Accumulate total update
        M_updated = solver.dem_prior + M_update
        
        error_updated = gt_interior - M_updated[1:-1, 1:-1]
        rmse_updated = np.sqrt(np.mean(error_updated**2))
        mae_updated = np.mean(np.abs(error_updated))
        
        # calculate improvement from previous iteration in percentage terms
        rmse_improvement = rmse_history[-1] - rmse_updated
        mae_improvement = mae_history[-1] - mae_updated
        rmse_improvement_pct = 100 * (rmse_improvement / rmse_history[-1]) if rmse_history[-1] > 0 else 0
        mae_improvement_pct = 100 * (mae_improvement / mae_history[-1]) if mae_history[-1] > 0 else 0
        
        rmse_history.append(rmse_updated)
        mae_history.append(mae_updated)
        
        print(f"  Updated DEM RMSE: {rmse_updated:.4f} m (change: {rmse_improvement:.4f} m, {-rmse_improvement_pct:+.1f}%)")
        print(f"  Updated DEM MAE:  {mae_updated:.4f} m (change: {mae_improvement:.4f} m, {-mae_improvement_pct:+.1f}%)")

    # Save DEM update visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))

    im = axes[0, 0].imshow(dem_initial, cmap='terrain', origin='lower')
    axes[0, 0].set_title('Prior DEM', pad=15, fontsize=12)
    fig.colorbar(im, ax=axes[0, 0], label='Elevation (m)')
    ax_format(axes[0, 0])

    im = axes[0, 1].imshow(M_update_total, cmap='bwr', origin='lower')
    axes[0, 1].set_title(f'DEM Total Update\n(Full, border zero)', pad=15, fontsize=12)
    fig.colorbar(im, ax=axes[0, 1], label='Update (m)')
    ax_format(axes[0, 1])

    im = axes[1, 0].imshow(M_updated, cmap='terrain', origin='lower')
    axes[1, 0].set_title('Updated DEM\n(Prior + Update)', pad=15, fontsize=12)
    fig.colorbar(im, ax=axes[1, 0], label='Elevation (m)')
    ax_format(axes[1, 0])

    # Compare with ground truth interior
    gt_interior = solver.gt_dem[1:-1, 1:-1]
    im = axes[1, 1].imshow(gt_interior, cmap='terrain', origin='lower')
    axes[1, 1].set_title('Ground Truth DEM\n(Interior for Comparison)', pad=15, fontsize=12)
    fig.colorbar(im, ax=axes[1, 1], label='Elevation (m)')
    ax_format(axes[1, 1])

    fig.suptitle(f"DEM Update Process after {i+1} iterations", fontsize=14)
    fig.tight_layout()
    save_path = os.path.join(FIGURES_DIR, '09_dem_after_n_updates.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved: {save_path}")
    
    # print total change in RMSE and RMA from appended history
    print(f"\nFinal DEM Error after {i+1} iterations:")
    print(f"  RMSE: {rmse_history[-1]:.4f} m (total change: {rmse_history[0] - rmse_history[-1]:.4f} m, {-100*(rmse_history[0] - rmse_history[-1])/rmse_history[0]:.1f}%)")
    print(f"  MAE:  {mae_history[-1]:.4f} m (total change: {mae_history[0] - mae_history[-1]:.4f} m, {-100*(mae_history[0] - mae_history[-1])/mae_history[0]:.1f}%)")
    # also plot the RMSE and MAE history over iterations
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(rmse_history, label='RMSE', marker='o')
    ax.plot(mae_history, label='MAE', marker='o')
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Error (m)', fontsize=11)
    ax.set_title('DEM Error History over Iterations', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    save_path = os.path.join(FIGURES_DIR, '10_dem_error_history.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved: {save_path}")

    
    
    
if __name__ == "__main__":
    main()