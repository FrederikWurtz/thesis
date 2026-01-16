"""
Step-by-step test for image generation pipeline with visualization.

This test file systematically validates each step of the LRO image generation 
pipeline to identify issues with the cummax shadow computation.

Steps tested:
1. DEM generation from LRO data
2. DEM normalization and statistics
3. Coordinate computation (world points)
4. Normal vector computation
5. Sun and camera vector computation
6. Incidence/emission angle computation
7. Hapke reflectance computation (without shadows)
8. Shadow map computation (with cummax)
9. Shadow application to reflectance
10. Camera projection and image rendering

Each step is visualized to identify where the pipeline fails.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Add master to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from master.configs.config_utils import load_config_file
from master.lro_data_sim.lro_generator import generate_and_return_lro_dem, get_lat_lon_radius
from master.render.dem_utils import DEM
from master.render.hapke_model import HapkeModel
from master.render.camera import Camera
from master.render.renderer import Renderer


def plot_tensor(tensor, title, ax, cmap='viridis', vmin=None, vmax=None, show_stats=True):
    """Helper to plot a 2D tensor with statistics."""
    if torch.is_tensor(tensor):
        data = tensor.detach().cpu().numpy()
    else:
        data = np.array(tensor)
    
    # Handle 1D or flat arrays
    if data.ndim == 1:
        # Reshape to square if possible
        side = int(np.sqrt(len(data)))
        if side * side == len(data):
            data = data.reshape(side, side)
        else:
            # Just plot as 1D
            ax.plot(data)
            ax.set_title(title)
            ax.grid(True)
            return
    
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    if show_stats:
        stats_text = (f"min: {np.nanmin(data):.4f}\n"
                     f"max: {np.nanmax(data):.4f}\n"
                     f"mean: {np.nanmean(data):.4f}\n"
                     f"std: {np.nanstd(data):.4f}")
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', 
                facecolor='white', alpha=0.8), fontsize=8)


def test_step_1_dem_generation(config, device='cpu', output_dir='.'):
    """Step 1: Generate DEM from LRO data and visualize."""
    print("\n" + "="*80)
    print("STEP 1: DEM Generation from LRO Data")
    print("="*80)
    
    dem_tensor = generate_and_return_lro_dem(config).to(device)
    
    print(f"DEM shape: {dem_tensor.shape}")
    print(f"DEM dtype: {dem_tensor.dtype}")
    print(f"DEM device: {dem_tensor.device}")
    print(f"DEM stats:")
    print(f"  Min: {dem_tensor.min().item():.4f}")
    print(f"  Max: {dem_tensor.max().item():.4f}")
    print(f"  Mean: {dem_tensor.mean().item():.4f}")
    print(f"  Std: {dem_tensor.std().item():.4f}")
    print(f"  Contains NaN: {torch.isnan(dem_tensor).any().item()}")
    print(f"  Contains Inf: {torch.isinf(dem_tensor).any().item()}")
    
    # Plot DEM
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    plot_tensor(dem_tensor, 'DEM Elevation', axes[0], cmap='terrain')
    
    # Plot histogram
    dem_np = dem_tensor.cpu().numpy().flatten()
    axes[1].hist(dem_np, bins=100, edgecolor='black')
    axes[1].set_title('DEM Elevation Distribution')
    axes[1].set_xlabel('Elevation')
    axes[1].set_ylabel('Count')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'step_1_dem_generation.png'), dpi=150)
    print("  ✓ Saved: step_1_dem_generation.png")
    plt.close()
    
    return dem_tensor


def test_step_2_dem_object_creation(dem_tensor, device='cpu', output_dir='.'):
    """Step 2: Create DEM object and compute normals."""
    print("\n" + "="*80)
    print("STEP 2: DEM Object Creation and Normal Computation")
    print("="*80)
    
    dem_obj = DEM(dem_tensor, cellsize=1, x0=0, y0=0)
    
    print(f"DEM object created")
    print(f"  Width: {dem_obj.width}, Height: {dem_obj.height}")
    print(f"  Cellsize: {dem_obj.cellsize}")
    print(f"  World points shape: {dem_obj.world_points.shape}")
    
    # Check normals
    print(f"\nNormal vectors:")
    print(f"  nx stats: min={dem_obj.nx.min().item():.4f}, max={dem_obj.nx.max().item():.4f}")
    print(f"  ny stats: min={dem_obj.ny.min().item():.4f}, max={dem_obj.ny.max().item():.4f}")
    print(f"  nz stats: min={dem_obj.nz.min().item():.4f}, max={dem_obj.nz.max().item():.4f}")
    
    # Compute normal magnitude (should be ~1 everywhere)
    normal_mag = torch.sqrt(dem_obj.nx**2 + dem_obj.ny**2 + dem_obj.nz**2)
    print(f"  Normal magnitude: min={normal_mag.min().item():.4f}, max={normal_mag.max().item():.4f}")
    
    # Plot normals
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    plot_tensor(dem_obj.nx, 'Normal X Component', axes[0, 0], cmap='RdBu_r', vmin=-1, vmax=1)
    plot_tensor(dem_obj.ny, 'Normal Y Component', axes[0, 1], cmap='RdBu_r', vmin=-1, vmax=1)
    plot_tensor(dem_obj.nz, 'Normal Z Component', axes[1, 0], cmap='viridis', vmin=0, vmax=1)
    plot_tensor(normal_mag, 'Normal Magnitude (should be ~1)', axes[1, 1], cmap='viridis')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'step_2_normals.png'), dpi=150)
    print("  ✓ Saved: step_2_normals.png")
    plt.close()
    
    return dem_obj


def test_step_3_lighting_geometry(dem_obj, sun_az_deg=0, sun_el_deg=25, 
                                  cam_az_deg=0, cam_el_deg=55, cam_dist=3000, output_dir='.'):
    """Step 3: Compute sun and view vectors, incidence and emission angles."""
    print("\n" + "="*80)
    print("STEP 3: Lighting Geometry")
    print("="*80)
    print(f"Sun: azimuth={sun_az_deg}°, elevation={sun_el_deg}°")
    print(f"Camera: azimuth={cam_az_deg}°, elevation={cam_el_deg}°, distance={cam_dist}")
    
    device = dem_obj.device
    
    # Compute sun direction
    sun_vec = Camera.unit_vec_from_az_el(sun_az_deg, sun_el_deg, device=device)
    sx, sy, sz = sun_vec[0], sun_vec[1], sun_vec[2]
    print(f"\nSun vector: [{sx:.4f}, {sy:.4f}, {sz:.4f}]")
    print(f"  Magnitude: {torch.norm(sun_vec).item():.4f}")
    
    # Compute camera position
    center_x = dem_obj.x0 + (dem_obj.width * dem_obj.cellsize) / 2
    center_y = dem_obj.y0 + (dem_obj.height * dem_obj.cellsize) / 2
    
    camera_az_rad = torch.deg2rad(torch.tensor(cam_az_deg, dtype=torch.float32, device=device))
    camera_el_rad = torch.deg2rad(torch.tensor(cam_el_deg, dtype=torch.float32, device=device))
    cx = center_x + cam_dist * torch.sin(camera_az_rad) * torch.cos(camera_el_rad)
    cy = center_y + cam_dist * torch.cos(camera_az_rad) * torch.cos(camera_el_rad)
    cz = cam_dist * torch.sin(camera_el_rad)
    camera_pos = torch.stack([cx, cy, cz])
    
    print(f"\nCamera position: [{cx:.2f}, {cy:.2f}, {cz:.2f}]")
    
    # Compute view vectors
    view_vectors = camera_pos - dem_obj.world_points
    view_vectors = view_vectors / torch.norm(view_vectors, dim=1, keepdim=True)
    
    # Flatten normals
    nx_flat = dem_obj.nx.flatten()
    ny_flat = dem_obj.ny.flatten()
    nz_flat = dem_obj.nz.flatten()
    
    # Compute angles
    mu = (nx_flat*view_vectors[:,0] + ny_flat*view_vectors[:,1] + 
          nz_flat*view_vectors[:,2]).reshape(dem_obj.dem.shape)
    mu0 = (nx_flat*sx + ny_flat*sy + nz_flat*sz).reshape(dem_obj.dem.shape)
    
    print(f"\nIncidence angle cosine (mu0):")
    print(f"  Min: {mu0.min().item():.4f}, Max: {mu0.max().item():.4f}")
    print(f"  Mean: {mu0.mean().item():.4f}")
    print(f"  Positive values (lit): {(mu0 > 0).sum().item()} / {mu0.numel()}")
    
    print(f"\nEmission angle cosine (mu):")
    print(f"  Min: {mu.min().item():.4f}, Max: {mu.max().item():.4f}")
    print(f"  Mean: {mu.mean().item():.4f}")
    print(f"  Positive values (visible): {(mu > 0).sum().item()} / {mu.numel()}")
    
    # Plot angles
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    plot_tensor(mu0, 'Incidence Angle Cosine (mu0)\nPositive = sunlit', 
                axes[0, 0], cmap='RdYlGn', vmin=-1, vmax=1)
    plot_tensor(mu, 'Emission Angle Cosine (mu)\nPositive = visible to camera', 
                axes[0, 1], cmap='RdYlGn', vmin=-1, vmax=1)
    
    # Binary masks
    lit_mask = (mu0 > 0).float()
    visible_mask = (mu > 0).float()
    both_mask = lit_mask * visible_mask
    
    plot_tensor(lit_mask, 'Sunlit Mask (mu0 > 0)', axes[1, 0], cmap='gray', vmin=0, vmax=1)
    plot_tensor(both_mask, 'Sunlit AND Visible Mask', axes[1, 1], cmap='gray', vmin=0, vmax=1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'step_3_lighting_geometry.png'), dpi=150)
    print("  ✓ Saved: step_3_lighting_geometry.png")
    plt.close()
    
    return mu0, mu, camera_pos


def test_step_4_hapke_reflectance(dem_obj, mu0, mu, sun_az_deg, sun_el_deg, camera_pos, output_dir='.'):
    """Step 4: Compute Hapke reflectance WITHOUT shadows."""
    print("\n" + "="*80)
    print("STEP 4: Hapke Reflectance Computation (NO SHADOWS)")
    print("="*80)
    
    device = dem_obj.device
    sun_vec = Camera.unit_vec_from_az_el(sun_az_deg, sun_el_deg, device=device)
    sx, sy, sz = sun_vec[0], sun_vec[1], sun_vec[2]
    
    # Compute view vectors
    view_vectors = camera_pos - dem_obj.world_points
    view_vectors = view_vectors / torch.norm(view_vectors, dim=1, keepdim=True)
    
    # Compute phase angle
    cos_g = sx*view_vectors[:,0] + sy*view_vectors[:,1] + sz*view_vectors[:,2]
    cos_g = torch.clamp(cos_g, -1, 1)
    g_rad = torch.acos(cos_g).reshape(dem_obj.dem.shape)
    
    print(f"Phase angle (degrees):")
    g_deg = torch.rad2deg(g_rad)
    print(f"  Min: {g_deg.min().item():.2f}°, Max: {g_deg.max().item():.2f}°")
    print(f"  Mean: {g_deg.mean().item():.2f}°")
    
    # Compute Hapke reflectance
    hapke = HapkeModel(w=0.6, B0=0.4, h=0.1, phase_fun="hg", xi=0.1)
    R = hapke.radiance_factor(mu0, mu, g_rad)
    
    print(f"\nHapke reflectance (R) without shadows:")
    print(f"  Min: {R.min().item():.6f}, Max: {R.max().item():.6f}")
    print(f"  Mean: {R.mean().item():.6f}")
    print(f"  Std: {R.std().item():.6f}")
    print(f"  Non-zero values: {(R > 0).sum().item()} / {R.numel()}")
    print(f"  Contains NaN: {torch.isnan(R).any().item()}")
    print(f"  Contains Inf: {torch.isinf(R).any().item()}")
    
    # Plot reflectance
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    plot_tensor(g_rad, 'Phase Angle (radians)', axes[0, 0], cmap='viridis')
    plot_tensor(R, 'Hapke Reflectance (no shadows)', axes[0, 1], cmap='gray', vmin=0)
    
    # Log scale for better visualization
    R_safe = torch.clamp(R, min=1e-10)
    plot_tensor(torch.log10(R_safe), 'Hapke Reflectance (log scale)', 
                axes[1, 0], cmap='gray')
    
    # Histogram
    R_np = R.cpu().numpy().flatten()
    R_np_nonzero = R_np[R_np > 0]
    axes[1, 1].hist(R_np_nonzero, bins=100, edgecolor='black')
    axes[1, 1].set_title(f'Reflectance Distribution (non-zero values)\nCount: {len(R_np_nonzero)}')
    axes[1, 1].set_xlabel('Reflectance')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'step_4_hapke_reflectance.png'), dpi=150)
    print("  ✓ Saved: step_4_hapke_reflectance.png")
    plt.close()
    
    return R


def test_step_5_shadow_map_computation(dem_obj, sun_az_deg, sun_el_deg, output_dir='.'):
    """Step 5: Compute shadow map using cummax method."""
    print("\n" + "="*80)
    print("STEP 5: Shadow Map Computation (CUMMAX METHOD)")
    print("="*80)
    print(f"Sun: azimuth={sun_az_deg}°, elevation={sun_el_deg}°")
    
    device = dem_obj.device
    shadow_map = Renderer.compute_shadow_map(dem_obj.dem, sun_az_deg, sun_el_deg, 
                                             cellsize=dem_obj.cellsize, device=device)
    
    print(f"\nShadow map statistics:")
    print(f"  Shape: {shadow_map.shape}")
    print(f"  Min: {shadow_map.min().item():.4f}, Max: {shadow_map.max().item():.4f}")
    print(f"  Mean: {shadow_map.mean().item():.4f}")
    print(f"  Lit pixels (value=1): {(shadow_map == 1).sum().item()} / {shadow_map.numel()}")
    print(f"  Shadow pixels (value=0): {(shadow_map == 0).sum().item()} / {shadow_map.numel()}")
    print(f"  Intermediate values: {((shadow_map > 0) & (shadow_map < 1)).sum().item()}")
    print(f"  Contains NaN: {torch.isnan(shadow_map).any().item()}")
    print(f"  Contains Inf: {torch.isinf(shadow_map).any().item()}")
    print(f"  Unique values: {torch.unique(shadow_map).cpu().numpy()}")
    
    # Plot shadow map
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    plot_tensor(shadow_map, 'Shadow Map\n1=lit, 0=shadow', axes[0, 0], 
                cmap='gray', vmin=0, vmax=1)
    plot_tensor(dem_obj.dem, 'DEM for Reference', axes[0, 1], cmap='terrain')
    
    # Overlay shadow on DEM
    dem_np = dem_obj.dem.cpu().numpy()
    shadow_np = shadow_map.cpu().numpy()
    overlay = axes[1, 0].imshow(dem_np, cmap='terrain', alpha=0.7, origin='lower')
    axes[1, 0].imshow(shadow_np, cmap='Reds', alpha=0.3, origin='lower')
    axes[1, 0].set_title('Shadow Overlay on DEM\n(Red = shadow)')
    plt.colorbar(overlay, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    # Histogram
    shadow_np_flat = shadow_np.flatten()
    axes[1, 1].hist(shadow_np_flat, bins=50, edgecolor='black')
    axes[1, 1].set_title('Shadow Map Value Distribution')
    axes[1, 1].set_xlabel('Shadow Value')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'step_5_shadow_map.png'), dpi=150)
    print("  ✓ Saved: step_5_shadow_map.png")
    plt.close()
    
    return shadow_map


def test_step_6_shadow_application(R, shadow_map, output_dir='.'):
    """Step 6: Apply shadow map to reflectance."""
    print("\n" + "="*80)
    print("STEP 6: Shadow Application to Reflectance")
    print("="*80)
    
    R_shadowed = R * shadow_map
    
    print(f"\nReflectance statistics:")
    print(f"  Before shadows: min={R.min().item():.6f}, max={R.max().item():.6f}, mean={R.mean().item():.6f}")
    print(f"  After shadows: min={R_shadowed.min().item():.6f}, max={R_shadowed.max().item():.6f}, mean={R_shadowed.mean().item():.6f}")
    print(f"  Non-zero before: {(R > 0).sum().item()}")
    print(f"  Non-zero after: {(R_shadowed > 0).sum().item()}")
    print(f"  Pixels removed by shadow: {((R > 0) & (R_shadowed == 0)).sum().item()}")
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    plot_tensor(R, 'Reflectance BEFORE shadows', axes[0, 0], cmap='gray', vmin=0)
    plot_tensor(R_shadowed, 'Reflectance AFTER shadows', axes[0, 1], cmap='gray', vmin=0)
    plot_tensor(shadow_map, 'Shadow Map Applied', axes[1, 0], cmap='gray', vmin=0, vmax=1)
    
    # Difference map
    diff = R - R_shadowed
    plot_tensor(diff, 'Reflectance Removed by Shadows', axes[1, 1], cmap='hot', vmin=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'step_6_shadow_application.png'), dpi=150)
    print("  ✓ Saved: step_6_shadow_application.png")
    plt.close()
    
    return R_shadowed


def test_step_7_camera_rendering(dem_obj, R_shadowed, camera_pos, 
                                image_w=128, image_h=128, focal_length=800, output_dir='.'):
    """Step 7: Render camera image."""
    print("\n" + "="*80)
    print("STEP 7: Camera Image Rendering")
    print("="*80)
    print(f"Image size: {image_w}x{image_h}, focal length: {focal_length}")
    
    device = dem_obj.device
    
    # Create camera
    camera = Camera(image_width=image_w, image_height=image_h, 
                   focal_length=focal_length, device=device)
    
    # Compute target (DEM center)
    center_x = dem_obj.x0 + (dem_obj.width * dem_obj.cellsize) / 2
    center_y = dem_obj.y0 + (dem_obj.height * dem_obj.cellsize) / 2
    center_z = torch.mean(dem_obj.dem)
    target = torch.tensor([center_x, center_y, center_z], dtype=torch.float32, device=device)
    
    # Look at target
    up = camera.up
    Rot = camera.look_at(camera_pos, target, up)
    
    # Transform to camera coordinates
    world_points = dem_obj.world_points
    Xc = camera.world_to_camera(world_points, Rot, camera_pos)
    
    print(f"\nCamera coordinates:")
    print(f"  X range: [{Xc[:,0].min().item():.2f}, {Xc[:,0].max().item():.2f}]")
    print(f"  Y range: [{Xc[:,1].min().item():.2f}, {Xc[:,1].max().item():.2f}]")
    print(f"  Z range: [{Xc[:,2].min().item():.2f}, {Xc[:,2].max().item():.2f}]")
    print(f"  Points in front of camera (Z>0): {(Xc[:,2] > 0).sum().item()} / {len(Xc)}")
    
    # Project to image
    fx, fy, cx, cy = camera.get_intrinsics()
    u = (fx * Xc[:,0] / Xc[:,2]) + cx
    v = (fy * Xc[:,1] / Xc[:,2]) + cy
    
    valid_mask = (Xc[:,2] > 0) & (u >= 0) & (u < image_w) & (v >= 0) & (v < image_h)
    print(f"\nProjection statistics:")
    print(f"  Valid projected points: {valid_mask.sum().item()} / {len(Xc)}")
    
    # Render with Z-buffer
    refl_flat = R_shadowed.flatten()
    
    image = torch.zeros((image_h, image_w), dtype=torch.float32, device=device)
    zbuffer = torch.full((image_h, image_w), float('inf'), dtype=torch.float32, device=device)
    
    valid_indices = torch.where(valid_mask)[0]
    print(f"  Rendering {len(valid_indices)} points...")
    
    for idx in valid_indices:
        ui = int(u[idx].item())
        vi = int(v[idx].item())
        z_val = Xc[idx, 2].item()
        
        if 0 <= ui < image_w and 0 <= vi < image_h:
            if z_val < zbuffer[vi, ui]:
                zbuffer[vi, ui] = z_val
                image[vi, ui] = refl_flat[idx]
    
    print(f"\nRendered image statistics:")
    print(f"  Min: {image.min().item():.6f}, Max: {image.max().item():.6f}")
    print(f"  Mean: {image.mean().item():.6f}")
    print(f"  Non-zero pixels: {(image > 0).sum().item()} / {image.numel()}")
    print(f"  Contains NaN: {torch.isnan(image).any().item()}")
    print(f"  Contains Inf: {torch.isinf(image).any().item()}")
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    plot_tensor(image, 'Rendered Camera Image', axes[0, 0], cmap='gray', vmin=0)
    plot_tensor(zbuffer, 'Z-Buffer (depth)', axes[0, 1], cmap='viridis')
    
    # Log scale
    image_safe = torch.clamp(image, min=1e-10)
    plot_tensor(torch.log10(image_safe), 'Rendered Image (log scale)', 
                axes[1, 0], cmap='gray')
    
    # Histogram
    image_np = image.cpu().numpy().flatten()
    image_nonzero = image_np[image_np > 0]
    axes[1, 1].hist(image_nonzero, bins=50, edgecolor='black')
    axes[1, 1].set_title(f'Image Intensity Distribution\nNon-zero pixels: {len(image_nonzero)}')
    axes[1, 1].set_xlabel('Intensity')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'step_7_camera_rendering.png'), dpi=150)
    print("  ✓ Saved: step_7_camera_rendering.png")
    plt.close()
    
    return image


def test_complete_pipeline():
    """Run complete pipeline test."""
    print("\n" + "="*80)
    print("COMPLETE IMAGE GENERATION PIPELINE TEST")
    print("="*80)
    
    # Create output directory for test results
    output_dir = '/work/FrederikWürtzSørensen#7865/master/tests/test_results'
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nSaving results to: {output_dir}")
    
    # Load config
    config_path = '/work/FrederikWürtzSørensen#7865/master/configs/defaults.ini'
    config = load_config_file(config_path)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # Pipeline parameters
    sun_az_deg = 0
    sun_el_deg = 25
    cam_az_deg = 0
    cam_el_deg = 55
    cam_dist = 3000
    image_w = config['IMAGE_W']
    image_h = config['IMAGE_H']
    focal_length = config['FOCAL_LENGTH']
    
    # Step 1: Generate DEM
    dem_tensor = test_step_1_dem_generation(config, device, output_dir)
    
    # Step 2: Create DEM object
    dem_obj = test_step_2_dem_object_creation(dem_tensor, device, output_dir)
    
    # Step 3: Compute lighting geometry
    mu0, mu, camera_pos = test_step_3_lighting_geometry(
        dem_obj, sun_az_deg, sun_el_deg, cam_az_deg, cam_el_deg, cam_dist, output_dir)
    
    # Step 4: Compute Hapke reflectance (no shadows)
    R = test_step_4_hapke_reflectance(dem_obj, mu0, mu, sun_az_deg, sun_el_deg, camera_pos, output_dir)
    
    # Step 5: Compute shadow map
    shadow_map = test_step_5_shadow_map_computation(dem_obj, sun_az_deg, sun_el_deg, output_dir)
    
    # Step 6: Apply shadows to reflectance
    R_shadowed = test_step_6_shadow_application(R, shadow_map, output_dir)
    
    # Step 7: Render camera image
    image = test_step_7_camera_rendering(dem_obj, R_shadowed, camera_pos,
                                        image_w, image_h, focal_length, output_dir)
    
    print("\n" + "="*80)
    print("PIPELINE TEST COMPLETE")
    print("="*80)
    print(f"\nAll plots saved to {output_dir}")
    print("\nCheck the following files:")
    print("  - step_1_dem_generation.png")
    print("  - step_2_normals.png")
    print("  - step_3_lighting_geometry.png")
    print("  - step_4_hapke_reflectance.png")
    print("  - step_5_shadow_map.png")
    print("  - step_6_shadow_application.png")
    print("  - step_7_camera_rendering.png")
    
    return image


if __name__ == "__main__":
    test_complete_pipeline()
