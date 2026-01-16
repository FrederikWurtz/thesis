"""
Detailed diagnostic test for the cummax shadow computation.

This test creates simple synthetic cases to verify the shadow algorithm logic.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import math
import torch.nn.functional as F

# Add master to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from master.render.renderer import Renderer


def test_simple_slope_shadow(output_dir='.'):
    """Test shadow on a simple slope - should be easy to verify."""
    print("\n" + "="*80)
    print("TEST 1: Simple Slope (should have obvious shadow)")
    print("="*80)
    
    # Create a simple slope: height increases from left to right
    size = 100
    dem = torch.zeros((size, size), dtype=torch.float32)
    
    # Create a slope: height increases linearly across rows
    for i in range(size):
        dem[i, :] = torch.linspace(0, 50, size)  # 50m high at right edge
    
    print(f"DEM shape: {dem.shape}")
    print(f"DEM height range: {dem.min().item():.2f} to {dem.max().item():.2f}")
    
    # Sun from the left (azimuth=270°, elevation=25°)
    # This should cast shadows on the right side of the slope
    sun_az = 270
    sun_el = 25
    
    shadow_map = Renderer.compute_shadow_map(dem, sun_az, sun_el, cellsize=1.0, device='cpu')
    
    lit_fraction = (shadow_map == 1).sum().item() / shadow_map.numel()
    print(f"\nShadow map:")
    print(f"  Lit pixels: {(shadow_map == 1).sum().item()} / {shadow_map.numel()}")
    print(f"  Lit fraction: {lit_fraction*100:.2f}%")
    print(f"  Unique values: {torch.unique(shadow_map).cpu().numpy()}")
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    axes[0].imshow(dem.cpu().numpy(), cmap='terrain', origin='lower')
    axes[0].set_title(f'DEM: Simple Slope\n(Height 0-50m)')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    plt.colorbar(axes[0].images[0], ax=axes[0])
    
    axes[1].imshow(shadow_map.cpu().numpy(), cmap='gray', vmin=0, vmax=1, origin='lower')
    axes[1].set_title(f'Shadow Map\nSun: az={sun_az}°, el={sun_el}°\nLit: {lit_fraction*100:.1f}%')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    
    # Cross-section
    mid_row = size // 2
    axes[2].plot(dem[mid_row, :].cpu().numpy(), label='DEM Height', linewidth=2)
    axes[2].plot(shadow_map[mid_row, :].cpu().numpy() * 50, label='Shadow (scaled)', linewidth=2, linestyle='--')
    axes[2].set_title(f'Cross-section at row {mid_row}')
    axes[2].set_xlabel('X position')
    axes[2].set_ylabel('Height / Shadow')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shadow_test_1_slope.png'), dpi=150)
    print("  ✓ Saved: shadow_test_1_slope.png")
    plt.close()
    
    return shadow_map


def test_single_peak_shadow(output_dir='.'):
    """Test shadow cast by a single peak."""
    print("\n" + "="*80)
    print("TEST 2: Single Peak (should cast directional shadow)")
    print("="*80)
    
    size = 200
    dem = torch.zeros((size, size), dtype=torch.float32)
    
    # Create a Gaussian peak in the center
    center = size // 2
    peak_height = 50
    sigma = size / 10
    
    for i in range(size):
        for j in range(size):
            r_sq = (i - center)**2 + (j - center)**2
            dem[i, j] = peak_height * torch.exp(torch.tensor(-r_sq / (2 * sigma**2)))
    
    print(f"DEM shape: {dem.shape}")
    print(f"DEM height range: {dem.min().item():.2f} to {dem.max().item():.2f}")
    print(f"Peak height: {dem.max().item():.2f}")
    
    # Test different sun angles
    sun_angles = [
        (0, 25, "North"),
        (90, 25, "East"),
        (180, 25, "South"),
        (270, 25, "West"),
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Plot DEM
    im0 = axes[0].imshow(dem.cpu().numpy(), cmap='terrain', origin='lower')
    axes[0].set_title(f'DEM: Single Peak\nHeight: {dem.max().item():.1f}m')
    plt.colorbar(im0, ax=axes[0])
    
    for idx, (sun_az, sun_el, direction) in enumerate(sun_angles):
        shadow_map = Renderer.compute_shadow_map(dem, sun_az, sun_el, cellsize=1.0, device='cpu')
        
        lit_fraction = (shadow_map == 1).sum().item() / shadow_map.numel()
        print(f"\n  Sun from {direction} (az={sun_az}°, el={sun_el}°):")
        print(f"    Lit pixels: {(shadow_map == 1).sum().item()} / {shadow_map.numel()}")
        print(f"    Lit fraction: {lit_fraction*100:.2f}%")
        
        ax = axes[idx + 1]
        ax.imshow(shadow_map.cpu().numpy(), cmap='gray', vmin=0, vmax=1, origin='lower')
        ax.set_title(f'Sun from {direction}\naz={sun_az}°, el={sun_el}°\nLit: {lit_fraction*100:.1f}%')
    
    # Hide last subplot
    axes[5].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shadow_test_2_peak.png'), dpi=150)
    print("  ✓ Saved: shadow_test_2_peak.png")
    plt.close()


def test_flat_terrain_no_shadow(output_dir='.'):
    """Test that flat terrain has no shadows."""
    print("\n" + "="*80)
    print("TEST 3: Flat Terrain (should have NO shadows)")
    print("="*80)
    
    size = 100
    dem = torch.zeros((size, size), dtype=torch.float32)
    
    print(f"DEM shape: {dem.shape}")
    print(f"DEM is completely flat (all zeros)")
    
    sun_az = 0
    sun_el = 25
    
    shadow_map = Renderer.compute_shadow_map(dem, sun_az, sun_el, cellsize=1.0, device='cpu')
    
    lit_fraction = (shadow_map == 1).sum().item() / shadow_map.numel()
    print(f"\nShadow map:")
    print(f"  Lit pixels: {(shadow_map == 1).sum().item()} / {shadow_map.numel()}")
    print(f"  Lit fraction: {lit_fraction*100:.2f}%")
    print(f"  EXPECTED: 100% lit (no shadows on flat terrain)")
    
    if lit_fraction < 0.99:
        print(f"  ❌ ERROR: Flat terrain should not have shadows!")
    else:
        print(f"  ✓ PASS: Flat terrain correctly has no shadows")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].imshow(dem.cpu().numpy(), cmap='terrain', origin='lower')
    axes[0].set_title('DEM: Flat Terrain')
    plt.colorbar(axes[0].images[0], ax=axes[0])
    
    axes[1].imshow(shadow_map.cpu().numpy(), cmap='gray', vmin=0, vmax=1, origin='lower')
    axes[1].set_title(f'Shadow Map\nLit: {lit_fraction*100:.1f}%')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shadow_test_3_flat.png'), dpi=150)
    print("  ✓ Saved: shadow_test_3_flat.png")
    plt.close()
    
    return lit_fraction


def test_cummax_internals(output_dir='.'):
    """Test the internal cummax logic step by step."""
    print("\n" + "="*80)
    print("TEST 4: Cummax Logic Internals")
    print("="*80)
    
    # Create a very simple slope
    size = 10
    dem = torch.zeros((size, size), dtype=torch.float32)
    
    # Simple diagonal slope
    for i in range(size):
        dem[i, :] = float(i * 5)  # Each row 5m higher
    
    print("Simple DEM (each row 5m higher than previous):")
    print(dem.cpu().numpy())
    
    # Sun parameters
    sun_az_deg = 0  # From north
    sun_el_deg = 25
    
    # Manually compute T field (what cummax operates on)
    device = dem.device
    cellsize = 1.0
    
    # This is the key transformation in the shadow algorithm
    el_rad = math.radians(sun_el_deg)
    tan_el = math.tan(el_rad)
    
    print(f"\nSun elevation: {sun_el_deg}°")
    print(f"tan(elevation): {tan_el:.4f}")
    
    # T = elevation - (row_index * cellsize * tan(el))
    rows = torch.arange(size, dtype=torch.float32, device=device).view(size, 1) * cellsize
    T = dem - rows * tan_el
    
    print(f"\nT field (elevation - row*tan(el)):")
    print(T.cpu().numpy())
    
    # Now run cummax (simplified version)
    T_safe = T
    T_flipped = torch.flip(T_safe, dims=[0])  # Flip rows for reverse cummax
    max_above_each_row = torch.cummax(T_flipped, dim=0)[0]
    max_above_each_row = torch.flip(max_above_each_row, dims=[0])  # Flip back
    
    print(f"\nCumulative max from above (max of all higher rows):")
    print(max_above_each_row.cpu().numpy())
    
    # Shift to get max above (excluding current row)
    min_val = torch.finfo(T.dtype).min / 2
    max_above = torch.cat([
        torch.full((1, size), min_val, device=device, dtype=T.dtype),
        max_above_each_row[:-1, :]
    ], dim=0)
    
    print(f"\nMax above (shifted):")
    print(max_above.cpu().numpy())
    
    # Lit where T >= max_above
    lit = (T >= max_above).float()
    
    print(f"\nLit map (T >= max_above):")
    print(lit.cpu().numpy())
    
    lit_fraction = lit.mean().item()
    print(f"\nLit fraction: {lit_fraction*100:.2f}%")
    
    # Now compare with actual shadow_map function
    shadow_map = Renderer.compute_shadow_map(dem, sun_az_deg, sun_el_deg, cellsize=cellsize, device=device)
    
    print(f"\nActual shadow_map from Renderer:")
    print(shadow_map.cpu().numpy())
    
    actual_lit_fraction = (shadow_map == 1).sum().item() / shadow_map.numel()
    print(f"Actual lit fraction: {actual_lit_fraction*100:.2f}%")
    
    # Plot comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    axes[0, 0].imshow(dem.cpu().numpy(), cmap='terrain', origin='lower')
    axes[0, 0].set_title('DEM')
    plt.colorbar(axes[0, 0].images[0], ax=axes[0, 0])
    
    axes[0, 1].imshow(T.cpu().numpy(), cmap='RdBu_r', origin='lower')
    axes[0, 1].set_title('T field')
    plt.colorbar(axes[0, 1].images[0], ax=axes[0, 1])
    
    axes[0, 2].imshow(max_above.cpu().numpy(), cmap='RdBu_r', origin='lower')
    axes[0, 2].set_title('Max above')
    plt.colorbar(axes[0, 2].images[0], ax=axes[0, 2])
    
    axes[1, 0].imshow(lit.cpu().numpy(), cmap='gray', vmin=0, vmax=1, origin='lower')
    axes[1, 0].set_title(f'Computed lit map\n{lit_fraction*100:.1f}% lit')
    
    axes[1, 1].imshow(shadow_map.cpu().numpy(), cmap='gray', vmin=0, vmax=1, origin='lower')
    axes[1, 1].set_title(f'Actual shadow_map\n{actual_lit_fraction*100:.1f}% lit')
    
    # Difference
    diff = lit - shadow_map
    axes[1, 2].imshow(diff.cpu().numpy(), cmap='RdBu_r', vmin=-1, vmax=1, origin='lower')
    axes[1, 2].set_title('Difference\n(Red = mismatch)')
    plt.colorbar(axes[1, 2].images[0], ax=axes[1, 2])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shadow_test_4_cummax_internals.png'), dpi=150)
    print("  ✓ Saved: shadow_test_4_cummax_internals.png")
    plt.close()


if __name__ == "__main__":
    print("="*80)
    print("DETAILED SHADOW CUMMAX DIAGNOSTIC TESTS")
    print("="*80)
    
    # Create output directory for test results
    output_dir = '/work/FrederikWürtzSørensen#7865/master/tests/test_results'
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nSaving results to: {output_dir}")
    
    # Test 3 first - flat terrain should have NO shadows (sanity check)
    lit_fraction_flat = test_flat_terrain_no_shadow(output_dir)
    
    # Other tests
    test_simple_slope_shadow(output_dir)
    test_single_peak_shadow(output_dir)
    test_cummax_internals(output_dir)
    
    print("\n" + "="*80)
    print("TESTS COMPLETE")
    print("="*80)
    print(f"\nAll diagnostic plots saved to {output_dir}")
    
    if lit_fraction_flat < 0.99:
        print("\n❌ CRITICAL: Flat terrain has shadows - cummax algorithm has a bug!")
    else:
        print("\n✓ Flat terrain test passed - algorithm may be working correctly")
