"""
Debug the compute_shadow_map function with instrumentation.
"""

import torch
import math
import torch.nn.functional as F


def compute_shadow_map_debug(dem_array, sun_az_deg, sun_el_deg, device=None, cellsize=1):
    """
    Instrumented version of compute_shadow_map for debugging.
    """
    
    AlignCorners = True

    if torch.is_tensor(dem_array):
        dem_t = dem_array
        if device is None:
            device = dem_array.device
    else:
        if device is None:
            device = torch.device('cpu')
        dem_t = torch.from_numpy(np.array(dem_array)).to(dtype=torch.float32, device=device)

    H, W = dem_t.shape
    print(f"Input DEM shape: ({H}, {W})")
    print(f"Sun: az={sun_az_deg}°, el={sun_el_deg}°")

    # pad to larger square (diagonal) to avoid clipping on rotation
    diag = int(math.ceil(math.hypot(H, W)))
    pad_h = max(0, (diag - H) // 2 + 2)
    pad_w = max(0, (diag - W) // 2 + 2)
    print(f"Padding: pad_h={pad_h}, pad_w={pad_w}")

    dem_b = dem_t.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    print(f"DEM batch shape before pad: {dem_b.shape}")
    
    if pad_h > 0 or pad_w > 0:
        dem_b = F.pad(dem_b, (pad_w, pad_w, pad_h, pad_h), mode='constant', value=0.0)
    
    print(f"DEM batch shape after pad: {dem_b.shape}")

    cellsize_up = cellsize

    H_p, W_p = dem_b.shape[-2], dem_b.shape[-1]

    # Rotate so sun az points along image rows
    az = float(sun_az_deg)
    theta = math.radians(-az)
    cos_t = math.cos(theta); sin_t = math.sin(theta)
    theta_mat = torch.tensor([[[cos_t, -sin_t, 0.0], [sin_t, cos_t, 0.0]]], dtype=torch.float32, device=device)

    grid = F.affine_grid(theta_mat, dem_b.size(), align_corners=AlignCorners)
    dem_rot = F.grid_sample(dem_b, grid, mode='bilinear', padding_mode='zeros', align_corners=AlignCorners)
    
    print(f"DEM rotated shape: {dem_rot.shape}")
    print(f"DEM rotated stats: min={dem_rot.min().item():.4f}, max={dem_rot.max().item():.4f}, mean={dem_rot.mean().item():.4f}")

    # Mask of valid (non-padded) pixels after rotation
    ones = torch.ones_like(dem_b)
    mask_rot = F.grid_sample(ones, grid, mode='nearest', padding_mode='zeros', align_corners=AlignCorners)

    # Transformed field T = elevation - (row_index * cellsize_up * tan(el))
    el_rad = math.radians(sun_el_deg)
    tan_el = math.tan(el_rad)
    print(f"tan(elevation) = {tan_el:.6f}")
    
    rows = torch.arange(H_p, dtype=torch.float32, device=device).view(1, 1, H_p, 1) * cellsize_up
    T = dem_rot - rows * tan_el  # shape (1,1,H_p,W_p)
    
    print(f"T field shape: {T.shape}")
    print(f"T field stats BEFORE masking: min={T.min().item():.4f}, max={T.max().item():.4f}, mean={T.mean().item():.4f}")
    
    # Check if T varies along columns
    T_column_std = T.std(dim=3)  # std along width for each row
    print(f"T std along columns (should be non-zero): min={T_column_std.min().item():.6f}, max={T_column_std.max().item():.6f}")
    
    # Mark padding as -inf so it never occludes
    neg_inf = -float('inf')
    T = torch.where(mask_rot == 0, torch.tensor(neg_inf, device=device, dtype=T.dtype), T)
    
    print(f"T field stats AFTER masking: min={T.min().item():.4f}, max={T.max().item():.4f}")

    # GPU-OPTIMIZED backward row-wise scan using cummax
    min_val = torch.finfo(T.dtype).min / 2  # Safe minimum value
    T_safe = torch.where(torch.isinf(T), torch.tensor(min_val, device=device, dtype=T.dtype), T)
    
    print(f"T_safe stats: min={T_safe.min().item():.4f}, max={T_safe.max().item():.4f}")
    print(f"T_safe shape: {T_safe.shape}")
    
    # Print a small sample of T_safe to see variation
    if H_p >= 5 and W_p >= 5:
        print(f"T_safe sample (first 5x5):")
        print(T_safe[0, 0, :5, :5].cpu().numpy())
    
    T_flipped = torch.flip(T_safe, dims=[2])  # Flip rows for reverse cummax
    print(f"T_flipped shape: {T_flipped.shape}")
    
    # Compute cumulative maximum along rows (reversed direction)
    max_above_each_row = torch.cummax(T_flipped, dim=2)[0]
    print(f"max_above_each_row shape: {max_above_each_row.shape}")
    print(f"max_above_each_row stats: min={max_above_each_row.min().item():.4f}, max={max_above_each_row.max().item():.4f}")
    
    # Check if cummax result varies
    cummax_column_std = max_above_each_row.std(dim=3)
    print(f"cummax std along columns: min={cummax_column_std.min().item():.6f}, max={cummax_column_std.max().item():.6f}")
    
    if H_p >= 5 and W_p >= 5:
        print(f"max_above_each_row sample (first 5x5, still flipped):")
        print(max_above_each_row[0, 0, :5, :5].cpu().numpy())
    
    max_above_each_row = torch.flip(max_above_each_row, dims=[2])  # Flip back
    
    if H_p >= 5 and W_p >= 5:
        print(f"max_above_each_row sample after flip back (first 5x5):")
        print(max_above_each_row[0, 0, :5, :5].cpu().numpy())

    # Shift by one to exclude current row
    max_above = torch.cat([
        torch.full((1, 1, 1, W_p), min_val, device=device, dtype=T.dtype),
        max_above_each_row[:, :, :-1, :]
    ], dim=2)
    
    if H_p >= 5 and W_p >= 5:
        print(f"max_above sample (first 5x5, with shift):")
        print(max_above[0, 0, :5, :5].cpu().numpy())

    lit_rot = (T_safe >= max_above).bool()
    
    lit_fraction = lit_rot.float().mean().item()
    print(f"Lit fraction after cummax: {lit_fraction*100:.2f}%")

    # Safety: Ensure no NaN/Inf in shadow map
    lit_rot_f = lit_rot.float()
    lit_rot_f = torch.nan_to_num(lit_rot_f, nan=0.0, posinf=1.0, neginf=0.0)
    
    # Rotate back
    theta_inv = math.radians(az)
    cos_i = math.cos(theta_inv); sin_i = math.sin(theta_inv)
    theta_mat_inv = torch.tensor([[[cos_i, -sin_i, 0.0], [sin_i, cos_i, 0.0]]], dtype=torch.float32, device=device)
    grid_inv = F.affine_grid(theta_mat_inv, dem_b.size(), align_corners=AlignCorners)

    shadow_b = F.grid_sample(lit_rot_f, grid_inv, mode='nearest', padding_mode='zeros', align_corners=AlignCorners)
    mask_back = F.grid_sample(mask_rot.float(), grid_inv, mode='nearest', padding_mode='zeros', align_corners=AlignCorners)

    # crop center region back to original DEM size
    start_h = pad_h
    end_h = start_h + H
    start_w = pad_w
    end_w = start_w + W
    shadow_cropped = shadow_b[:, :, start_h:end_h, start_w:end_w]
    mask_cropped = mask_back[:, :, start_h:end_h, start_w:end_w]

    # remove padding-origin pixels
    shadow_cropped = shadow_cropped * (mask_cropped > 0.5).float()

    shadow_map_t = shadow_cropped.squeeze(0).squeeze(0)
    
    print(f"Final shadow map shape: {shadow_map_t.shape}")
    lit_final = (shadow_map_t == 1).sum().item() / shadow_map_t.numel()
    print(f"Final lit fraction: {lit_final*100:.2f}%")

    return shadow_map_t


if __name__ == "__main__":
    import numpy as np
    
    print("="*80)
    print("DEBUG FLAT TERRAIN")
    print("="*80)
    
    # Flat terrain
    dem = torch.zeros((100, 100), dtype=torch.float32)
    shadow_map = compute_shadow_map_debug(dem, sun_az_deg=0, sun_el_deg=25, device='cpu', cellsize=1.0)
    
    print("\n" + "="*80)
    print("Result: Flat terrain should be 100% lit, got {:.2f}%".format((shadow_map == 1).sum().item() / shadow_map.numel() * 100))
    print("="*80)
