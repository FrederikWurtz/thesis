"""Loss and reflectance utilities split out from trainer_core."""

import torch
import torch.nn.functional as F
from master.render.dem_utils import DEM
from master.render.camera import Camera
from master.render.renderer import Renderer
from master.render.hapke_model import HapkeModel, FullHapkeModel

def _render_shading_batched(dem_tensor_flat, meta_flat, camera, hapke_model_or_params, device, debug=False):
    """
    Render shading for multiple DEMs and viewing angles in a batch.
    
    Args:
        dem_tensor_flat: Flattened DEM tensor [N, H, W] where N = B*5
        meta_flat: Flattened metadata [N, 5]
        camera: Shared camera across all renders
        hapke_model_or_params: Either HapkeModel instance OR dict with {'w': tensor, 'theta_bar': tensor}
        device: torch device
        
    Returns:
        reflectance_maps_flat: [N, H, W] reflectance maps
    """
    N = dem_tensor_flat.shape[0]
    H, W = dem_tensor_flat.shape[1:]
    reflectance_flat = torch.zeros(N, H, W, dtype=dem_tensor_flat.dtype, device=device)
    
    # Check if we have spatial parameters
    is_spatial = isinstance(hapke_model_or_params, dict)
    
    for i in range(N):
        dem_curr = dem_tensor_flat[i]
        dem_obj = DEM(dem_curr, cellsize=1, x0=0, y0=0)
        
        # Create model for this specific render
        if is_spatial:
            w_map = hapke_model_or_params['w'][i]  # [H, W]
            theta_map = hapke_model_or_params['theta_bar'][i]  # [H, W]
            #hapke_model = FullHapkeModel(w=w_map, theta_bar_rad=theta_map, debug=debug)
            hapke_model = FullHapkeModel(w=w_map, b=theta_map, debug=debug)
        else:
            hapke_model = hapke_model_or_params  # Reuse shared model
        
        renderer = Renderer(dem_obj, hapke_model, camera)
        
        sun_az = float(meta_flat[i, 0].item())
        sun_el = float(meta_flat[i, 1].item())
        cam_az = float(meta_flat[i, 2].item())
        cam_el = float(meta_flat[i, 3].item())
        cam_dist = float(meta_flat[i, 4].item())
        
        renderer.render_shading(
            sun_az_deg=sun_az,
            sun_el_deg=sun_el,
            camera_az_deg=cam_az,
            camera_el_deg=cam_el,
            camera_distance_from_center=cam_dist
        )
        
        refl_map = renderer.reflectance_map
        if refl_map.dim() > 2:
            refl_map = refl_map.squeeze()
        reflectance_flat[i] = refl_map
    
    return reflectance_flat


def compute_reflectance_map_from_dem(dem_tensor, meta, device, camera_params, hapke_params, debug=False):
    """
    Compute reflectance map from a DEM tensor using physics-based rendering.
    GPU-OPTIMIZED: Batched processing to maximize parallelization and GPU utilization.
    
    Key optimizations:
    1. Camera and HapkeModel created ONCE and reused across all batch items/images
    2. All tensors stay on GPU during processing (no CPU conversions)
    3. Metadata extracted in batch to minimize Python-tensor boundary crossings
    4. Pre-allocated output tensor to avoid append/stack overhead
    5. Ready for future vmap vectorization (with Renderer refactoring)
    
    Gradients are ENABLED for backpropagation through the reflectance computation.
    
    Args:
        dem_tensor: DEM tensor [B, 1, H, W] - predicted or target DEM
        meta: Metadata tensor [B, 5, 5] - (sun_az, sun_el, cam_az, cam_el, cam_dist) for each of 5 images
        device: torch device
        camera_params: Dict with 'image_width', 'image_height', 'focal_length'
        hapke_params: Dict with Hapke model parameters
        
    Returns:
        reflectance_maps: [B, 5, H, W] - reflectance maps for each of the 5 viewing conditions
    """
    B, _, H, W = dem_tensor.shape
    
    # Create shared Camera and HapkeModel (reused across all rendering calls)
    camera = Camera(
        image_width=camera_params['image_width'],
        image_height=camera_params['image_height'],
        focal_length=camera_params['focal_length'],
        device=device
    )
    hapke_model = HapkeModel(
        w=hapke_params['w'], 
        B0=hapke_params['B0'],
        h=hapke_params['h'], 
        phase_fun=hapke_params['phase_fun'],
        xi=hapke_params['xi']
    )
    
    # Pre-allocate output on GPU
    reflectance_maps = torch.zeros(B, 5, H, W, dtype=dem_tensor.dtype, device=device)
    
    # Reshape for batch processing: [B, 1, H, W] → [B*5, H, W] with repeated meta
    # This groups all work items (B*5 DEM+viewpoint pairs) together
    dem_flat = dem_tensor[:, 0]  # [B, H, W]
    dem_expanded = dem_flat.repeat_interleave(5, dim=0)  # [B*5, H, W]
    meta_expanded = meta.reshape(B * 5, 5)  # [B*5, 5]
    
    # Render all B*5 (dem, viewpoint) pairs in batch using shared Camera/HapkeModel
    reflectance_flat = _render_shading_batched(dem_expanded, meta_expanded, camera, hapke_model, device, debug=debug)  # [B*5, H, W]
    
    # Reshape back to [B, 5, H, W]
    reflectance_maps = reflectance_flat.reshape(B, 5, H, W)
    
    return reflectance_maps


def compute_reflectance_map_from_dem_multi_band(dem_tensor, meta, device, camera_params, w_band, theta_band, debug=False):
    """
    Compute reflectance map with spatially-varying Hapke parameters.
    """
    B, _, H, W = dem_tensor.shape
    
    camera = Camera(
        image_width=camera_params['image_width'],
        image_height=camera_params['image_height'],
        focal_length=camera_params['focal_length'],
        device=device
    )
    
    # Reshape DEMs and meta
    dem_flat = dem_tensor[:, 0]  # [B, H, W]
    dem_expanded = dem_flat.repeat_interleave(5, dim=0)  # [B*5, H, W]
    meta_expanded = meta.reshape(B * 5, 5)  # [B*5, 5]
    
    # Expand w and theta bands to match
    w_flat = w_band[:, 0].repeat_interleave(5, dim=0)  # [B*5, H, W]
    theta_flat = theta_band[:, 0].repeat_interleave(5, dim=0)  # [B*5, H, W]
    
    # Pass spatial parameters as dict
    hapke_params = {'w': w_flat, 'theta_bar': theta_flat}
    
    reflectance_flat = _render_shading_batched(dem_expanded, meta_expanded, camera, hapke_params, device, debug=debug)
    
    reflectance_maps = reflectance_flat.reshape(B, 5, H, W)
    return reflectance_maps


def compute_gradients(tensor: torch.Tensor, spacing=(1.0, 1.0)) -> torch.Tensor:
    """
    Compute per-pixel gradient magnitude using torch.gradient.

    Accepts input shaped either [B, 1, H, W] or [B, H, W].
    Returns the gradient magnitude with the same batch/channel layout
    (i.e., [B, 1, H, W] for [B,1,H,W] input).
    Args:
        tensor: input DEM tensor
        spacing: tuple (dy, dx) specifying grid spacing (default: 1.0, 1.0)
    """
    # Normalize shapes to [B, H, W]
    if tensor.ndim == 4 and tensor.shape[1] == 1:
        vals = tensor[:, 0]
        keep_channel = True
    elif tensor.ndim == 3:
        vals = tensor
        keep_channel = False
    else:
        raise ValueError(
            f"compute_gradients: expected shape [B,1,H,W] or [B,H,W], got {tuple(tensor.shape)}"
        )

    # Compute derivatives along (rows, cols) -> (dy, dx)
    dy, dx = torch.gradient(vals, spacing=spacing, dim=(-2, -1))

    # Magnitude with small epsilon for numerical stability
    grad_mag = torch.sqrt(dx * dx + dy * dy + 1e-8)

    # Restore channel dimension if input had one
    if keep_channel:
        grad_mag = grad_mag.unsqueeze(1)

    return grad_mag

def compute_spatial_gradients(tensor: torch.Tensor, spacing=(1.0, 1.0)) -> torch.Tensor:
    """
    Compute per-pixel gradient magnitude using torch.gradient.

    Accepts input shaped either [B, 1, H, W] or [B, H, W].
    Returns the gradient magnitude with the same batch/channel layout
    (i.e., [B, 1, H, W] for [B,1,H,W] input).
    Args:
        tensor: input DEM tensor
        spacing: tuple (dy, dx) specifying grid spacing (default: 1.0, 1.0)
    """
    # Normalize shapes to [B, H, W]
    if tensor.ndim == 4 and tensor.shape[1] == 1:
        vals = tensor[:, 0]
        keep_channel = True
    elif tensor.ndim == 3:
        vals = tensor
        keep_channel = False
    else:
        raise ValueError(
            f"compute_spatial_gradients: expected shape [B,1,H,W] or [B,H,W], got {tuple(tensor.shape)}"
        )

    # Compute derivatives along (rows, cols) -> (dy, dx)
    dy, dx = torch.gradient(vals, spacing=spacing, dim=(-2, -1))

    # Restore channel dimension if input had one
    if keep_channel:
        dx = dx.unsqueeze(1)
        dy = dy.unsqueeze(1)

    return dx, dy

def calculate_total_loss(outputs, targets, target_reflectance_maps, meta, hapke_params=None, device=None,
                        camera_params=None, w_mse=1.0, w_grad=1.0, w_refl=1.0, height_norm=None, return_components=False, debug=False):
    """
    Calculate total loss combining MSE, gradient loss, and reflectance map loss.
    
    Args:
        outputs: Predicted DEMs [B, 1, H, W]
        targets: Target DEMs [B, 1, H, W]
        target_reflectance_maps: Ground truth reflectance maps [B, 5, H, W]
        meta: Metadata tensor [B, 5, 5]
        hapke_model: Hapke reflectance model instance
        device: torch device
        camera_params: Dict with 'image_width', 'image_height', 'focal_length' from metadata
        w_mse: Weight for MSE term (default: 1.0)
        w_grad: Weight for gradient term (default: 1.0)
        w_refl: Weight for reflectance map term (default: 1.0)
        
    Returns:
        total_loss: Weighted sum of all loss components
    """
    print("Using Single band")
    outputs_norm = outputs / height_norm if height_norm is not None else outputs
    targets_norm = targets / height_norm if height_norm is not None else targets

    # 1. MSE Loss - basic elevation accuracy
    loss_mse = F.mse_loss(outputs_norm, targets_norm, reduction='mean') # MSE loss, returns mean by default, but made explicit here
    
    # 2. Gradient Loss - captures steep slopes and terrain features
    out_grad_mag = compute_gradients(outputs_norm)
    tgt_grad_mag = compute_gradients(targets_norm)
    loss_grad = F.mse_loss(out_grad_mag, tgt_grad_mag, reduction='mean') # Gradient loss, returns mean by default
    
    # 3. Reflectance Map Loss - physics-based constraint
    # Compute reflectance maps from predicted DEM (WITH GRADIENTS for backprop)
    predicted_reflectance_maps = compute_reflectance_map_from_dem(outputs, meta, device, camera_params, hapke_params, debug=debug)
    
    loss_refl = F.mse_loss(predicted_reflectance_maps, target_reflectance_maps, reduction='mean')
    
    # Combine losses
    total_loss = w_mse * loss_mse + w_grad * loss_grad + w_refl * loss_refl
    
    if return_components:
        return loss_mse, loss_grad, loss_refl, total_loss
    else:
        return total_loss
def calculate_total_loss_multi_band(dem_outputs, dem_targets, reflectance_map_targets, metas, w_outputs, w_targets, theta_outputs, theta_targets, device=None,
                        config=None, return_components=False, debug=False):
    
    """
    Calculate total loss combining MSE, gradient loss, and reflectance map loss.
    
    Args:
        dem_outputs: Predicted DEMs [B, 1, H, W]
        dem_targets: Target DEMs [B, 1, H, W]
        reflectance_map_targets: Ground truth reflectance maps [B, 5, H, W]
        metas: Metadata tensor [B, 5, 5]
        w_outputs: Albedo band tensors [B, 1, H, W]
        w_targets: Albedo band tensors [B, 1, H, W]
        theta_outputs: Phase function band tensors [B, 1, H, W]
        theta_targets: Phase function band tensors [B, 1, H, W]
        hapke_params: Hapke reflectance model parameters
        device: torch device
        camera_params: Dict with 'image_width', 'image_height', 'focal_length' from metadata
        w_mse: Weight for MSE term (default: 1.0)
        w_grad: Weight for gradient term (default: 1.0)
        w_refl: Weight for reflectance map term (default: 1.0)
        w_w: Weight for albedo consistency term (default: 1.0)
        w_theta: Weight for phase function consistency term (default: 1.0)
    Returns:
        total_loss: Weighted sum of all loss components
    """
    if debug:
        # Safety checks before normalization
        print(f"🔍 DEM outputs - min: {dem_outputs.min():.6f}, max: {dem_outputs.max():.6f}, mean: {dem_outputs.mean():.6f}, has_nan: {torch.isnan(dem_outputs).any()}, has_inf: {torch.isinf(dem_outputs).any()}")
        print(f"🔍 DEM targets - min: {dem_targets.min():.6f}, max: {dem_targets.max():.6f}, mean: {dem_targets.mean():.6f}, has_nan: {torch.isnan(dem_targets).any()}, has_inf: {torch.isinf(dem_targets).any()}")
        print(f"🔍 W outputs - min: {w_outputs.min():.6f}, max: {w_outputs.max():.6f}, mean: {w_outputs.mean():.6f}, has_nan: {torch.isnan(w_outputs).any()}, has_inf: {torch.isinf(w_outputs).any()}")
        print(f"🔍 W targets - min: {w_targets.min():.6f}, max: {w_targets.max():.6f}, mean: {w_targets.mean():.6f}, has_nan: {torch.isnan(w_targets).any()}, has_inf: {torch.isinf(w_targets).any()}")
        print(f"🔍 Theta outputs - min: {theta_outputs.min():.6f}, max: {theta_outputs.max():.6f}, mean: {theta_outputs.mean():.6f}, has_nan: {torch.isnan(theta_outputs).any()}, has_inf: {torch.isinf(theta_outputs).any()}")
        print(f"🔍 Theta targets - min: {theta_targets.min():.6f}, max: {theta_targets.max():.6f}, mean: {theta_targets.mean():.6f}, has_nan: {torch.isnan(theta_targets).any()}, has_inf: {torch.isinf(theta_targets).any()}")
        
        # Check normalization parameters
        print(f"🔍 Normalization params - HEIGHT_NORM: {config['HEIGHT_NORMALIZATION']}, HEIGHT_NORM_PM: {config['HEIGHT_NORMALIZATION_PM']}")
        print(f"🔍 W range: [{config['W_MIN']}, {config['W_MAX']}], Theta range: [{config['THETA_BAR_MIN']}, {config['THETA_BAR_MAX']}]")
    
    # Normalize DEMs
    dem_outputs_norm = dem_outputs #/ config["HEIGHT_NORMALIZATION"] + config["HEIGHT_NORMALIZATION_PM"]
    dem_targets_norm = dem_targets #/ config["HEIGHT_NORMALIZATION"] + config["HEIGHT_NORMALIZATION_PM"]
    
    if debug:
        print(f"🔍 After DEM norm - outputs has_nan: {torch.isnan(dem_outputs_norm).any()}, targets has_nan: {torch.isnan(dem_targets_norm).any()}")

    # Normalize w and theta to [0, 1] range
    w_range = config["W_MAX"] - config["W_MIN"]
    theta_range = config["THETA_BAR_MAX"] - config["THETA_BAR_MIN"]

    if debug:
        print(f"🔍 W range value: {w_range}, Theta range value: {theta_range}")
    
    w_outputs_norm = (w_outputs - config["W_MIN"]) / (config["W_MAX"] - config["W_MIN"] + 1e-8)
    w_targets_norm = (w_targets - config["W_MIN"]) / (config["W_MAX"] - config["W_MIN"] + 1e-8)
    theta_outputs_norm = (theta_outputs - config["THETA_BAR_MIN"]) / (config["THETA_BAR_MAX"] - config["THETA_BAR_MIN"] + 1e-8)
    theta_targets_norm = (theta_targets - config["THETA_BAR_MIN"]) / (config["THETA_BAR_MAX"] - config["THETA_BAR_MIN"] + 1e-8)

    if debug:
        print(f"🔍 After W norm - outputs: min={w_outputs_norm.min():.6f}, max={w_outputs_norm.max():.6f}, has_nan: {torch.isnan(w_outputs_norm).any()}")
        print(f"🔍 After Theta norm - outputs: min={theta_outputs_norm.min():.6f}, max={theta_outputs_norm.max():.6f}, has_nan: {torch.isnan(theta_outputs_norm).any()}")


    # 1. MSE Loss - basic elevation accuracy
    # loss_mse = F.mse_loss(dem_outputs_norm, dem_targets_norm, reduction='mean')
    loss_mse = F.l1_loss(dem_outputs_norm, dem_targets_norm, reduction='mean')
    if debug:
        print(f"🔍 loss_mse: {loss_mse.item():.6f}, has_nan: {torch.isnan(loss_mse).any()}")
    
    # 2. Gradient Loss - captures steep slopes and terrain features
    # out_grad_mag = compute_gradients(dem_outputs_norm)
    # tgt_grad_mag = compute_gradients(dem_targets_norm)
    # normalize by looking at tgt gradient magnitude - not output, to avoid collapse
    # grad_scale = torch.linalg.vector_norm(
    #                                     tgt_grad_mag,
    #                                     ord=2,
    #                                     dim=(1, 2, 3),
    #                                     keepdim=True,
    #                                 ).clamp(min=1e-8)    
    # out_grad_mag_norm = out_grad_mag # / grad_scale
    # tgt_grad_mag_norm = tgt_grad_mag #/ grad_scale
    
    # if debug:
    #     print(f"🔍 Gradients - out_grad has_nan: {torch.isnan(out_grad_mag_norm).any()}, tgt_grad has_nan: {torch.isnan(tgt_grad_mag_norm).any()}")
    
    # if debug:
    #     print(f"🔍 loss_grad: {loss_grad.item():.6f}, has_nan: {torch.isnan(loss_grad).any()}")
    # loss_grad = F.l1_loss(out_grad_mag_norm, tgt_grad_mag_norm, reduction='mean')
    
    
    dx_out, dy_out = compute_spatial_gradients(dem_outputs_norm)
    dx_tgt, dy_tgt = compute_spatial_gradients(dem_targets_norm)
    loss_grad = F.l1_loss(dx_out, dx_tgt, reduction='mean') + F.l1_loss(dy_out, dy_tgt, reduction='mean')
    
    
    # # Try without reflectance to see how much time it takes
    # # 3. Reflectance Map Loss - physics-based constraint
    predicted_reflectance_maps = compute_reflectance_map_from_dem_multi_band(dem_outputs,
                                                                             metas, 
                                                                             device, 
                                                                             config["CAMERA_PARAMS"], 
                                                                             w_band=w_outputs, 
                                                                             theta_band=theta_outputs, 
                                                                             debug=debug
                                                                             )
 #   predicted_reflectance_maps = compute_reflectance_map_from_dem_multi_band(dem_outputs.detach(), # stop reflectance gradients affecting the DEM 
 #                                                                            metas, 
 #                                                                            device, 
 #                                                                            config["CAMERA_PARAMS"], 
 #                                                                            w_band=w_outputs, 
 #                                                                            theta_band=theta_outputs, 
 #                                                                            debug=debug
 #                                                                            )
    
    
    eps = 1e-6
    scale = reflectance_map_targets.abs().mean(dim=(1,2,3), keepdim=True).clamp(min=eps)

    # subsample to speed up reflectance loss calculation - this is a bit hacky but it works and is much faster than using the full resolution maps for loss
    # consider only calculating refl every 4 steps
    # if step % 4 == 0:
    #     loss_refl = ...
    # else:
    #     loss_refl = 0


    pred_norm = predicted_reflectance_maps / scale
    tgt_norm  = reflectance_map_targets / scale
    
    loss_refl = F.l1_loss(pred_norm, tgt_norm, reduction="mean")

    # # 4. w_band Loss - albedo consistency
    # loss_w = F.mse_loss(w_outputs_norm, w_targets_norm, reduction='mean')
    # if debug:
    #     print(f"🔍 loss_w: {loss_w.item():.6f}, has_nan: {torch.isnan(loss_w).any()}")
    # # 5. theta_band Loss - phase function consistency
    # loss_theta = F.mse_loss(theta_outputs_norm, theta_targets_norm, reduction='mean')
    # if debug:
    #     print(f"🔍 loss_theta: {loss_theta.item():.6f}, has_nan: {torch.isnan(loss_theta).any()}")
    
    loss_w = torch.tensor(0.0, device=device)
    loss_theta = torch.tensor(0.0, device=device)
    
    # # Only residual reflectance that geometry cannot explain should be explained by w and theta - this is a bit hacky but it encourages the model to use w and theta to explain things that DEM cannot, rather than using them as a crutch to explain everything and not learning good geometry
    # refl_geom = compute_reflectance(
    #     dem_outputs.detach(),
    #     metas,
    #     w_band=w_outputs.detach(),     # or mean(w), see note below
    #     theta_band=None                # or fixed theta
    # )

    # # Full reflectance
    # refl_full = compute_reflectance(
    #     dem_outputs.detach(),
    #     metas,
    #     w_band=w_outputs.detach(),
    #     theta_band=theta_outputs
    # )

    # # Residual that geometry cannot explain
    # refl_residual = reflectance_targets - refl_geom

    # # Theta only explains residual
    # loss_theta_refl = L1(
    #     normalize(refl_full - refl_geom),
    #     normalize(refl_residual)
    # )
        
    # Combine losses
    total_loss = config["W_MSE"] * loss_mse + config["W_GRAD"] * loss_grad + config["W_REFL"] * loss_refl + config["W_W"] * loss_w + config["W_THETA"] * loss_theta 
    if debug:
        print(f"🔍 Weights - W_MSE: {config['W_MSE']}, W_GRAD: {config['W_GRAD']}, W_REFL: {config['W_REFL']}, W_W: {config['W_W']}, W_THETA: {config['W_THETA']}")
        print(f"🔍 total_loss: {total_loss.item():.6f}, has_nan: {torch.isnan(total_loss).any()}")

    
    if return_components:
        returned_components = [loss_mse, loss_grad, loss_refl, loss_w, loss_theta, total_loss]
        return returned_components
    else:
        return total_loss






# def compute_reflectance_map_from_dem(dem_tensor, meta, device, camera_params, hapke_params):
#     B, _, H, W = dem_tensor.shape
#     reflectance_maps = []
#     for b in range(B):
#         dem_np = dem_tensor[b, 0].detach().cpu().numpy()
#         dem_obj = DEM(dem_np, cellsize=1, x0=0, y0=0)
#         camera = Camera(
#             image_width=camera_params['image_width'],
#             image_height=camera_params['image_height'],
#             focal_length=camera_params['focal_length']
#         )
#         hapke_model = HapkeModel(w=hapke_params['w'], B0=hapke_params['B0'],
#                                  h=hapke_params['h'], phase_fun=hapke_params['phase_fun'],
#                                  xi=hapke_params['xi'])
#         renderer = Renderer(dem_obj, hapke_model, camera)
#         batch_reflectance_maps = []
#         for img_idx in range(meta.shape[1]):
#             sun_az = meta[b, img_idx, 0].item()
#             sun_el = meta[b, img_idx, 1].item()
#             cam_az = meta[b, img_idx, 2].item()
#             cam_el = meta[b, img_idx, 3].item()
#             cam_dist = meta[b, img_idx, 4].item()
#             renderer.render_shading(
#                 sun_az_deg=sun_az,
#                 sun_el_deg=sun_el,
#                 camera_az_deg=cam_az,
#                 camera_el_deg=cam_el,
#                 camera_distance_from_center=cam_dist,
#                 model="hapke"
#             )
#             refl_map = renderer.reflectance_map
#             batch_reflectance_maps.append(refl_map)
#         batch_reflectance_maps = torch.stack(batch_reflectance_maps, dim=0)
#         reflectance_maps.append(batch_reflectance_maps)
#     reflectance_maps = torch.stack(reflectance_maps, dim=0).to(device)
#     reflectance_maps = torch.nan_to_num(reflectance_maps, nan=0.0, posinf=1.0, neginf=0.0)
#     return reflectance_maps


# def calculate_total_loss(outputs, targets, target_reflectance_maps, meta, hapke_params=None, device=None,
#                         camera_params=None, w_mse=None, w_grad=None, w_refl=None):
#     outputs = torch.nan_to_num(outputs, nan=0.0, posinf=1000.0, neginf=-1000.0)  # Begrens DEM-verdier til et realistisk område
#     loss_mse = F.mse_loss(outputs, targets)

#     out_grad_mag = _compute_gradients(outputs)
#     tgt_grad_mag = _compute_gradients(targets)
#     loss_grad = F.mse_loss(out_grad_mag, tgt_grad_mag)

#     predicted_reflectance_maps = compute_reflectance_map_from_dem(outputs, meta, device, camera_params, hapke_params)
#     loss_refl = F.mse_loss(predicted_reflectance_maps, target_reflectance_maps)

#     total_loss = w_mse * loss_mse + w_grad * loss_grad + w_refl * loss_refl
#     total_loss.loss_components = {'mse': loss_mse.item(), 'grad': loss_grad.item(), 'refl': loss_refl.item()}

#     if torch.isnan(loss_mse) or torch.isnan(loss_grad) or torch.isnan(loss_refl):
#         print(f"NaN detected! mse={loss_mse.item()}, grad={loss_grad.item()}, refl={loss_refl.item()}")

#     return total_loss



__all__ = ['compute_reflectance_map_from_dem', 'calculate_total_loss']
