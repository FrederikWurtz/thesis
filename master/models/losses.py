"""Loss and reflectance utilities split out from trainer_core."""

import torch
import torch.nn.functional as F
from master.render.dem_utils import DEM
from master.render.camera import Camera
from master.render.renderer import Renderer
from master.render.hapke_model import HapkeModel

def _render_shading_batched(dem_tensor_flat, meta_flat, camera, hapke_model, device):
    """
    Render shading for multiple DEMs and viewing angles in a batch.
    This is a helper for parallel rendering across multiple work items.
    
    Args:
        dem_tensor_flat: Flattened DEM tensor [N, H, W] where N = B*5 or B*K
        meta_flat: Flattened metadata [N, 5]
        camera, hapke_model: Shared across all renders
        device: torch device
        
    Returns:
        reflectance_maps_flat: [N, H, W] reflectance maps
    """
    N = dem_tensor_flat.shape[0]
    H, W = dem_tensor_flat.shape[1:]
    reflectance_flat = torch.zeros(N, H, W, dtype=dem_tensor_flat.dtype, device=device)
    
    # Use torch.vmap concept - but implemented as loop since Renderer isn't fully vectorized
    # Future: this could be actual vmap if Renderer is refactored
    for i in range(N):
        dem_curr = dem_tensor_flat[i]
        dem_obj = DEM(dem_curr, cellsize=1, x0=0, y0=0)
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
            camera_distance_from_center=cam_dist,
            model="hapke"
        )
        reflectance_flat[i] = renderer.reflectance_map
    
    return reflectance_flat


def compute_reflectance_map_from_dem(dem_tensor, meta, device, camera_params, hapke_params):
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
    reflectance_flat = _render_shading_batched(dem_expanded, meta_expanded, camera, hapke_model, device)
    
    # Reshape back to [B, 5, H, W]
    reflectance_maps = reflectance_flat.reshape(B, 5, H, W)
    
    return reflectance_maps


def compute_gradients(tensor):
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                            dtype=tensor.dtype, device=tensor.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                            dtype=tensor.dtype, device=tensor.device).view(1, 1, 3, 3)
    
    grad_x = F.conv2d(tensor, sobel_x, padding=1)
    grad_y = F.conv2d(tensor, sobel_y, padding=1)
    grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
    return grad_magnitude

def calculate_total_loss(outputs, targets, target_reflectance_maps, meta, hapke_params=None, device=None,
                        camera_params=None, w_mse=1.0, w_grad=1.0, w_refl=1.0, height_norm=None, return_components=False):
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
    predicted_reflectance_maps = compute_reflectance_map_from_dem(outputs, meta, device, camera_params, hapke_params)
    
    loss_refl = F.mse_loss(predicted_reflectance_maps, target_reflectance_maps, reduction='mean')
    
    # Combine losses
    total_loss = w_mse * loss_mse + w_grad * loss_grad + w_refl * loss_refl
    
    if return_components:
        return loss_mse, loss_grad, loss_refl, total_loss
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

# def _compute_gradients(tensor):
#     sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
#                             dtype=tensor.dtype, device=tensor.device).view(1, 1, 3, 3)
#     sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
#                             dtype=tensor.dtype, device=tensor.device).view(1, 1, 3, 3)
#     grad_x = F.conv2d(tensor, sobel_x, padding=1)
#     grad_y = F.conv2d(tensor, sobel_y, padding=1)
#     grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
#     return grad_magnitude


__all__ = ['compute_reflectance_map_from_dem', 'calculate_total_loss']
