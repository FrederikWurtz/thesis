"""Loss and reflectance utilities split out from trainer_core."""

import torch
import torch.nn.functional as F
from master.render.dem_utils import DEM
from master.render.camera import Camera
from master.render.renderer import Renderer
from master.render.hapke_model import HapkeModel

def compute_reflectance_map_from_dem(dem_tensor, meta, device, camera_params, hapke_params):
    """
    Compute reflectance map from a DEM tensor using physics-based rendering.
    Gradients are ENABLED for backpropagation through the reflectance computation.
    
    Args:
        dem_tensor: DEM tensor [B, 1, H, W] - predicted or target DEM
        meta: Metadata tensor [B, 5, 5] - (sun_az, sun_el, cam_az, cam_el, cam_dist) for each of 5 images
        hapke_model: Hapke reflectance model instance
        device: torch device
        camera_params: Dict with 'image_width', 'image_height', 'focal_length' from metadata
        
    Returns:
        reflectance_maps: [B, 5, H, W] - reflectance maps for each of the 5 viewing conditions
    """
    B, _, H, W = dem_tensor.shape
    reflectance_maps = []
    
    for b in range(B):
        dem_np = dem_tensor[b, 0].detach().cpu().numpy()  # Extract DEM for this batch item
        dem_obj = DEM(dem_np, cellsize=1, x0=0, y0=0)
        
        # Create camera with parameters from metadata
        camera = Camera(
            image_width=camera_params['image_width'],
            image_height=camera_params['image_height'],
            focal_length=camera_params['focal_length'],
            device=device
        )
        hapke_model = HapkeModel(w=hapke_params['w'], B0=hapke_params['B0'],
                                 h=hapke_params['h'], phase_fun=hapke_params['phase_fun'],
                                 xi=hapke_params['xi'])
        renderer = Renderer(dem_obj, hapke_model, camera)
        
        batch_reflectance_maps = []
        for img_idx in range(5):  # 5 images per sample
            sun_az = meta[b, img_idx, 0].item()
            sun_el = meta[b, img_idx, 1].item()
            cam_az = meta[b, img_idx, 2].item()
            cam_el = meta[b, img_idx, 3].item()
            cam_dist = meta[b, img_idx, 4].item()
            
            # Render shading (computes reflectance map with gradient tracking)
            # NOTE: This enables gradients through the physics model
            renderer.render_shading(
                sun_az_deg=sun_az,
                sun_el_deg=sun_el,
                camera_az_deg=cam_az,
                camera_el_deg=cam_el,
                camera_distance_from_center=cam_dist,
                model="hapke"
            )
            
            # Get reflectance map (this is a torch tensor with gradients enabled)
            refl_map = renderer.reflectance_map  # [H, W]
            batch_reflectance_maps.append(refl_map)
        
        # Stack reflectance maps for this batch item: [5, H, W]
        batch_reflectance_maps = torch.stack(batch_reflectance_maps, dim=0)
        reflectance_maps.append(batch_reflectance_maps)
    
    # Stack all batches: [B, 5, H, W]
    reflectance_maps = torch.stack(reflectance_maps, dim=0).to(device)
    return reflectance_maps

# def calculate_total_loss(outputs, targets, target_reflectance_maps, meta, hapke_params=None, device=None,
#                         camera_params=None, w_mse=None, w_grad=None, w_refl=None):
def calculate_total_loss(outputs, targets, target_reflectance_maps, meta, hapke_params=None, device=None,
                        camera_params=None, w_mse=1.0, w_grad=1.0, w_refl=1.0):
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
    # 1. MSE Loss - basic elevation accuracy
    loss_mse = F.mse_loss(outputs, targets)
    
    # 2. Gradient Loss - captures steep slopes and terrain features
    def compute_gradients(tensor):
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=tensor.dtype, device=tensor.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                               dtype=tensor.dtype, device=tensor.device).view(1, 1, 3, 3)
        
        grad_x = F.conv2d(tensor, sobel_x, padding=1)
        grad_y = F.conv2d(tensor, sobel_y, padding=1)
        grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
        return grad_magnitude
    
    out_grad_mag = compute_gradients(outputs)
    tgt_grad_mag = compute_gradients(targets)
    loss_grad = F.mse_loss(out_grad_mag, tgt_grad_mag)
    
    # 3. Reflectance Map Loss - physics-based constraint
    # Compute reflectance maps from predicted DEM (WITH GRADIENTS for backprop)
    predicted_reflectance_maps = compute_reflectance_map_from_dem(outputs, meta, device, camera_params, hapke_params)
    loss_refl = F.mse_loss(predicted_reflectance_maps, target_reflectance_maps)
    
    # Combine losses
    total_loss = w_mse * loss_mse + w_grad * loss_grad + w_refl * loss_refl
    
    # Store loss components for debugging
    total_loss.loss_components = {
        'mse': loss_mse.item(),
        'grad': loss_grad.item(),
        'refl': loss_refl.item()
    }
    
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
