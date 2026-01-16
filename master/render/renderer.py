from master.render.dem_utils import DEM
from master.render.hapke_model import HapkeModel
from master.render.camera import Camera
import numpy as np
import torch
import torch.nn.functional as F
import math

class Renderer:
    def __init__(self, dem: DEM, model: HapkeModel, camera: Camera):
        """
        Initialize the Renderer with a DEM (Digital Elevation Model) and a reflectance model.
        
        Args:
            dem (DEM): The digital elevation model containing terrain and normal vectors.
            model (HapkeModel): The reflectance model (e.g., Hapke or Lambertian).
        """
        self.dem = dem
        self.device = dem.device
        self.model = model
        self.rendered_shading = None
        self.reflectance_map = None  # Store reflectance map for gradient tracking
        self.camera = camera
        self.sun_az_deg = None
        self.sun_el_deg = None
        self.camera_distance_from_center = None
        self.camera_az_deg = None
        self.camera_el_deg = None
        self._last_camera_pos = None
        self.target = None

    def render_shading(self, sun_az_deg, sun_el_deg, camera_az_deg, camera_el_deg, camera_distance_from_center, model="hapke"):
        """
        Render surface shading with shadows using specified reflectance model.
        
        Args:
            sun_az_deg: Sun azimuth angle in degrees
            sun_el_deg: Sun elevation angle in degrees
            camera_az_deg: Camera azimuth angle in degrees
            camera_el_deg: Camera elevation angle in degrees
            camera_distance_from_center: Distance from DEM center to camera
            model: Reflectance model ("hapke" or "lambertian")
        """
        # Compute sun direction vector
        sun_vec = Camera.unit_vec_from_az_el(sun_az_deg, sun_el_deg, device=self.device)
        
        # Compute camera position
        center_x = self.dem.x0 + (self.dem.width * self.dem.cellsize) / 2
        center_y = self.dem.y0 + (self.dem.height * self.dem.cellsize) / 2
        
        cam_az_rad = torch.deg2rad(torch.tensor(camera_az_deg, dtype=torch.float32, device=self.device))
        cam_el_rad = torch.deg2rad(torch.tensor(camera_el_deg, dtype=torch.float32, device=self.device))
        
        cx = center_x + camera_distance_from_center * torch.sin(cam_az_rad) * torch.cos(cam_el_rad)
        cy = center_y + camera_distance_from_center * torch.cos(cam_az_rad) * torch.cos(cam_el_rad)
        cz = camera_distance_from_center * torch.sin(cam_el_rad)
        camera_pos = torch.stack([cx, cy, cz])
        
        # Cache camera position and parameters
        self._last_camera_pos = camera_pos
        self.sun_az_deg = sun_az_deg
        self.sun_el_deg = sun_el_deg
        self.camera_az_deg = camera_az_deg
        self.camera_el_deg = camera_el_deg
        self.camera_distance_from_center = camera_distance_from_center
        
        # Compute view vectors (normalized)
        view_vectors = camera_pos - self.dem.world_points
        view_vectors = view_vectors / torch.norm(view_vectors, dim=1, keepdim=True)
        
        # Flatten normals
        normals_flat = torch.stack([self.dem.nx.flatten(), self.dem.ny.flatten(), self.dem.nz.flatten()], dim=1)
        
        # Compute incidence and emission angles
        mu = (normals_flat * view_vectors).sum(dim=1).reshape(self.dem.dem.shape)
        mu0 = (normals_flat * sun_vec).sum(dim=1).reshape(self.dem.dem.shape)
        
        # Lambertian model (early return)
        if model.lower() == "lambertian":
            R = self.model.w / torch.pi * mu0
            self.rendered_shading = torch.where((mu0 > 0) & (mu > 0), R, torch.zeros_like(R))
            self.reflectance_map = self.rendered_shading
            return
        
        if model.lower() != "hapke":
            raise ValueError("Unknown model. Use 'hapke' or 'lambertian'.")
        
        # Hapke model: compute phase angle
        cos_g = (view_vectors * sun_vec).sum(dim=1)
        cos_g = torch.clamp(cos_g, -1, 1)
        g_rad = torch.acos(cos_g).reshape(self.dem.dem.shape)
        
        # Compute Hapke reflectance
        R = self.model.radiance_factor(mu0, mu, g_rad)
        
        # Compute and apply shadow map
        shadow_map = Renderer.compute_shadow_map(
            self.dem.dem, sun_az_deg, sun_el_deg, 
            cellsize=self.dem.cellsize, device=self.device
        )
        
        # Apply shadows (convert bool to float if needed)
        R_shadowed = R * shadow_map.type_as(R)
        
        # Store results, so that render_camera_image can use them
        self.rendered_shading = R_shadowed
        self.reflectance_map = R_shadowed


    def render_camera_image(self, 
                        sun_az_deg=None, 
                        sun_el_deg=None, 
                        camera_az_deg=None, 
                        camera_el_deg=None, 
                        camera_distance_from_center=None,
                        img_height=None,
                        img_width=None,
                        verbose=False):
        """
        Render a camera image from the DEM and reflectance data using pinhole camera model.
        
        Returns:
            tuple: (rendered_image, reflectance_map)
        """
        # Compute shading with shadows
        self.render_shading(sun_az_deg, sun_el_deg, camera_az_deg, camera_el_deg, camera_distance_from_center)
        
        # Setup camera geometry
        camera_pos = self._last_camera_pos
        center_x = self.dem.x0 + (self.dem.width * self.dem.cellsize) / 2
        center_y = self.dem.y0 + (self.dem.height * self.dem.cellsize) / 2
        center_z = torch.mean(self.dem.dem)
        target = torch.tensor([center_x, center_y, center_z], dtype=torch.float32, device=self.device)
        
        # Transform world points to camera space
        fx, fy, cx, cy = self.camera.get_intrinsics()
        Rot = self.camera.look_at(camera_pos, target, self.camera.up)
        Xc = self.camera.world_to_camera(self.dem.world_points, Rot, camera_pos)
        
        # Filter points in front of camera
        in_front = Xc[:, 2] > 0
        if not in_front.any():
            return torch.zeros((img_height, img_width), dtype=torch.float32, device=self.device), self.reflectance_map
        
        Xc = Xc[in_front]
        refl = self.rendered_shading.flatten()[in_front]
        depth = Xc[:, 2]
        
        # Project to image plane
        uv = self.camera.project_to_image(Xc, fx, fy, cx, cy)
        u = torch.round(uv[:, 0]).long()
        v = img_height - 1 - torch.round(uv[:, 1]).long()  # Flip for origin='lower'
        
        # Filter to image bounds
        valid = (u >= 0) & (u < img_width) & (v >= 0) & (v < img_height)
        if not valid.any():
            return torch.zeros((img_height, img_width), dtype=torch.float32, device=self.device), self.reflectance_map
        
        u, v, refl, depth = u[valid], v[valid], refl[valid], depth[valid]
        
        # Z-buffer using scatter_reduce with 'amin' (closest depth wins)
        pixel_idx = v * img_width + u
        
        # Create image using scatter_reduce to automatically handle Z-buffering
        # Method: For each pixel, keep the reflectance value with minimum depth
        img_flat = torch.full((img_height * img_width,), float('inf'), dtype=torch.float32, device=self.device)
        depth_map = torch.full((img_height * img_width,), float('inf'), dtype=torch.float32, device=self.device)
        
        # First pass: get minimum depth per pixel
        depth_map.scatter_reduce_(0, pixel_idx, depth, reduce='amin', include_self=False)
        
        # Second pass: only keep reflectance where depth matches minimum
        closest = (depth == depth_map[pixel_idx])
        img_flat = torch.zeros((img_height * img_width,), dtype=torch.float32, device=self.device)
        img_flat.scatter_(0, pixel_idx[closest], refl[closest])
        
        # Reshape and flip to match origin='lower'
        img = img_flat.reshape(img_height, img_width)
        img = torch.flipud(torch.fliplr(img))
        
        if verbose:
            non_zero = torch.count_nonzero(img)
            print(f"  Rendered: {non_zero}/{img.numel()} pixels ({100*non_zero/img.numel():.1f}%)")
            print(f"  Range: [{img.min():.6f}, {img.max():.6f}]")
        
        return img, self.reflectance_map

    @staticmethod
    def compute_shadow_map_batched(dem_array, sun_params_list, device=None, cellsize=1):
        """
        Compute shadow maps for multiple sun angles in a single GPU pass.
        This is more efficient than calling compute_shadow_map multiple times.
        
        Args:
            dem_array: 2D DEM tensor [H, W]
            sun_params_list: List of (sun_az_deg, sun_el_deg) tuples
            device: torch device
            cellsize: DEM cell size
            
        Returns:
            shadow_maps: List of shadow maps, one per sun angle
        """
        shadow_maps = []
        for sun_az_deg, sun_el_deg in sun_params_list:
            shadow_map = Renderer.compute_shadow_map(dem_array, sun_az_deg, sun_el_deg, device, cellsize)
            shadow_maps.append(shadow_map)
        return shadow_maps

    @staticmethod
    def compute_shadow_map(dem_array, sun_az_deg, sun_el_deg, device=None, cellsize=1):
        """
        Optimized torch-based scan-line shadow map (sun at infinity).
        Returns lit map (1=lit, 0=shadow) as boolean tensor.
        
        Algorithm:
        1. Rotate DEM to align sun azimuth with image rows
        2. Compute transformed elevation field T = elevation - row * cellsize * tan(sun_el)
        3. Use cummax to find horizon from bottom->top (backward scan)
        4. Rotate result back to original orientation
        
        Args:
            dem_array: 2D numpy array or torch tensor (elevation map)
            sun_az_deg: Sun azimuth in degrees
            sun_el_deg: Sun elevation in degrees
            device: torch device (defaults to input tensor device or CPU)
            cellsize: DEM cell size in meters
        
        Returns:
            Boolean tensor where True=lit, False=shadow
        """
        # Convert input to torch tensor and handle device placement
        if torch.is_tensor(dem_array):
            dem_t = dem_array.to(dtype=torch.float32)
            device = device or dem_array.device
        else:
            device = device or torch.device('cpu')
            dem_t = torch.as_tensor(dem_array, dtype=torch.float32, device=device)

        H, W = dem_t.shape

        # Pad to diagonal size to prevent rotation clipping
        diag = int(math.ceil(math.hypot(H, W))) + 4  # +4 for margin
        pad = (diag - H) // 2, (diag - W) // 2
        
        dem_b = dem_t.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H, W)
        dem_b = F.pad(dem_b, (pad[1], pad[1], pad[0], pad[0]), mode='constant', value=0.0)
        H_p, W_p = dem_b.shape[-2:]

        # Precompute rotation angles (rotate to align sun azimuth with rows)
        az_rad = math.radians(-sun_az_deg)
        cos_az, sin_az = math.cos(az_rad), math.sin(az_rad)
        
        # Forward rotation: align sun direction with image rows
        theta_mat = torch.tensor([[[cos_az, -sin_az, 0.0], 
                                    [sin_az, cos_az, 0.0]]], 
                                dtype=torch.float32, device=device)
        grid = F.affine_grid(theta_mat, dem_b.size(), align_corners=True)
        
        # Rotate DEM and create valid pixel mask
        dem_rot = F.grid_sample(dem_b, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        mask_rot = F.grid_sample(torch.ones_like(dem_b), grid, mode='nearest', 
                                padding_mode='zeros', align_corners=True)

        # Compute transformed elevation field T
        # T = elevation - row_distance * tan(sun_elevation)
        # This transforms the shadow problem into finding the cumulative max
        el_rad = math.radians(sun_el_deg)
        rows = torch.arange(H_p, dtype=torch.float32, device=device).view(1, 1, H_p, 1)
        T = dem_rot - rows * (cellsize * math.tan(el_rad))
        
        # Mark invalid (padded) regions as -inf so they never cast shadows
        T = torch.where(mask_rot > 0, T, torch.full_like(T, -float('inf')))

        # Vectorized backward scan (bottom->top) using cummax
        # Flip rows, compute cumulative maximum, flip back
        T_flipped = torch.flip(T, dims=[2])
        cummax_flipped = torch.cummax(T_flipped, dim=2)[0]
        prev_max = torch.flip(cummax_flipped, dims=[2])
        
        # Compare each pixel with max from rows below (shifted)
        # A pixel is lit if its T value >= maximum T from all rows below
        lit_rot = torch.zeros_like(T, dtype=torch.bool)
        lit_rot[:, :, :-1, :] = T[:, :, :-1, :] >= prev_max[:, :, 1:, :]
        lit_rot[:, :, -1:, :] = True  # Bottom row always lit (no obstruction below)

        # Inverse rotation: restore original orientation
        theta_mat_inv = torch.tensor([[[cos_az, sin_az, 0.0], 
                                        [-sin_az, cos_az, 0.0]]], 
                                    dtype=torch.float32, device=device)
        grid_inv = F.affine_grid(theta_mat_inv, dem_b.size(), align_corners=True)
        
        # Rotate shadow map and mask back to original orientation
        shadow_b = F.grid_sample(lit_rot.float(), grid_inv, mode='nearest', 
                                padding_mode='zeros', align_corners=True)
        mask_back = F.grid_sample(mask_rot, grid_inv, mode='nearest', 
                                padding_mode='zeros', align_corners=True)

        # Crop to original DEM size and apply mask
        shadow_cropped = shadow_b[:, :, pad[0]:pad[0]+H, pad[1]:pad[1]+W]
        mask_cropped = mask_back[:, :, pad[0]:pad[0]+H, pad[1]:pad[1]+W]
        
        # Combine shadow map with valid pixel mask
        shadow_map = (shadow_cropped * (mask_cropped > 0.5)).squeeze(0).squeeze(0)
        
        return shadow_map.bool()
