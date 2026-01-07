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
        Render the surface shading using the specified reflectance model.
        Now uses PyTorch for gradient tracking and stores full reflectance map.

        Args:
            sun_az_deg (float): Sun azimuth angle in degrees.
            sun_el_deg (float): Sun elevation angle in degrees.
            camera_az_deg (float): Camera azimuth angle in degrees.
            camera_el_deg (float): Camera elevation angle in degrees.
            camera_distance_from_center (float): Distance from the DEM center to the camera.
            model (str): Reflectance model to use ("hapke" or "lambertian").
        """
        # Compute sun direction unit vector from azimuth and elevation
        sun_vec = Camera.unit_vec_from_az_el(sun_az_deg, sun_el_deg)
        sx, sy, sz = sun_vec[0], sun_vec[1], sun_vec[2]

        # Calculate the center coordinates of the DEM in world space
        center_x = self.dem.x0 + (self.dem.width * self.dem.cellsize) / 2
        center_y = self.dem.y0 + (self.dem.height * self.dem.cellsize) / 2

        # Compute camera position in world coordinates based on azimuth, elevation, and distance
        camera_az_rad = torch.deg2rad(torch.tensor(camera_az_deg, dtype=torch.float32))
        camera_el_rad = torch.deg2rad(torch.tensor(camera_el_deg, dtype=torch.float32))
        cx = center_x + camera_distance_from_center * torch.sin(camera_az_rad) * torch.cos(camera_el_rad)
        cy = center_y + camera_distance_from_center * torch.cos(camera_az_rad) * torch.cos(camera_el_rad)
        cz = camera_distance_from_center * torch.sin(camera_el_rad)
        camera_pos = torch.stack([cx, cy, cz])

        self._last_camera_pos = camera_pos
        self.sun_az_deg = sun_az_deg
        self.sun_el_deg = sun_el_deg
        self.camera_distance_from_center = camera_distance_from_center
        self.camera_az_deg = camera_az_deg
        self.camera_el_deg = camera_el_deg

        # Calculate view vectors from each DEM point to the camera
        # Each vector points from the DEM point to the camera position
        view_vectors = camera_pos - self.dem.world_points
        # Normalize the view vectors to unit length
        view_vectors = view_vectors / torch.norm(view_vectors, dim=1, keepdim=True)

        # Flatten normal vectors for computation (1D arrays)
        nx_flat = self.dem.nx.flatten()
        ny_flat = self.dem.ny.flatten()
        nz_flat = self.dem.nz.flatten()

        # Compute cosines of emission (mu) and incidence (mu0) angles
        # mu: cosine of emission angle (between surface normal and view vector)
        mu = (nx_flat*view_vectors[:,0] + ny_flat*view_vectors[:,1] + nz_flat*view_vectors[:,2]).reshape(self.dem.dem.shape)
        # mu0: cosine of incidence angle (between surface normal and sun direction)
        mu0 = (nx_flat*sx + ny_flat*sy + nz_flat*sz).reshape(self.dem.dem.shape)

        if model.lower() == "lambertian":
            # Lambertian reflectance model: R = w/pi * mu0
            R = self.model.w / torch.pi * mu0
            # Only keep values where both mu0 and mu are positive (lit and visible)
            self.rendered_shading = torch.where((mu0 > 0) & (mu > 0), R, torch.tensor(0.0))
            return
        elif model.lower() != "hapke":
            # Raise error for unknown model
            raise ValueError("Unknown model. Use 'hapke' or 'lambertian'.")

        # Compute phase angle (g) between sun and view directions for each DEM point
        # cos_g: cosine of phase angle between sun direction and view vector
        cos_g = sx*view_vectors[:,0] + sy*view_vectors[:,1] + sz*view_vectors[:,2]
        cos_g = torch.clamp(cos_g, -1, 1)  # Ensure within valid range for arccos
        g_rad = torch.acos(cos_g).reshape(self.dem.dem.shape)

        # Hapke reflectance model: use the model's radiance_factor method
        R = self.model.radiance_factor(mu0, mu, g_rad)

        # Calculate shadow map (1=lit, 0=shadow) using the DEM and sun angles
        shadow_map = Renderer.compute_shadow_map(self.dem.dem, sun_az_deg, sun_el_deg, cellsize=self.dem.cellsize)

        # Move and cast shadow map to the same dtype/device as R for safe arithmetic
        if shadow_map.device != R.device or shadow_map.dtype != R.dtype:
            shadow_map = shadow_map.to(device=R.device, dtype=R.dtype)

        # Sanity-check shape before applying mask
        if shadow_map.shape != R.shape:
            raise ValueError(f"Shadow map shape {tuple(shadow_map.shape)} does not match reflectance shape {tuple(R.shape)}")

        # Apply the shadow mask to the reflectance: zeros where shadow==0, keep R where shadow==1
        # shadow_map currently contains 0/1 values (as R.dtype), so simple multiplication suffices
        R_masked = R * shadow_map
        self.rendered_shading = R_masked

        # Store the masked reflectance map so all downstream users see shading with shadows applied
        self.reflectance_map = R_masked


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

        This method implements a complete camera projection pipeline:
        1. Transform 3D world points to camera coordinate system
        2. Project camera coordinates to 2D image plane
        3. Apply Z-buffer rendering for depth testing

        Parameters:
        -----------
        camera_pos : ndarray, optional
            Camera position in world coordinates (3,). If None, uses self.camera.pos if available.
        target : ndarray, optional
            Target position to look at (3,). If None, uses self.camera.target if available.
        up : ndarray, optional
            Up vector (3,). If None, uses self.camera.up if available, else [0,0,1].
        intrinsics : tuple, optional
            Camera intrinsics (fx, fy, cx, cy). If None, uses self.camera.intrinsics if available.
        img_size : tuple, optional
            Image size (height, width) in pixels. If None, uses self.camera.img_size if available.
        sun_az_deg, sun_el_deg, camera_az_deg, camera_el_deg, camera_distance_from_center, model : optional
            If any are specified, will recalculate shading before rendering.
        verbose : bool, optional
            Whether to print detailed rendering information (default: True)

        Returns:
        --------
        tuple of (torch.Tensor, torch.Tensor)
            - Rendered camera image with radiance factor (I/F) values
            - Reflectance map with radiance factor values (full DEM resolution)

        Raises:
        -------
        ValueError
            If render_shading() hasn't been called first to generate reflectance data
        """

        self.render_shading(sun_az_deg, sun_el_deg, camera_az_deg, camera_el_deg, camera_distance_from_center)

        # camera position has now been set by render_shading if not provided
        camera_pos = self._last_camera_pos

        # set target to DEM center
        center_x = self.dem.x0 + (self.dem.width * self.dem.cellsize) / 2
        center_y = self.dem.y0 + (self.dem.height * self.dem.cellsize) / 2
        center_z = torch.mean(self.dem.dem)
        target = torch.tensor([center_x, center_y, center_z], dtype=torch.float32)
        self.target = target

        # Use defaults from camera
        fx, fy, cx, cy = self.camera.get_intrinsics()
        up = self.camera.up

        # Use camera object's look_at method
        Rot = self.camera.look_at(camera_pos, target, up)

        # Transform world points to camera coordinates using camera object
        world_points = self.dem.world_points
        if verbose:
            print(f"  Total world points: {len(world_points)}")
        Xc = self.camera.world_to_camera(world_points, Rot, camera_pos)

        if verbose:
            print(f"  Camera coordinates Z range: [{Xc[:,2].min().item():.1f}, {Xc[:,2].max().item():.1f}]")

        # Ensure reflectance and points have compatible shapes
        refl_flat = self.rendered_shading.flatten()
        if len(refl_flat) != len(world_points):
            raise ValueError(f"Reflectance size {len(refl_flat)} doesn't match points size {len(world_points)}")

        # Filter points behind camera (negative Z in camera coordinates)
        in_front = Xc[:,2] > 0
        if verbose:
            print(f"  Points in front of camera: {torch.sum(in_front).item()}/{len(world_points)} ({100*torch.sum(in_front).item()/len(world_points):.1f}%)")

        if torch.sum(in_front).item() == 0:
            if verbose:
                print("  WARNING: No points in front of camera!")
            # Return empty image and empty reflectance map
            return torch.zeros((img_height, img_width), dtype=torch.float32), self.reflectance_map

        # Keep only points in front of camera
        Xc = Xc[in_front]
        refl = refl_flat[in_front]

        # Project to image plane using pinhole camera model via camera object
        uv = self.camera.project_to_image(Xc, fx, fy, cx, cy)
        if verbose:
            print(f"  Projected coordinates range: U=[{uv[:,0].min().item():.1f}, {uv[:,0].max().item():.1f}], V=[{uv[:,1].min().item():.1f}, {uv[:,1].max().item():.1f}]")

        # Round to integer pixel coordinates and filter to image bounds
        u = torch.round(uv[:,0]).long()
        v = torch.round(uv[:,1]).long()

        # Flip v coordinate to match origin='lower' convention (y=0 at bottom)
        v = img_height - 1 - v

        mask = (u >= 0) & (u < img_width) & (v >= 0) & (v < img_height)

        if verbose:
            print(f"  Points within image bounds: {torch.sum(mask).item()}/{len(u)} ({100*torch.sum(mask).item()/len(u):.1f}%)")

        if torch.sum(mask).item() == 0:
            if verbose:
                print("  WARNING: No points within image bounds!")
                print("  Suggestion: Try adjusting camera position, focal length, or image size")
            # Return empty image and full reflectance map
            return torch.zeros((img_height, img_width), dtype=torch.float32), self.reflectance_map

        # Apply mask to keep only valid points
        u, v = u[mask], v[mask]
        refl = refl[mask]
        depth = Xc[mask,2]  # Z-depth for each valid point

        # VECTORIZED Z-BUFFER (FIXED - proper lexicographic sort)
        # Convert 2D pixel coordinates to 1D linear indices
        pixel_indices = v * img_width + u  # Linear index for each point
        
        # Proper lexicographic sort: sort by depth first, then use stable sort by pixel
        # Step 1: Sort by depth (closest first)
        depth_sort_idx = torch.argsort(depth)
        pixel_indices_depth_sorted = pixel_indices[depth_sort_idx]
        refl_depth_sorted = refl[depth_sort_idx]
        depth_sorted = depth[depth_sort_idx]
        
        # Step 2: Stable sort by pixel index (keeps depth ordering within each pixel)
        # PyTorch's argsort is stable, so within each pixel, depths remain ordered
        pixel_sort_idx = torch.argsort(pixel_indices_depth_sorted, stable=True)
        pixel_indices_sorted = pixel_indices_depth_sorted[pixel_sort_idx]
        refl_sorted = refl_depth_sorted[pixel_sort_idx]
        
        # Find first occurrence of each unique pixel (which is the closest due to sorting)
        # Create mask for first occurrence (closest point for each pixel)
        first_occurrence_mask = torch.cat([
            torch.tensor([True], device=pixel_indices.device),
            pixel_indices_sorted[1:] != pixel_indices_sorted[:-1]
        ])
        
        # Keep only the closest point for each pixel
        final_pixel_indices = pixel_indices_sorted[first_occurrence_mask]
        final_refl = refl_sorted[first_occurrence_mask]
        
        # Create output image and scatter the reflectance values
        img_flat = torch.zeros((img_height * img_width,), dtype=torch.float32, device=refl.device)
        img_flat[final_pixel_indices] = final_refl
        img = img_flat.reshape(img_height, img_width)

        # # LOOP-BASED Z-BUFFER (SLOWER BUT SIMPLER)
        # img = torch.zeros((img_height, img_width), dtype=torch.float32)
        # zbuf = torch.full((img_height, img_width), float('inf'), dtype=torch.float32)
        # for i in range(len(u)):
        #     ui, vi = u[i], v[i]
        #     if depth[i] < zbuf[vi, ui]:  # Keep closest point
        #         img[vi, ui] = refl[i]
        #         zbuf[vi, ui] = depth[i]

        # flip y axis back to origin='lower' convention
        img = torch.flipud(img)
        img = torch.fliplr(img)

        if verbose:
            print(f"  Rendered image statistics:")
            print(f"    Non-zero pixels: {torch.count_nonzero(img).item()}/{img.numel()} ({100*torch.count_nonzero(img).item()/img.numel():.1f}%)")
            print(f"    Value range: [{img.min().item():.6f}, {img.max().item():.6f}]")
            if torch.count_nonzero(img).item() > 0:
                print(f"    Mean non-zero value: {img[img > 0].mean().item():.6f}")

        # Return both the camera image and the full reflectance map
        return img, self.reflectance_map

    
    def compute_shadow_map(dem_array, sun_az_deg, sun_el_deg, cellsize=1):
        """
        Torch-based scan-line shadow map (sun at infinity) returning the
        backward (bottom->top) scan result as the final lit map (1=lit, 0=shadow).

        This version runs at the DEM's native resolution and does not perform
        any internal upsampling/downsamping.

        Cleaning options:
        - If `clean` is True, a pure-Torch neighborhood majority filter is applied
            with a window of size `neighborhood_size`. A pixel is lit if at least
            `neighborhood_threshold` fraction of the window is lit.

        Parameters:
        dem_array: 2D numpy array or torch tensor
        sun_az_deg, sun_el_deg: sun geometry in degrees
        cellsize: size of DEM cells (meters)
        clean: whether to apply neighborhood cleaning
        neighborhood_size: odd int fallback window size (pure-Torch)
        neighborhood_threshold: fraction in (0,1] of lit pixels in window to consider lit
        """

        AlignCorners = True

        if torch.is_tensor(dem_array):
            dem_t = dem_array
        else:
            dem_t = torch.from_numpy(np.array(dem_array)).to(dtype=torch.float32)

        if dem_t.dim() != 2:
            raise ValueError('dem_array must be 2D')

        device = dem_t.device
        dem_t = dem_t.to(dtype=torch.float32, device=device)
        H, W = dem_t.shape

        # pad to larger square (diagonal) to avoid clipping on rotation
        diag = int(math.ceil(math.hypot(H, W)))
        pad_h = max(0, (diag - H) // 2 + 2)  # small extra margin
        pad_w = max(0, (diag - W) // 2 + 2)

        dem_b = dem_t.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        if pad_h > 0 or pad_w > 0:
            dem_b = F.pad(dem_b, (pad_w, pad_w, pad_h, pad_h), mode='constant', value=0.0)

        cellsize_up = cellsize

        H_p, W_p = dem_b.shape[-2], dem_b.shape[-1]

        # Rotate so sun az points along image rows (clockwise rotation by az)
        az = float(sun_az_deg)
        theta = math.radians(-az)
        cos_t = math.cos(theta); sin_t = math.sin(theta)
        theta_mat = torch.tensor([[[cos_t, -sin_t, 0.0], [sin_t, cos_t, 0.0]]], dtype=torch.float32, device=device)

        grid = F.affine_grid(theta_mat, dem_b.size(), align_corners=AlignCorners)
        dem_rot = F.grid_sample(dem_b, grid, mode='bilinear', padding_mode='zeros', align_corners=AlignCorners)

        # Mask of valid (non-padded) pixels after rotation
        ones = torch.ones_like(dem_b)
        mask_rot = F.grid_sample(ones, grid, mode='nearest', padding_mode='zeros', align_corners=AlignCorners)

        # Transformed field T = elevation - (row_index * cellsize_up * tan(el))
        el_rad = math.radians(sun_el_deg)
        tan_el = math.tan(el_rad)
        rows = torch.arange(H_p, dtype=torch.float32, device=device).view(1, 1, H_p, 1) * cellsize_up
        T = dem_rot - rows * tan_el  # shape (1,1,H_p,W_p)

        # Mark padding as -inf so it never occludes
        neg_inf = -float('inf')
        T = torch.where(mask_rot == 0, torch.tensor(neg_inf, device=device, dtype=T.dtype), T)

        # Explicit backward row-wise scan (bottom -> top)
        lit_rot = torch.zeros_like(T, dtype=torch.bool)
        prev_max = torch.full((1, 1, 1, W_p), neg_inf, device=device, dtype=T.dtype)
        for r in range(H_p-1, -1, -1):
            cur = T[:, :, r:r+1, :]
            is_lit = cur >= prev_max
            lit_rot[:, :, r:r+1, :] = is_lit
            prev_max = torch.maximum(prev_max, cur)

        # Rotate lit map back to original orientation
        lit_rot_f = lit_rot.float()
        theta_inv = math.radians(az)
        cos_i = math.cos(theta_inv); sin_i = math.sin(theta_inv)
        theta_mat_inv = torch.tensor([[[cos_i, -sin_i, 0.0], [sin_i, cos_i, 0.0]]], dtype=torch.float32, device=device)
        grid_inv = F.affine_grid(theta_mat_inv, dem_b.size(), align_corners=AlignCorners)

        # inverse-sample lit map and reproject validity mask
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

        shadow_map_t = shadow_cropped.squeeze(0).squeeze(0).to(dtype=torch.uint8)

        if shadow_map_t.shape != dem_t.shape:
            raise ValueError(f"Shadow map shape {tuple(shadow_map_t.shape)} does not match DEM shape {tuple(dem_t.shape)}")

        return shadow_map_t
