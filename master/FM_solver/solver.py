import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.ndimage import zoom
import torch

from .solver_utils import (
    unit_vec_from_az_el,
    compute_normal_from_gradients,
    gradients_to_dfdx_dfdy,
    build_G,
    build_G_transpose,
    build_G_prime,
    build_G_prime_transpose,
    downsample_and_smooth_dem,
    validate_normal_field,
    compute_angular_error
)



class FMSolver:
    """
    Fernandes-Mosegaard Photometric Stereo Solver for Planetary DEM Refinement.
    
    This solver combines photometric stereo (estimating surface normals from
    multi-illumination images) with a prior DEM using Bayesian inference.
    It follows the approach of Fernandes & Mosegaard (2022).
    
    COORDINATE SYSTEM CONVENTION:
    -----------------------------
    - x-axis: Easting (positive East)
    - y-axis: Northing (positive North)
    - z-axis: Up (positive upward)
    - Right-handed coordinate system
    
    ARRAY INDEXING:
    - dem[row, col] = dem[y_index, x_index]
    - Row index increases Northward (when origin='lower')
    - Column index increases Eastward
    
    ANGLE CONVENTIONS:
    - Azimuth: 0° = North, 90° = East, 180° = South, 270° = West
    - Elevation: 0° = horizon, 90° = zenith
    
    NORMAL VECTOR CONVENTION:
    - Surface normals point upward (positive z-component)
    - n = [-dz/dx, -dz/dy, 1] normalized
    - nx points "away" from positive slope in x-direction (East)
    - ny points "away" from positive slope in y-direction (North)
    
    Parameters
    ----------
    reflectance_maps : tuple of torch.Tensor
        K reflectance maps, each of shape (H, W).
    sun_azs : list of float
        K azimuth angles in degrees.
    sun_el : float
        Solar elevation angle in degrees (same for all images).
    gt_dem : torch.Tensor or np.ndarray
        Ground truth DEM with shape (H, W).
    scale_down_factor : int, optional
        Downsampling factor for prior DEM. Default 40.
    sigma_smooth : float, optional
        Gaussian smoothing sigma for prior DEM. If None, uses scale_down_factor / 2.
    max_slope_deviation_prior : float, optional
        Maximum expected surface slope deviation in degrees. Used for prior covariance. Default 30.
    noise_fraction_brightness : float, optional
        Brightness noise as fraction of max reflectance. Default 0.05.
    debug : bool, optional
        Enable verbose debug output. Default False.
    sigma_data : float, optional
        Standard deviation of data noise (in reflectance units). Default 1.0.
    sigma_m : float, optional
        Standard deviation of the prior model (in elevation units). Default 1.0.
    """
    
    def __init__(self, reflectance_maps, sun_azs, sun_el, gt_dem, scale_down_factor=40, sigma_smooth=None, max_slope_deviation_prior=30, 
                 noise_fraction_brightness=0.05, pixel_size=1.0, debug=False, sigma_M = 1.0, cellsize=1.0):
        self.debug = debug
        self._debug_print("="*70)
        self._debug_print("Initializing FMSolver")
        self._debug_print("="*70)
        
        self.cellsize = cellsize
        self.prior_has_been_updated = False  # Flag to track if the DEM prior has been updated since last estimation
        # Store illumination parameters
        self.K = len(reflectance_maps)
        self.sun_azs = sun_azs
        self.sun_el = sun_el
        
        self._debug_print(f"Number of illumination angles: {self.K}")
        self._debug_print(f"Sun azimuths: {sun_azs}°")
        self._debug_print(f"Sun elevation: {sun_el}°")
        self._debug_print(f"Pixel size (dx=dy): {pixel_size}")
        self._debug_print(f"Cell size: {self.cellsize}")
        
        # Convert reflectance maps to numpy and vectorize
        reflectance_maps_np = [rm.detach().cpu().numpy() for rm in reflectance_maps]
        H, W = reflectance_maps_np[0].shape
        self._debug_print(f"Reflectance map shape: ({H}, {W})")
        
        # Stack as vectors: each pixel has K reflectance values
        self.reflectance_maps_vectors = np.stack([rm.flatten() for rm in reflectance_maps_np], axis=1)  # shape (H*W, K)
        self._debug_print(f"Vectorized reflectance shape: {self.reflectance_maps_vectors.shape}")
        
        # Convert and store ground truth DEM
        self.gt_dem = gt_dem.cpu().numpy() if torch.is_tensor(gt_dem) else gt_dem
        self._debug_print(f"Ground truth DEM shape: {self.gt_dem.shape}")
        self._debug_print(f"DEM elevation range: [{self.gt_dem.min():.2f}, {self.gt_dem.max():.2f}]")
        
        # Compute sun direction vectors
        self.sun_normals = self.set_sun_normals()  # shape (K, 3)

        # Pixel spacing (meters per pixel)
        self.dx = float(pixel_size)
        self.dy = float(pixel_size)
        
        # Set up Bayesian priors
        # Create smoothed prior DEM
        self.dem_lower, self.dem_prior = self.compute_prior_dem(
            self.gt_dem, 
            scale_down_factor=scale_down_factor, 
            smooth_sigma_px=sigma_smooth
        )
        self.dem_prior_initial = self.dem_prior.copy()  # Store the initial downscaled DEM for reference
        
        self.dem_prior_normals = self.compute_surface_normals(self.dem_prior, self.cellsize)  # Precompute normals from prior DEM for covariance setup
        
        self.set_covariance_priors(max_slope_deviation=max_slope_deviation_prior)
        self.set_covariance_brightness(reflectance_maps_np, noise_fraction=noise_fraction_brightness)
        

        # Vals for finite difference regularization
        self.sigma_M = sigma_M
        
        # Cached photometric estimates (normals/albedo)
        self.normals_est = None
        self.albedo_est = None
        self.estimation_device = None
        
        self._debug_print("="*70)
        self._debug_print("FMSolver initialization complete")
        self._debug_print("="*70)
    
    def _debug_print(self, message):
        """Print debug message if debug mode is enabled."""
        if self.debug:
            print(f"[FMSolver] {message}")

    def _clear_estimates(self):
        """Invalidate cached normal and albedo estimates."""
        self.normals_est = None
        self.albedo_est = None
        self.estimation_device = None

    def _select_device(self, device=None):
        """Choose computation device with preference CUDA > CPU (MPS avoided by default)."""
        if device is not None:
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        # MPS can be unstable on some macOS setups for large allocations; default to CPU if CUDA is absent
        return torch.device("cpu")
    
    
    def set_sun_normals(self):
        """
        Compute sun direction vectors from azimuth and elevation angles.
        
        Convention:
        - Azimuth: 0° = North, 90° = East, 180° = South, 270° = West
        - Elevation: 0° = horizon, 90° = zenith
        - Returns unit vectors in [x_east, y_north, z_up] coordinates
        
        Returns
        -------
        np.ndarray, shape (K, 3)
            Sun direction unit vectors. Each row is [sx, sy, sz].
        """
        self._debug_print("\n--- Computing Sun Direction Vectors ---")
        
        sun_vecs = np.zeros((self.K, 3), dtype=float)
        for i, az in enumerate(self.sun_azs):
            sun_vecs[i] = unit_vec_from_az_el(az, self.sun_el)
            
            if self.debug:
                # Verify direction for sanity check
                direction = ""
                if np.abs(az - 0) < 1:    direction = "North"
                elif np.abs(az - 90) < 1: direction = "East"
                elif np.abs(az - 180) < 1: direction = "South"
                elif np.abs(az - 270) < 1: direction = "West"
                
                self._debug_print(f"  Sun {i}: az={az:.1f}°, el={self.sun_el:.1f}° {direction}")
                self._debug_print(f"    Vector: [{sun_vecs[i, 0]:+.4f}, {sun_vecs[i, 1]:+.4f}, {sun_vecs[i, 2]:+.4f}]")
                self._debug_print(f"    |v| = {np.linalg.norm(sun_vecs[i]):.6f} (should be 1.0)")
        
        # Validate unit vectors
        if self.debug:
            self._validate_sun_vectors(sun_vecs)
        
        self.sun_normals = sun_vecs
        return self.sun_normals
    
    # def set_covariance_priors(self, max_slope_deviation=10):
    #     """
    #     Set covariance matrix for normal vector prior.
        
    #     Assumes Gaussian prior centered at zero slope (vertical normal)
    #     with standard deviation corresponding to max_slope.
        
    #     Parameters
    #     ----------
    #     max_slope_deviation : float
    #         Maximum expected surface slope deviation in degrees.
    #     """
    #     self._debug_print("\n--- Setting Normal Prior Covariance ---")
        
    #     slope_variance_rad = np.deg2rad(max_slope_deviation) ** 2
    #     covariance_prior = np.eye(3) * slope_variance_rad  # shape (3, 3)
    #     covariance_prior[2, 2] = 1e-4  # Very low variance for nz
        
    #     self._debug_print(f"Max expected slope deviation: {max_slope_deviation}°")
    #     self._debug_print(f"Slope variance (rad²): {slope_variance_rad:.6f}")
    #     self._debug_print(f"Slope std dev (rad): {np.sqrt(slope_variance_rad):.6f}")
    #     self._debug_print(f"Slope std dev (deg): {np.sqrt(slope_variance_rad) * 180 / np.pi:.2f}°")
        
    #     # covariance_prior is (3,3) - but they will later be distinct for each pixel, so expand to (N,3,3) with same values for all pixels
    #     covariance_prior_full = np.tile(covariance_prior, (self.reflectance_maps_vectors.shape[0], 1, 1))  # shape (N, 3, 3)
    #     print(f"Covariance prior shape after tiling: {covariance_prior_full.shape}")
        
    #     self.covariance_prior = covariance_prior_full
    #     self.covariance_prior_T = covariance_prior_full.transpose(0, 2, 1)  # Precompute transpose 
    
    # def set_covariance_priors(self, max_slope=10, scale_down_factor=2):
        
    #     print("\n--- Setting Normal Prior Covariance with Rotation ---")
    #     # 1) Lokal varians
    #     theta = np.deg2rad(max_slope / scale_down_factor)
    #     sigma_theta2 = theta**2
    #     sigma_parallel2 = theta**4  # meget lille, stabilt valg

    #     Cov_local = np.diag([sigma_theta2, sigma_theta2, sigma_parallel2])

    #     # 2) Normaliser prior-normals
    #     n_p = self.dem_prior_normals.reshape(-1, 3)
    #     n_p = n_p / (np.linalg.norm(n_p, axis=1, keepdims=True) + 1e-12)

    #     # 3) Find rotationsmatricer, vectoriseret
    #     e3 = np.array([0.0, 0.0, 1.0])
    #     v = np.cross(np.tile(e3, (len(n_p), 1)), n_p)
    #     s = np.linalg.norm(v, axis=1, keepdims=True)
    #     c = np.dot(e3, n_p.T).T  # shape (N,1)

    #     # Skalér vektoriseret krydsproduktmatrix
    #     vx = np.zeros((len(n_p), 3, 3))
    #     vx[:,0,1] = -v[:,2]
    #     vx[:,0,2] =  v[:,1]
    #     vx[:,1,0] =  v[:,2]
    #     vx[:,1,2] = -v[:,0]
    #     vx[:,2,0] = -v[:,1]
    #     vx[:,2,1] =  v[:,0]

    #     R = np.eye(3)[None,:,:] + vx + np.einsum(
    #         'nij,njk->nik',
    #         vx,
    #         vx
    #     ) * ((1 - c) / (s**2 + 1e-15))[:,None,None]

    #     # Hvis parallelt → brug identitet
    #     parallel = (s[:,0] < 1e-12)
    #     R[parallel,:,:] = np.eye(3)

    #     # 4) Rotér lokal kovarians → global
    #     covariance_prior_full = R @ Cov_local @ R.transpose(0,2,1)

    #     self.covariance_prior = covariance_prior_full
    #     self.covariance_prior_T = covariance_prior_full.transpose(0,2,1)
    #     print(f"Covariance prior shape after rotation: {covariance_prior_full.shape}")
    
    def set_covariance_priors(self, max_slope_deviation=10, scale_down_factor=2):
        
        self._debug_print("\n--- Setting Normal Prior Covariance with Rotation ---")
        theta = np.deg2rad(max_slope_deviation / scale_down_factor)
        sigma_theta2 = theta**2
        sigma_parallel2 = theta**4

        n_p = self.dem_prior_normals.reshape(-1, 3)
        n_p = n_p / (np.linalg.norm(n_p, axis=1, keepdims=True) + 1e-12)

        I = np.eye(3)[None,:,:]
        outer = n_p[:, :, None] @ n_p[:, None, :]

        covariance_prior_full = (
            sigma_theta2 * (I - outer)
            + sigma_parallel2 * outer
        )

        self.covariance_prior = covariance_prior_full
        self.covariance_prior_T = covariance_prior_full.transpose(0,2,1)
        self._debug_print(f"Covariance prior shape after rotation: {covariance_prior_full.shape}")

        

    def set_covariance_brightness(self, reflectance_maps, noise_fraction=0.1):
        """
        Set covariance matrix for brightness measurement noise.
        
        Assumes independent Gaussian noise on each reflectance measurement,
        with standard deviation proportional to max brightness.
        
        Parameters
        ----------
        reflectance_maps : list of np.ndarray
            K reflectance maps.
        noise_fraction : float
            Noise as fraction of max reflectance.
        """
        self._debug_print("\n--- Setting Brightness Covariance ---")
        
        # Compute max brightness across all K maps
        reflectance_max = np.max(reflectance_maps)
        reflectance_mean = np.mean(reflectance_maps)
        
        # Noise scales with brightness
        brightness_std = noise_fraction * reflectance_max
        brightness_variance = brightness_std**2
        
        self._debug_print(f"Max reflectance: {reflectance_max:.6f}")
        self._debug_print(f"Mean reflectance: {reflectance_mean:.6f}")
        self._debug_print(f"Noise fraction: {noise_fraction:.3f}")
        self._debug_print(f"Brightness std dev: {brightness_std:.6f}")
        self._debug_print(f"Brightness variance: {brightness_variance:.6f}")
        
        self.brightness_covariance = np.eye(self.K) * brightness_variance


    def compute_prior_dem(self, gt_dem, scale_down_factor=40, smooth_sigma_px=5):
        """
        Create a smoothed, upscaled low-resolution prior DEM.
        
        Follows Fig. 1C in Fernandes & Mosegaard (2022):
        1. Downsample DEM by scale_down_factor
        2. Upsample back to original resolution
        3. Apply Gaussian smoothing
        
        Parameters
        ----------
        gt_dem : np.ndarray
            Ground truth DEM.
        scale_down_factor : int
            Downsampling factor.
        smooth_sigma_px : float, optional
            Gaussian smoothing sigma in pixels.
        
        Returns
        -------
        dem_low : np.ndarray
            Downsampled DEM (not upsampled).
        dem_prior : np.ndarray
            Smoothed prior DEM with original shape.
        """
        self._debug_print("\n--- Computing Prior DEM ---")
        self._debug_print(f"Original DEM shape: {gt_dem.shape}")
        self._debug_print(f"Original elevation range: [{gt_dem.min():.2f}, {gt_dem.max():.2f}]")
        self._debug_print(f"Scale down factor: {scale_down_factor}")
        
        # smooth_sigma_px is in pixels (original convention); report the physical scale
        smooth_sigma_m = smooth_sigma_px * self.dx
        self._debug_print(
            f"Smoothing sigma: {smooth_sigma_px:.2f} px ({smooth_sigma_m:.2f} m @ dx={self.dx:.4f} m)"
        )
        
        dem_low, dem_prior = downsample_and_smooth_dem(
            gt_dem, 
            scale_down_factor=scale_down_factor, 
            smooth_sigma=smooth_sigma_px
        )
        
        self._debug_print(f"Downsampled DEM shape: {dem_low.shape}")
        self._debug_print(f"Prior DEM shape: {dem_prior.shape}")
        self._debug_print(f"Prior elevation range: [{dem_prior.min():.2f}, {dem_prior.max():.2f}]")
        self._debug_print(f"Elevation std - Original: {gt_dem.std():.2f}, Prior: {dem_prior.std():.2f}")
        
        return dem_low, dem_prior
    
            

    # def get_prior_normal(self, pixel_idx):
    #     """
    #     Get prior normal vector from prior DEM at a given pixel.
        
    #     Uses central differences to compute gradients, then converts
    #     to normal using the standard convention: n = [-dz/dx, -dz/dy, 1] normalized.
        
    #     Parameters
    #     ----------
    #     pixel_idx : int
    #         Flattened pixel index.
        
    #     Returns
    #     -------
    #     np.ndarray, shape (3,)
    #         Unit normal vector [nx, ny, nz].
    #     """
    #     H, W = self.dem_prior.shape
    #     y, x = divmod(pixel_idx, W)
        
    #     # Compute gradient using central differences (if not on boundary)
    #     if x > 0 and x < W-1 and y > 0 and y < H-1:
    #         dz_dx = (self.dem_prior[y, x+1] - self.dem_prior[y, x-1]) / (2 * self.dx)
    #         dz_dy = (self.dem_prior[y+1, x] - self.dem_prior[y-1, x]) / (2 * self.dy)
    #         n_prior = compute_normal_from_gradients(dz_dx, dz_dy)
    #     else:
    #         # Boundary: assume flat (vertical normal)
    #         n_prior = np.array([0, 0, 1])
        
    #     return n_prior
    
    
    # def _compute_surface_normals_from_dem(self, dem):
    #     """
    #     Computes full surface normals from a DEM using numpy gradients (no torch ops).
    #     Caches the result on the instance as dem_prior_normals.
    #     """

    #     dem_np = dem.cpu().numpy() if torch.is_tensor(dem) else dem

    #     # Central differences in y (axis 0) and x (axis 1); np.gradient accepts physical spacing
    #     dz_dy, dz_dx = np.gradient(dem_np, self.dy, self.dx)

    #     nx = -dz_dx
    #     ny = -dz_dy
    #     nz = np.ones_like(dem_np)

    #     norm = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-12

    #     self.nx = nx / norm
    #     self.ny = ny / norm
    #     self.nz = nz / norm

    #     normals = np.stack([self.nx, self.ny, self.nz], axis=-1)  # shape (H, W, 3)
    #     self.dem_prior_normals = normals
    #     return normals
    
    def compute_surface_normals(self, dem, cellsize):
        """
        Computes the surface normal vectors for each grid cell using the DEM gradients.
        Uses self.y_down to determine if y-gradient should be inverted.
        """
        # Compute gradients in y and x directions using PyTorch
        # Add batch and channel dimensions for gradient computation
        
        dem_tensor = torch.from_numpy(dem).float().unsqueeze(0).unsqueeze(0) if isinstance(dem, np.ndarray) else dem.float().unsqueeze(0).unsqueeze(0)
        device = 'cpu'  # Force CPU for this computation to avoid GPU memory issues
        
        # Sobel-like gradients for y and x
        # Create gradient kernels
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=device)/ (8 * cellsize)
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device) / (8 * cellsize)
        ky = ky.view(1, 1, 3, 3)
        kx = kx.view(1, 1, 3, 3)

        
        # Apply convolution for gradients
        dz_dy_raw = torch.nn.functional.conv2d(dem_tensor, ky, padding=1)[0, 0] # shape (H, W)
        dz_dx = torch.nn.functional.conv2d(dem_tensor, kx, padding=1)[0, 0]  # shape (H, W)
        
        # Optionally invert y-gradient for image coordinate systems
        dz_dy = dz_dy_raw
        
        # Surface normal components
        nx = -dz_dx
        ny = -dz_dy
        nz = torch.ones_like(dz_dx, device=device)
        
        # Normalize the normal vectors
        norm = torch.sqrt(nx*nx + ny*ny + nz*nz) + 1e-12
        self.nx = nx / norm
        self.ny = ny / norm
        self.nz = nz / norm
        
        normals = torch.stack([self.nx, self.ny, self.nz], dim=-1)  # shape (H, W, 3)
        return normals.cpu().numpy()

    def _get_normal_from_dem_pixel(self, dem, pixel_id):
        """
        Retrieve a single normal vector for a flattened pixel index.
        Computes normals if they are not already cached or shape-mismatched.
        """

        dem_np = dem.cpu().numpy() if torch.is_tensor(dem) else dem
        H, W = dem_np.shape

        # Compute normals if missing or outdated
        cached = getattr(self, "dem_prior_normals", None)
        if cached is None or cached.shape[:2] != dem_np.shape:
            cached = self.compute_surface_normals(dem_np, self.cellsize)

        y, x = divmod(pixel_id, W)
        return cached[y, x]
    
    def update_covariance_prior_normals(self, new_covariance):
        """
        Update the covariance matrix for the normal prior and invalidate cached estimates.
        
        Parameters
        ----------
        new_covariance : np.ndarray, shape (N, 3, 3)
            New covariance matrix for the normal prior.
        """
        self._debug_print("\n--- Updating Normal Prior Covariance ---")
        self._debug_print(f"New covariance shape:\n{new_covariance.shape}")
        
        self.covariance_prior = new_covariance
        self.covariance_prior_T = new_covariance.transpose(0, 2, 1)  # Update transpose as well
    
    
    def estimate_normal_and_albedo(self, n_iters=3, device=None):
        """
        Joint MAP estimate of surface normal and albedo at each pixel.

        Uses batched torch ops with optional GPU (CUDA preferred; CPU fallback if GPU fails).
        Caches the result to avoid recomputation when called repeatedly.

        Parameters
        ----------
        n_iters : int
            Number of refinement iterations. Default 3.
        device : str or torch.device, optional
            Force computation device. If None, auto-selects CUDA else CPU (MPS not selected by default).

        Returns
        -------
        normals : np.ndarray, shape (H, W, 3)
            Estimated unit normal vectors.
        albedos : np.ndarray, shape (H, W)
            Estimated albedo values.
        """
        self._debug_print("\n" + "="*70)
        self._debug_print("Starting Normal & Albedo Estimation")
        self._debug_print("="*70)

        compute_device = self._select_device(device)
        dtype = torch.float64

        # Return cached estimates if available for this device
        if not self.prior_has_been_updated:
            if (
                self.normals_est is not None
                and self.albedo_est is not None
                and self.estimation_device == str(compute_device)
            ):
                self._debug_print("Using cached normals/albedo")
                return self.normals_est, self.albedo_est

        H, W = self.dem_prior.shape
        N_pixels = H * W

        self._debug_print(f"Image size: {H} x {W} = {N_pixels} pixels")
        self._debug_print(f"Refinement iterations: {n_iters}")
        self._debug_print(f"Compute device: {compute_device}")

        def run_estimation(target_device):
            # Move inputs to the chosen device
            delta_dev = torch.from_numpy(self.reflectance_maps_vectors).to(device=target_device, dtype=dtype)  # (N, K)
            S_dev = torch.from_numpy(self.sun_normals).to(device=target_device, dtype=dtype)  # (K, 3) sun direction vectors
        
            # if run first time C_n is (3,3), same for all pixels - otherwise it is (N,3,3) and different for each pixel; we need the inverse for the linear system, so precompute that
            C_n_inv_dev = torch.from_numpy(np.linalg.inv(self.covariance_prior)).to(device=target_device, dtype=dtype)  # (N, 3, 3)
            C_delta_inv_dev = torch.from_numpy(np.linalg.inv(self.brightness_covariance)).to(device=target_device, dtype=dtype)  # (K, K)

            # Build prior normals in batch using cached numpy computation
            n_prior_np = self.compute_surface_normals(self.dem_prior, self.cellsize)
            n_prior_dev = torch.from_numpy(n_prior_np).to(device=target_device, dtype=dtype)

            self._debug_print(f"reflectance_maps_vectors shape: {self.reflectance_maps_vectors.shape}")
            self._debug_print(f"sun_normals shape: {self.sun_normals.shape}")
            # also print sun normals for debugging
            for i in range(self.K):
                self._debug_print(f"Sun normal {i}: [{self.sun_normals[i, 0]:+.4f}, {self.sun_normals[i, 1]:+.4f}, {self.sun_normals[i, 2]:+.4f}]")
            self._debug_print(f"n_prior_np shape: {n_prior_np.shape}")

            # H, W, _ = n_prior_np.shape
            # N = H * W # number of pixels
            # S = self.sun_normals  # (K,3)
            # delta = self.reflectance_maps_vectors  # (N,K) number of pixels x number of sun directions

            # n_flat = n_prior_np.reshape(N, 3)      # (N,3) number of pixels x 3 components (nx, ny, nz)
            # Sn = n_flat @ S.T                      # (N,K) predicted reflectance at each pixel for each sun direction based on prior normals. 
            #                                        #  Sun normal multiplied with pixel normal gives cosine of angle between them, which is proportional to reflectance in Lambertian model.
            

            # eps = 1e-6
            # ratio = delta / (Sn + eps)             # (N,K) ratio of observed to predicted reflectance based on prior normals; should be close to 1 if prior is good

            # print("Lambert test (delta / (S n_prior)):")
            # for k in range(S.shape[0]):
            #     r_k = ratio[:, k]
            #     print(f"  k={k}: min={r_k.min():.4f}, max={r_k.max():.4f}, mean={r_k.mean():.4f}, std={r_k.std():.4f}")


            # # Tjek at N stemmer
            # N_delta, K_delta = self.reflectance_maps_vectors.shape
            # assert N_delta == N_pixels, f"N mismatch: {N_delta} vs {N_pixels}"
            # K_sun = self.sun_normals.shape[0]
            # assert K_delta == K_sun, f"K mismatch: {K_delta} vs {K_sun}"
            
            # y, x = 10, 20
            # idx = y * W + x

            # self._debug_print(f"Pixel idx={idx}, (y,x)=({y},{x})")
            # self._debug_print(f"Prior normal at that pixel: {n_prior_np[y, x]}")  # (3,)
            # self._debug_print(f"Same via flat reshape: {n_prior_np.reshape(-1,3)[idx]}")
            # self._debug_print(f"Reflectances at that pixel: {self.reflectance_maps_vectors[idx]}")
            
            
            n_dev = n_prior_dev.reshape(N_pixels, 3) # (N,3) number of pixels x 3 components (nx, ny, nz); initialized to prior normals; will be updated iteratively

            # Precompute shared terms
            ST_C_del_inv = S_dev.T @ C_delta_inv_dev  # (3, K)
            A_mat = ST_C_del_inv @ S_dev  # (3, 3) S^T C_delta^-1 S, used in the linear system for normal update
            delta_C = delta_dev @ C_delta_inv_dev  # (N, K)
            
            # Hvis C_n_inv_dev er (N,3,3), brug batch-matmul:
            # n_prior_C_n_inv[i] = C_n_inv[i] @ n_prior[i]
            n_prior_flat = n_prior_dev.reshape(N_pixels, 3)  # (N,3)
            n_prior_C_n_inv = torch.bmm(
                C_n_inv_dev,                 # (N,3,3)
                n_prior_flat.unsqueeze(-1)   # (N,3,1)
            ).squeeze(-1)                    # (N,3)


            # def compute_albedo(curr_n):
            #     Sn = curr_n @ S_dev.T  # (N, K)
            #     Sn_C = Sn @ C_delta_inv_dev  # (N, K)
            #     num = (Sn * delta_C).sum(dim=1)
            #     den = (Sn * Sn_C).sum(dim=1)
            #     return torch.where(den > 1e-12, num / den, torch.zeros_like(den))
            
            def compute_albedo(curr_n): # this is eq. (11) in Fernandes & Mosegaard (2022) solved for albedo, where we assume components of C_n^-1 are small
                Sn = curr_n @ S_dev.T          # (N, K)
                Sn_C = Sn @ C_delta_inv_dev    # (N, K)
                num = (Sn * delta_C).sum(dim=1)
                den = (Sn * Sn_C).sum(dim=1)
                return torch.where(den > 1e-12, num / den, torch.zeros_like(den))

            # # DEBUG: albedo KUN med DEM-normaler
            # albedo_dem = compute_albedo(n_dev)  # n_dev = prior normals (N,3)
            # albedo_dem_np = albedo_dem.reshape(H, W).detach().cpu().numpy()
            # self._debug_print("[DEBUG] Albedo from DEM normals only:")
            # self._debug_print(
            #     f"  range: [{albedo_dem_np.min():.4f}, {albedo_dem_np.max():.4f}]"
            # )
            # self._debug_print(
            #     f"  mean: {albedo_dem_np.mean():.4f}, std: {albedo_dem_np.std():.4f}"
            # )
            
            
            # Iterative refinement (batched)
            albedo = compute_albedo(n_dev) # shape (N,)
            
            for _ in range(n_iters):
                # Build per-pixel linear systems
                # M_i = a_i^2 * A_mat + C_n_inv_i, where A_mat is S^T C_delta^-1 S 
                M_batch = (albedo ** 2).view(N_pixels, 1, 1) * A_mat + C_n_inv_dev  # (N,3,3)    
                               
                # rhs_i = a_i * S^T C_delta^-1 δ_i + C_n_inv_i n_prior_i
                data_term = (ST_C_del_inv @ delta_dev.T).T           # (N,3)
                rhs_batch = albedo.view(N_pixels, 1) * data_term + n_prior_C_n_inv  # (N,3)


                n_raw = torch.linalg.solve(M_batch, rhs_batch.unsqueeze(-1)).squeeze(-1) # This sets up eq. (9) in Fernandes & Mosegaard (2022) as a linear system M n = rhs for each pixel, where M is the combined precision matrix of the prior and data terms, and rhs is the combined influence of the observed reflectance and the prior normal. We solve for n (the normal vector) using torch.linalg.solve, which is efficient for small systems.
                n_norm = torch.linalg.norm(n_raw, dim=1, keepdim=True).clamp(min=1e-12)
                n_dev = n_raw / n_norm

                albedo = compute_albedo(n_dev)

            # Also compute updated covariance matrix after estimation
            updated_covariance_prior_normals = torch.linalg.inv((albedo ** 2).unsqueeze(-1).unsqueeze(-1) * A_mat + C_n_inv_dev.unsqueeze(0)) # (N, 3, 3) eq. (10) in Fernandes & Mosegaard (2022)
            
            return n_dev, albedo, updated_covariance_prior_normals
        

        # Run estimation with fallback to CPU if the selected device fails
        try:
            n_out, albedo_out, C_n_updated = run_estimation(compute_device)
        except Exception as e:
            self._debug_print(f"Compute on {compute_device} failed ({e}); falling back to CPU")
            compute_device = torch.device("cpu")
            n_out, albedo_out, C_n_updated = run_estimation(compute_device)

        # Move results back to CPU numpy and cache
        normals_np = n_out.reshape(H, W, 3).detach().cpu().numpy()
        albedos_np = albedo_out.reshape(H, W).detach().cpu().numpy()
        cov_np = C_n_updated.detach().cpu().numpy().squeeze()  # shape (N, 3, 3)

        # Debug statistics
        self._debug_print("\n--- Estimation Statistics ---")
        self._debug_print(f"Albedo range: [{albedos_np.min():.4f}, {albedos_np.max():.4f}]")
        self._debug_print(f"Albedo mean: {albedos_np.mean():.4f}, std: {albedos_np.std():.4f}")
        negatives = (albedos_np < 0).sum()
        self._debug_print(f"Negative albedos: {negatives} / {N_pixels} ({100*negatives/N_pixels:.2f}%)")

        if self.debug:
            validation = validate_normal_field(normals_np)
            self._debug_print("\n--- Normal Field Validation ---")
            self._debug_print(f"All unit length: {validation['all_unit']}")
            self._debug_print(f"All upward (z>0): {validation['all_upward']}")
            self._debug_print(f"Mean length: {validation['mean_length']:.6f}")
            self._debug_print(f"Length std dev: {validation['std_length']:.6e}")
            self._debug_print(f"Z-component range: [{validation['min_z']:.4f}, {validation['max_z']:.4f}]")
            self._debug_print(f"Non-unit normals: {validation['num_invalid']}")

        self._debug_print("="*70)

        # Cache and return
        self.normals_est = normals_np
        self.albedo_est = albedos_np
        self.update_covariance_prior_normals(cov_np)  # Update the covariance prior with the new estimates
        self.estimation_device = str(compute_device)

        self.prior_has_been_updated = False  # Reset flag after having updated estimates based on current prior

        return normals_np, albedos_np
    
    def compute_normals_from_dem(self, dem):
        """
        Compute surface normals from DEM using central differences.
        
        Parameters
        ----------
        dem : torch.Tensor or np.ndarray
            Digital elevation model.
        
        Returns
        -------
        np.ndarray, shape (H, W, 3)
            Unit normal vectors at each pixel.
        """
        dem = dem.cpu().numpy() if torch.is_tensor(dem) else dem
        H, W = dem.shape
        normals = np.zeros((H, W, 3))
        
        for y in range(H):
            for x in range(W):
                if x > 0 and x < W-1 and y > 0 and y < H-1:
                    dz_dx = (dem[y, x+1] - dem[y, x-1]) / (2 * self.dx)
                    dz_dy = (dem[y+1, x] - dem[y-1, x]) / (2 * self.dy)
                    normals[y, x] = compute_normal_from_gradients(dz_dx, dz_dy)
                else:
                    normals[y, x] = np.array([0, 0, 1])
        
        return normals
        
    def remove_outer_n_pixels(self, n):
        """
        Remove outer n pixels from all data fields.
        
        Edge pixels have less accurate normals due to boundary effects
        in gradient computation. Remove them before solving.
        
        Parameters
        ----------
        n : int
            Number of pixels to remove from each edge.
        """
        self._debug_print(f"\n--- Removing Outer {n} Pixels ---")
        
        H, W = self.gt_dem.shape
        self._debug_print(f"Original shape: ({H}, {W})")
        
        # Crop all data fields
        self.gt_dem = self.gt_dem[n:H-n, n:W-n]
        self.dem_prior = self.dem_prior[n:H-n, n:W-n]
        self.reflectance_maps_vectors = self.reflectance_maps_vectors.reshape(H, W, self.K)[n:H-n, n:W-n, :].reshape(-1, self.K)

        # Cached photometric estimates are no longer valid after cropping
        self._clear_estimates()
        
        H_new, W_new = self.gt_dem.shape
        self._debug_print(f"New shape: ({H_new}, {W_new})")
        self._debug_print(f"Removed {H*W - H_new*W_new} pixels ({100*(H*W - H_new*W_new)/(H*W):.1f}% of original)")
        
    
    def construct_sylvester_matrices(self):
        """
        Construct Sylvester equation matrices A, B, C for DEM update.
        
        Following Fernandes & Mosegaard (2022), the DEM update M solves:
            A @ M + M @ B = C
        
        where:
        - A operates on rows (North-South differences)
        - B operates on columns (East-West differences)
        - C contains gradient residuals between estimated and prior
        
        Parameters
        ----------
        sigma_data : float
            Data uncertainty parameter.
        sigma_m : float
            Model uncertainty parameter.
        
        Returns
        -------
        A : np.ndarray, shape (H_int, H_int)
            Vertical (row) operator matrix.
        B : np.ndarray, shape (W_int, W_int)
            Horizontal (column) operator matrix.
        C : np.ndarray, shape (H_int, W_int)
            Gradient residual matrix.
        """
        self._debug_print("\n" + "="*70)
        self._debug_print("Constructing Sylvester Equation Matrices")
        self._debug_print("="*70)
        
        H, W = self.dem_prior.shape
        n_int = 1 # Number of pixels to remove from each edge for interior region; should match the gradient computation method (e.g., central differences require n_int=1)
        H_int, W_int = H - 2*n_int, W - 2*n_int  # Interior region (avoid edges)
        
        self._debug_print(f"Full DEM shape: ({H}, {W})")
        self._debug_print(f"Interior region: ({H_int}, {W_int})")
        
        # Build finite difference operators
        self._debug_print("\nBuilding finite difference operators...")
        G_raw  = build_G(H_int)  # Vertical differences (rows/North-South)
        Gp_raw = build_G_prime(W_int)  # Horizontal differences (columns/East-West)
        # Scale operators by physical spacing so gradients are in meters
        G  = G_raw / self.dy
        Gp = Gp_raw / self.dx
        self._debug_print(f"  G shape: {G.shape} (vertical/North-South), scaled by 1/dy={1/self.dy:.4f}")
        self._debug_print(f"  G' shape: {Gp.shape} (horizontal/East-West), scaled by 1/dx={1/self.dx:.4f}")
        
        # Estimate normals and albedo from reflectance
        self._debug_print("\nEstimating normals from photometry...")
        normals_est, albedo_est = self.estimate_normal_and_albedo(n_iters=3)
        
        C_n = self.covariance_prior  # (N, 3, 3) covariance of normal estimates; should have been updated in estimate_normal_and_albedo
        # reshape til (H, W, 3, 3)
        C_n = C_n.reshape(H, W, 3, 3)


        # Vi arbejder kun på interior: Use H_int and W_int to index into the center region of the normal estimates and their covariance
        n_center   = normals_est[n_int:-n_int, n_int:-n_int, :]      # (H_int, W_int, 3)
        C_n_center = C_n[n_int:-n_int, n_int:-n_int, :, :]          # (H_int, W_int, 3, 3)

        n3_center = n_center[..., 2]                  # (H_int, W_int)
        n3_sq_inv = 1.0 / (np.maximum(n3_center, 1e-12)**2)  # (H_int, W_int)

        # top-left 2x2 blok af C_n
        C_n_2x2 = C_n_center[..., 0:2, 0:2]           # (H_int, W_int, 2, 2)

        # Eq. (16): C_r = (1 / n3^2) * C_n^{(2x2)}
        C_r = n3_sq_inv[..., None, None] * C_n_2x2    # (H_int, W_int, 2, 2)

        
        # Convert normals to gradients
        self._debug_print("\nConverting normals to gradients...")
        df_dx_est, df_dy_est = gradients_to_dfdx_dfdy(normals_est)
        
        self._debug_print(f"  df/dx range: [{df_dx_est.min():.4f}, {df_dx_est.max():.4f}]")
        self._debug_print(f"  df/dy range: [{df_dy_est.min():.4f}, {df_dy_est.max():.4f}]")
        self._debug_print(f"  df/dx mean/std: {df_dx_est.mean():+.4f} / {df_dx_est.std():.4f}")
        self._debug_print(f"  df/dy mean/std: {df_dy_est.mean():+.4f} / {df_dy_est.std():.4f}")
        
        # Compute prior gradients from smoothed DEM (respect physical spacing)
        self._debug_print("\nComputing prior gradients from DEM...")
        # Note: dem_prior[:, 2:] - dem_prior[:, :-2] is forward-backward diff
        # Central difference: (f[i+1] - f[i-1]) / 2
        df_dx_prior = (self.dem_prior[:, 2:] - self.dem_prior[:, :-2]) / (2 * self.dx)  # (H, W-2)
        df_dy_prior = (self.dem_prior[2:, :] - self.dem_prior[:-2, :]) / (2 * self.dy)  # (H-2, W)
        
        self._debug_print(f"  Prior df/dx shape: {df_dx_prior.shape}")
        self._debug_print(f"  Prior df/dy shape: {df_dy_prior.shape}")
        self._debug_print(f"  Prior df/dx range: [{df_dx_prior.min():.4f}, {df_dx_prior.max():.4f}]")
        self._debug_print(f"  Prior df/dy range: [{df_dy_prior.min():.4f}, {df_dy_prior.max():.4f}]")
        self._debug_print(f"  Prior df/dx mean/std: {df_dx_prior.mean():+.4f} / {df_dx_prior.std():.4f}")
        self._debug_print(f"  Prior df/dy mean/std: {df_dy_prior.mean():+.4f} / {df_dy_prior.std():.4f}")
        
        # Crop estimated and prior gradients to interior region
        df_dx_est_center = df_dx_est[1:-1, 1:-1]  # (H-2, W-2)
        df_dy_est_center = df_dy_est[1:-1, 1:-1]  # (H-2, W-2)
        
        df_dx_prior_center = df_dx_prior[1:-1, :]  # (H-2, W-2)
        df_dy_prior_center = df_dy_prior[:, 1:-1]  # (H-2, W-2)
        
        # self._debug_print("\nInterior gradient shapes (should all be same):")
        # self._debug_print(f"  df/dx estimated: {df_dx_est_center.shape}")
        # self._debug_print(f"  df/dy estimated: {df_dy_est_center.shape}")
        # self._debug_print(f"  df/dx prior:     {df_dx_prior_center.shape}")
        # self._debug_print(f"  df/dy prior:     {df_dy_prior_center.shape}")
        
        # Compute gradient residuals
        # self._debug_print("\n--- Gradient Residual Assignment ---")
        # self._debug_print("Coordinate system: x=East, y=North")
        # self._debug_print("G operates on rows (North-South, y-direction)")
        # self._debug_print("G' operates on columns (East-West, x-direction)")
        # self._debug_print("")
        # self._debug_print("From Fernandes & Mosegaard (2022):")
        # self._debug_print("  X provides NS (North-South) slope information")
        # self._debug_print("  Y provides EW (East-West) slope information")
        # self._debug_print("  GM = X  (G operates in NS direction)")
        # self._debug_print("  MG^T = Y  (G^T operates in EW direction)")
        # self._debug_print("")
        # self._debug_print("Therefore in Sylvester equation A @ M + M @ B = C:")
        # self._debug_print("  A = G^T @ G    (NS operator)")
        # self._debug_print("  B = G'^T @ G'  (EW operator, G' ≈ G^T for square DEMs)")
        # self._debug_print("  C = G^T @ Delta_X + Delta_Y @ G'")
        # self._debug_print("  where Delta_X contains NS info (df/dy residuals)")
        # self._debug_print("        Delta_Y contains EW info (df/dx residuals)")
        # self._debug_print("")
        
        # Differences between photometric estimates and prior gradients
        dfdx_residuals = df_dx_est_center - df_dx_prior_center
        dfdy_residuals = df_dy_est_center - df_dy_prior_center

        # Assign gradient residuals per paper's notation:
        # Delta_X contains NS (North-South) information → df/dy residuals
        # Delta_Y contains EW (East-West) information → df/dx residuals
        # This matches: X = NS slopes, Y = EW slopes
        # Delta_X = dfdy_residuals  # NS information (df/dy)
        # Delta_Y = dfdx_residuals  # EW information (df/dx)
        
        
        # Now for the whitening part:
        
        # Flatten gradientresidualer -> (N_int, 2)
        r_vec = np.stack([
            dfdx_residuals.reshape(-1),   # df/dx
            dfdy_residuals.reshape(-1)    # df/dy
        ], axis=1)                        # (N_int, 2)

        # Flatten C_r -> (N_int, 2, 2)
        C_r_flat = C_r.reshape(-1, 2, 2)
        
        
        device = self._select_device()  # Use same device selection logic as before
        dtype  = torch.float32

        C_r_dev = torch.from_numpy(C_r_flat).to(device=device, dtype=dtype)     # (N_int, 2, 2)
        r_dev   = torch.from_numpy(r_vec).to(device=device, dtype=dtype)        # (N_int, 2)

        # Lille jitter for numerisk stabilitet
        eye2 = torch.eye(2, device=device, dtype=dtype).unsqueeze(0)            # (1,2,2)
        C_r_dev = C_r_dev + 1e-10 * eye2

        # Cholesky: C_r = L L^T
        L = torch.linalg.cholesky(C_r_dev)                                      # (N_int, 2, 2)

        # Whiten: L * r_white = r  => r_white = L^{-1} r
        r_dev_vec = r_dev.unsqueeze(-1)                                         # (N_int, 2, 1)
        r_white = torch.linalg.solve(L, r_dev_vec).squeeze(-1)                  # (N_int, 2)

        r_white_np = r_white.cpu().numpy()                                      # (N_int, 2)

        dfdx_residuals_white = r_white_np[:, 0].reshape(H_int, W_int)
        dfdy_residuals_white = r_white_np[:, 1].reshape(H_int, W_int)
        
        Delta_X = dfdy_residuals_white   # NS-info (df/dy), som før
        Delta_Y = dfdx_residuals_white   # EW-info (df/dx), som før    
        
        # self._debug_print("Gradient residual assignment:")
        # self._debug_print(f"  Delta_X = df/dy residuals (NS info): range [{Delta_X.min():.4f}, {Delta_X.max():.4f}] mean/std {Delta_X.mean():+.4f}/{Delta_X.std():.4f}")
        # self._debug_print(f"  Delta_Y = df/dx residuals (EW info): range [{Delta_Y.min():.4f}, {Delta_Y.max():.4f}] mean/std {Delta_Y.mean():+.4f}/{Delta_Y.std():.4f}")
        # self._debug_print("")
        
        # Having whitened the residuals, we can now set sigma_data_eff to 1 for the Sylvester equation, since the whitening has normalized the data term to have unit covariance. The model uncertainty sigma_m_eff can be adjusted relative to this as needed.
        sigma_data_eff = 1.0
        
        eps_sq = sigma_data_eff**2 / self.sigma_M**2

        # check that whitening has worked as intended by verifying that the covariance of the whitened residuals is close to identity
        cov_white = np.cov(r_white_np, rowvar=False)  # (2, 2)
        self._debug_print("\nCovariance of whitened residuals (should be close to identity):")
        self._debug_print(cov_white)

        # Construct Sylvester matrices following Fernandes & Mosegaard (2022):
        # A @ M + M @ B = C
        # where C = G^T @ Delta_X + Delta_Y @ G'
        # Delta_X contains NS (df/dy) info, Delta_Y contains EW (df/dx) info
        # self._debug_print("\nConstructing matrices...")
        A = G.T @ G + eps_sq * np.eye(H_int)
        B = Gp.T @ Gp # + eps_sq * np.eye(W_int) # They do not do this in the paper, but it helps with conditioning and is consistent with their regularization discussion
        C = G.T @ Delta_X + Delta_Y @ Gp
        
        self._debug_print(f"  A shape: {A.shape}, condition number: {np.linalg.cond(A):.2e}")
        self._debug_print(f"  B shape: {B.shape}, condition number: {np.linalg.cond(B):.2e}")
        self._debug_print(f"  C shape: {C.shape}, range: [{C.min():.4f}, {C.max():.4f}]")
        
        self._debug_print("="*70)
        
        
        # ---------------------------------------------------------------------
        # SANITY CHECK: Whitening Diagnostics
        # ---------------------------------------------------------------------

        self._debug_print("\n----------------- WHITENING SANITY CHECK -----------------")

        # 1) Kovarians før global rescale
        cov_before = np.cov(r_white_np, rowvar=False)
        std_before = np.sqrt(np.diag(cov_before))
        mean_before = np.mean(r_white_np, axis=0)

        self._debug_print("Covariance BEFORE global rescale:")
        self._debug_print(cov_before)
        self._debug_print(f"Std BEFORE: {std_before}")
        self._debug_print(f"Mean BEFORE: {mean_before}")

        # 2) Global rescale
        scale = np.mean(np.diag(cov_before))  # typisk ~0.005–0.01 i dit eksempel
        r_white_np_rescaled = r_white_np / np.sqrt(scale)

        # 3) Kovarians efter rescale
        cov_after = np.cov(r_white_np_rescaled, rowvar=False)
        std_after = np.sqrt(np.diag(cov_after))
        mean_after = np.mean(r_white_np_rescaled, axis=0)

        self._debug_print("\nCovariance AFTER global rescale (should be close to identity):")
        self._debug_print(cov_after)
        self._debug_print(f"Std AFTER: {std_after}")
        self._debug_print(f"Mean AFTER: {mean_after}")

        self._debug_print("-----------------------------------------------------------\n")

        
        return A, B, C
    
    def construct_sylvester_matrices_no_whitening(self):
        print("Using unwhitened residuals for Sylvester equation construction")
        H, W = self.dem_prior.shape
        n_int = 1 # Number of pixels to remove from each edge for interior region; should match the gradient computation method (e.g., central differences require n_int=1)
        H_int, W_int = H - 2*n_int, W - 2*n_int  # Interior region (avoid edges)
        
        self._debug_print(f"Full DEM shape: ({H}, {W})")
        self._debug_print(f"Interior region: ({H_int}, {W_int})")
        
        # Build finite difference operators
        G_raw  = build_G(H_int)  # Vertical differences (rows/North-South)
        Gp_raw = build_G_prime(W_int)  # Horizontal differences (columns/East-West)
        # Scale operators by physical spacing so gradients are in meters
        G  = G_raw / self.dy
        Gp = Gp_raw / self.dx

        
        # Estimate normals and albedo from reflectance
        normals_est, albedo_est = self.estimate_normal_and_albedo(n_iters=3)
        
        # Convert normals to gradients
        df_dx_est, df_dy_est = gradients_to_dfdx_dfdy(normals_est)
        

        
        # Compute prior gradients from smoothed DEM (respect physical spacing)
        # Note: dem_prior[:, 2:] - dem_prior[:, :-2] is forward-backward diff
        # Central difference: (f[i+1] - f[i-1]) / 2
        df_dx_prior = (self.dem_prior[:, 2:] - self.dem_prior[:, :-2]) / (2 * self.dx)  # (H, W-2)
        df_dy_prior = (self.dem_prior[2:, :] - self.dem_prior[:-2, :]) / (2 * self.dy)  # (H-2, W)

        
        # Crop estimated and prior gradients to interior region
        df_dx_est_center = df_dx_est[1:-1, 1:-1]  # (H-2, W-2)
        df_dy_est_center = df_dy_est[1:-1, 1:-1]  # (H-2, W-2)
        
        df_dx_prior_center = df_dx_prior[1:-1, :]  # (H-2, W-2)
        df_dy_prior_center = df_dy_prior[:, 1:-1]  # (H-2, W-2)
        
        # self._debug_print("\nInterior gradient shapes (should all be same):")
        # self._debug_print(f"  df/dx estimated: {df_dx_est_center.shape}")
        # self._debug_print(f"  df/dy estimated: {df_dy_est_center.shape}")
        # self._debug_print(f"  df/dx prior:     {df_dx_prior_center.shape}")
        # self._debug_print(f"  df/dy prior:     {df_dy_prior_center.shape}")
        
        # Compute gradient residuals
        # self._debug_print("\n--- Gradient Residual Assignment ---")
        # self._debug_print("Coordinate system: x=East, y=North")
        # self._debug_print("G operates on rows (North-South, y-direction)")
        # self._debug_print("G' operates on columns (East-West, x-direction)")
        # self._debug_print("")
        # self._debug_print("From Fernandes & Mosegaard (2022):")
        # self._debug_print("  X provides NS (North-South) slope information")
        # self._debug_print("  Y provides EW (East-West) slope information")
        # self._debug_print("  GM = X  (G operates in NS direction)")
        # self._debug_print("  MG^T = Y  (G^T operates in EW direction)")
        # self._debug_print("")
        # self._debug_print("Therefore in Sylvester equation A @ M + M @ B = C:")
        # self._debug_print("  A = G^T @ G    (NS operator)")
        # self._debug_print("  B = G'^T @ G'  (EW operator, G' ≈ G^T for square DEMs)")
        # self._debug_print("  C = G^T @ Delta_X + Delta_Y @ G'")
        # self._debug_print("  where Delta_X contains NS info (df/dy residuals)")
        # self._debug_print("        Delta_Y contains EW info (df/dx residuals)")
        # self._debug_print("")
        
        # Differences between photometric estimates and prior gradients
        dfdx_residuals = df_dx_est_center - df_dx_prior_center
        dfdy_residuals = df_dy_est_center - df_dy_prior_center

        # Assign gradient residuals per paper's notation:
        # Delta_X contains NS (North-South) information → df/dy residuals
        # Delta_Y contains EW (East-West) information → df/dx residuals
        # This matches: X = NS slopes, Y = EW slopes
        Delta_X = dfdy_residuals  # NS information (df/dy)
        Delta_Y = dfdx_residuals  # EW information (df/dx)
        
        
        # Having whitened the residuals, we can now set sigma_data_eff to 1 for the Sylvester equation, since the whitening has normalized the data term to have unit covariance. The model uncertainty sigma_m_eff can be adjusted relative to this as needed.
        sigma_data_eff=0.01
        sigma_m_eff=20.0
        
        eps_sq = np.power(sigma_data_eff, 2) / np.power(sigma_m_eff, 2)

        A = G.T @ G + eps_sq * np.eye(H_int)
        B = Gp.T @ Gp # + eps_sq * np.eye(W_int) # They do not do this in the paper, but it helps with conditioning and is consistent with their regularization discussion
        C = G.T @ Delta_X + Delta_Y @ Gp
        
        return A, B, C
    
    def solve_sylvester_eq(self, A, B, C):
        """
        Solve the Sylvester equation A @ X + X @ B = C for X.
        
        Parameters
        ----------
        A : np.ndarray, shape (n, n)
            Left-side matrix.
        B : np.ndarray, shape (m, m)
            Right-side matrix.
        C : np.ndarray, shape (n, m)
            Right-hand side.
        
        Returns
        -------
        X : np.ndarray, shape (n, m)
            Solution matrix.
        """
        from scipy.linalg import solve_sylvester
        
        self._debug_print("\n--- Solving Sylvester Equation ---")
        self._debug_print(f"  A @ X + X @ B = C")
        self._debug_print(f"  A: {A.shape}, B: {B.shape}, C: {C.shape}")
        
        X = solve_sylvester(A, B, C)
        
        self._debug_print(f"  Solution X: {X.shape}")
        self._debug_print(f"  X range: [{X.min():.4f}, {X.max():.4f}]")
        self._debug_print(f"  X mean: {X.mean():.4f}, std: {X.std():.4f}")
        
        # Verify solution quality
        residual = A @ X + X @ B - C
        residual_norm = np.linalg.norm(residual)
        self._debug_print(f"  Residual norm: {residual_norm:.6e}")
        
        return X
    
    def compute_model_update(self, use_iterative_solver=False):
        """
        Compute DEM update using Sylvester equation.
        
        This is the main solver interface. It constructs and solves
        the Sylvester equation to find the DEM correction M.
        
        Parameters
        ----------
        sigma_data : float
            Data uncertainty parameter.
        sigma_m : float
            Model uncertainty parameter.
        
        Returns
        -------
        M_update_full : np.ndarray
            DEM elevation corrections with same shape as DEM (zeros on 1-pixel border).
        """
        if use_iterative_solver:
            A, B, C = self.construct_sylvester_matrices_no_whitening()
            M_update_interior = self.solve_sylvester_eq(A, B, C)
        else:
            A, B, C = self.construct_sylvester_matrices()
            M_update_interior = self.solve_sylvester_eq(A, B, C)

        # Embed interior update into full-sized array
        M_update_full = np.zeros_like(self.dem_prior)
        M_update_full[1:-1, 1:-1] = M_update_interior

        # Cache for potential reuse/inspection
        self.M_update_interior = M_update_interior
        self.M_update_full = M_update_full
        
        # add the update to the current DEM prior to get the new DEM estimate, so that the next iteration of photometric estimation uses the updated DEM
        self.dem_prior += M_update_full
        self.prior_has_been_updated = True  # Mark that the prior has been updated, so cached photometric estimates should not be reused

        return M_update_full
    
    # ==========================================================================
    # VALIDATION METHODS (called when debug=True)
    # ==========================================================================
    
    def _validate_sun_vectors(self, sun_vecs):
        """
        Validate sun direction vectors.
        
        Checks:
        - All vectors are unit length
        - Directions match expected azimuth conventions
        """
        self._debug_print("\n--- Validating Sun Vectors ---")
        
        all_unit = True
        for i, v in enumerate(sun_vecs):
            length = np.linalg.norm(v)
            if not np.isclose(length, 1.0, atol=1e-6):
                self._debug_print(f"  ⚠ Sun {i}: length = {length:.6f} (not unit!)")
                all_unit = False
        
        if all_unit:
            self._debug_print("  ✓ All sun vectors are unit length")
        
        # Verify z-components are positive (sun above horizon)
        if np.all(sun_vecs[:, 2] > 0):
            self._debug_print("  ✓ All sun vectors above horizon (z > 0)")
        else:
            self._debug_print("  ⚠ Some sun vectors below horizon!")
    
    def _validate_normals(self, normals_est):
        """
        Validate estimated normals against ground truth.
        
        Parameters
        ----------
        normals_est : np.ndarray, shape (H, W, 3)
            Estimated normal vectors.
        """
        self._debug_print("\n--- Validating Against Ground Truth ---")
        
        # Compute ground truth normals from DEM
        normals_gt = self.compute_normals_from_dem(self.gt_dem)
        
        # Compute angular errors
        angular_errors = compute_angular_error(normals_est, normals_gt)
        
        self._debug_print(f"Angular error statistics (degrees):")
        self._debug_print(f"  Mean: {angular_errors.mean():.2f}°")
        self._debug_print(f"  Median: {np.median(angular_errors):.2f}°")
        self._debug_print(f"  Std: {angular_errors.std():.2f}°")
        self._debug_print(f"  Min: {angular_errors.min():.2f}°")
        self._debug_print(f"  Max: {angular_errors.max():.2f}°")
        self._debug_print(f"  90th percentile: {np.percentile(angular_errors, 90):.2f}°")
        
        # Check component-wise agreement
        self._debug_print("\nComponent-wise comparison:")
        for i, comp in enumerate(['x', 'y', 'z']):
            diff = normals_est[..., i] - normals_gt[..., i]
            self._debug_print(f"  n{comp} - Mean diff: {diff.mean():+.4f}, RMSE: {np.sqrt(np.mean(diff**2)):.4f}")
        
        return angular_errors
    
    def _validate_gradients(self, df_dx, df_dy):
        """
        Validate gradient field for continuity and reasonable values.
        
        Parameters
        ----------
        df_dx : np.ndarray, shape (H, W)
            Gradient in x-direction.
        df_dy : np.ndarray, shape (H, W)
            Gradient in y-direction.
        """
        self._debug_print("\n--- Gradient Field Validation ---")
        
        # Compute gradient magnitudes
        grad_mag = np.sqrt(df_dx**2 + df_dy**2)
        
        self._debug_print(f"Gradient magnitude statistics:")
        self._debug_print(f"  Mean: {grad_mag.mean():.4f}")
        self._debug_print(f"  Std: {grad_mag.std():.4f}")
        self._debug_print(f"  Max: {grad_mag.max():.4f}")
        
        # Convert to slope angles
        slope_angles = np.rad2deg(np.arctan(grad_mag))
        self._debug_print(f"\nSlope angle statistics (degrees):")
        self._debug_print(f"  Mean: {slope_angles.mean():.2f}°")
        self._debug_print(f"  Median: {np.median(slope_angles):.2f}°")
        self._debug_print(f"  Max: {slope_angles.max():.2f}°")
        
        # Check for discontinuities (large changes between neighbors)
        dx_diff = np.abs(np.diff(df_dx, axis=1))
        dy_diff = np.abs(np.diff(df_dy, axis=0))
        
        self._debug_print(f"\nGradient continuity:")
        self._debug_print(f"  Max df/dx jump: {dx_diff.max():.4f}")
        self._debug_print(f"  Max df/dy jump: {dy_diff.max():.4f}")
    
    def _compare_with_ground_truth(self):
        """
        Full comparison of estimated normals with ground truth DEM.
        
        This method is called automatically when debug=True and provides
        comprehensive validation output.
        """
        if not hasattr(self, 'gt_dem'):
            self._debug_print("No ground truth DEM available for comparison")
            return
        
        self._debug_print("\n" + "="*70)
        self._debug_print("Ground Truth Comparison")
        self._debug_print("="*70)
        
        # Estimate normals
        normals_est, albedo_est = self.estimate_normal_and_albedo(n_iters=3)
        
        # Validate
        angular_errors = self._validate_normals(normals_est)
        
        # Validate gradients
        df_dx_est, df_dy_est = gradients_to_dfdx_dfdy(normals_est)
        self._validate_gradients(df_dx_est, df_dy_est)
        
        self._debug_print("="*70)