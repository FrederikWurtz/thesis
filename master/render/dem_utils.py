import numpy as np
import torch

class DEM:
    """
    Digital Elevation Model (DEM) utility class for handling raster elevation data and computing surface properties.

    Attributes:
        dem (np.ndarray): 2D array representing elevation values.
        cellsize (float): Size of each grid cell (spatial resolution).
        x0 (float): X-coordinate of the origin (upper-left corner).
        y0 (float): Y-coordinate of the origin (upper-left corner).
        height (int): Number of rows in the DEM.
        width (int): Number of columns in the DEM.
        extent (list): Spatial extent of the DEM in the format [xmin, xmax, ymin, ymax].
        nx (np.ndarray): X-component of the surface normal vectors at each grid cell.
        ny (np.ndarray): Y-component of the surface normal vectors at each grid cell.
        nz (np.ndarray): Z-component of the surface normal vectors at each grid cell.
        world_points (np.ndarray): Array of 3D world coordinates for each grid cell, shape (height*width, 3).

    Methods:
        __init__(dem, cellsize, x0, y0):
            Initializes the DEM object, computes surface normals, and builds world coordinates.

        _compute_surface_normals(y_down=False):
            Computes the surface normal vectors for each grid cell using the gradient of the DEM.
            Args:
                y_down (bool): If True, inverts the y-gradient to account for image coordinate systems
                               where y increases downward. Default is False (y increases upward).

        _build_world_points():
            Constructs a (height*width, 3) array of 3D world coordinates (x, y, z) for each grid cell.
    """
    def __init__(self, dem, cellsize, x0, y0, y_down=False):
        """
        Initializes the DEM object with elevation data and spatial parameters.
        Computes surface normals and builds world coordinates for each grid cell.

        Args:
            dem (np.ndarray or torch.Tensor): 2D array of elevation values.
            cellsize (float): Spatial resolution of each grid cell.
            x0 (float): X-coordinate of the origin (upper-left corner).
            y0 (float): Y-coordinate of the origin (upper-left corner).
        """
        self.y_down = y_down
        # Convert to torch tensor if numpy array
        if isinstance(dem, np.ndarray):
            self.dem = torch.from_numpy(dem).float()
        else:
            self.dem = dem.float()
        self.cellsize = cellsize
        self.device = self.dem.device
        self.x0 = x0
        self.y0 = y0
        self.height, self.width = self.dem.shape
        # Calculate the spatial extent of the DEM
        self.extent = [x0, x0 + self.width * cellsize, y0, y0 + self.height * cellsize]
        # Compute surface normals for each grid cell
        self._compute_surface_normals()
        # Build 3D world coordinates for each grid cell
        self._build_world_points()

    def _compute_surface_normals(self):
        """
        Computes the surface normal vectors for each grid cell using the DEM gradients.
        Uses self.y_down to determine if y-gradient should be inverted.
        """
        # Compute gradients in y and x directions using PyTorch
        # Add batch and channel dimensions for gradient computation
        dem_4d = self.dem.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        
        # Sobel-like gradients for y and x
        # Create gradient kernels
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0) / (8 * self.cellsize)
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0) / (8 * self.cellsize)
        
        # Apply convolution for gradients
        dz_dy_raw = torch.nn.functional.conv2d(dem_4d, ky, padding=1).squeeze()
        dz_dx = torch.nn.functional.conv2d(dem_4d, kx, padding=1).squeeze()
        
        # Optionally invert y-gradient for image coordinate systems
        dz_dy = -dz_dy_raw if self.y_down else dz_dy_raw
        # Surface normal components
        nx = -dz_dx
        ny = -dz_dy
        nz = torch.ones_like(self.dem, device=self.device)
        # Normalize the normal vectors
        norm = torch.sqrt(nx*nx + ny*ny + nz*nz) + 1e-12
        self.nx = nx / norm
        self.ny = ny / norm
        self.nz = nz / norm

    def _build_world_points(self):
        """
        Constructs a (height*width, 3) array of 3D world coordinates (x, y, z) for each grid cell.
        """
        # Generate x and y coordinates for each grid cell
        xs = self.x0 + torch.arange(self.width, dtype=torch.float32, device=self.device) * self.cellsize
        ys = self.y0 + torch.arange(self.height, dtype=torch.float32, device=self.device) * self.cellsize
        xx, yy = torch.meshgrid(xs, ys, indexing='xy')
        zz = self.dem
        # Stack x, y, z into a single array and reshape to (N, 3)
        self.world_points = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
