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
    def __init__(self, dem, cellsize, x0, y0, y_down=False, debug=False):
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
        self.debug = debug
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
        dem = self.dem

        if self.debug:
            if torch.isnan(dem).any() or torch.isinf(dem).any():
                raise ValueError(
                    f"DEM contains NaN/Inf before normal computation: "
                    f"min={dem.min().item():.6e}, max={dem.max().item():.6e}"
                )

        # torch.gradient returns derivatives along each dimension:
        # dim 0 -> y/rows, dim 1 -> x/cols
        dz_dy, dz_dx = torch.gradient(dem, spacing=(self.cellsize, self.cellsize))

        if self.y_down:
            dz_dy = -dz_dy

        nx = -dz_dx
        ny = -dz_dy
        nz = torch.ones_like(dem, device=self.device)

        norm = torch.sqrt(nx * nx + ny * ny + nz * nz).clamp_min(1e-12)

        self.nx = nx / norm
        self.ny = ny / norm
        self.nz = nz / norm

        if self.debug:
            for name, tensor in [
                ("dz_dx", dz_dx),
                ("dz_dy", dz_dy),
                ("nx", self.nx),
                ("ny", self.ny),
                ("nz", self.nz),
            ]:
                if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                    raise ValueError(
                        f"{name} contains NaN/Inf after normal computation: "
                        f"min={tensor.min().item():.6e}, max={tensor.max().item():.6e}"
                    )

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
