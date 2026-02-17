"""
Utility functions for FM (Fernandes-Mosegaard) Solver.

This module contains helper functions for:
- Coordinate system transformations (azimuth/elevation to unit vectors)
- Normal vector computations from DEM gradients
- Finite difference matrix operators (G, G')
- Prior DEM processing (downsampling, smoothing)
- Gradient field validation

COORDINATE SYSTEM CONVENTION:
- x-axis: Easting (positive East)
- y-axis: Northing (positive North)
- z-axis: Up (positive upward)
- Right-handed coordinate system
- Array indexing: dem[row, col] = dem[y_index, x_index]
- Azimuth: 0° = North, 90° = East, 180° = South, 270° = West
- Elevation: 0° = horizon, 90° = zenith

NORMAL VECTOR CONVENTION:
- Surface normals point upward (positive z-component)
- n = [-dz/dx, -dz/dy, 1] normalized
- This follows from cross product of surface tangent vectors
"""

import numpy as np
from scipy.ndimage import gaussian_filter, zoom
import torch


# ==============================================================================
# COORDINATE UTILITIES
# ==============================================================================

def unit_vec_from_az_el(az_deg, el_deg):
    """
    Compute unit vector(s) from azimuth and elevation angles.
    
    Convention:
    - Azimuth: 0° = North (+y), 90° = East (+x), 180° = South (-y), 270° = West (-x)
    - Elevation: 0° = horizon, 90° = zenith (+z)
    
    Spherical to Cartesian transformation:
    - x = sin(az) * cos(el)  [Easting component]
    - y = cos(az) * cos(el)  [Northing component]
    - z = sin(el)            [Up component]
    
    Parameters
    ----------
    az_deg : float or np.ndarray
        Azimuth angle(s) in degrees.
    el_deg : float or np.ndarray
        Elevation angle(s) in degrees.
    
    Returns
    -------
    np.ndarray
        Unit vector(s) with shape (3,) or (3, N).
        Components are [x_east, y_north, z_up].
    
    Examples
    --------
    >>> unit_vec_from_az_el(0, 45)  # North at 45° elevation
    array([0., 0.707, 0.707])
    
    >>> unit_vec_from_az_el(90, 45)  # East at 45° elevation
    array([0.707, 0., 0.707])
    """
    az = np.deg2rad(az_deg)
    el = np.deg2rad(el_deg)
    
    x = np.sin(az) * np.cos(el)  # Easting
    y = np.cos(az) * np.cos(el)  # Northing
    z = np.sin(el)                # Up
    
    v = np.stack([x, y, z], axis=0)
    
    # Normalize to unit length
    norm = np.linalg.norm(v, axis=0, keepdims=True)
    return v / norm

# def unit_vec_from_az_el(az_deg, el_deg, device):
#         """
#         Computes a unit vector from azimuth and elevation angles (in degrees).

#         Parameters:
#             az_deg (float or torch.Tensor): Azimuth angle(s) in degrees.
#             el_deg (float or torch.Tensor): Elevation angle(s) in degrees.

#         Returns:
#             torch.Tensor: Unit vector(s) corresponding to the given azimuth and elevation.
#                             Shape: (3,) if scalars, (3, N) if arrays.
#         """
#         # Convert to tensors if needed
#         if not isinstance(az_deg, torch.Tensor):
#             az_deg = torch.tensor(az_deg, dtype=torch.float32, device=device)
#         if not isinstance(el_deg, torch.Tensor):
#             el_deg = torch.tensor(el_deg, dtype=torch.float32, device=device)
            
#         az = torch.deg2rad(az_deg)  # Convert azimuth to radians
#         el = torch.deg2rad(el_deg)  # Convert elevation to radians
        
#         # Spherical to Cartesian conversion
#         x = torch.sin(az) * torch.cos(el)
#         y = torch.cos(az) * torch.cos(el)
#         z = torch.sin(el)
#         v = torch.stack([x, y, z], dim=0)
#         # Normalize to unit length
#         vec_np = (v / torch.norm(v, dim=0)).cpu().numpy()
#         return vec_np


def compute_normal_from_gradients(dz_dx, dz_dy):
    """
    Compute surface normal from elevation gradients.
    
    The normal vector is computed as:
        n = [-dz/dx, -dz/dy, 1] / ||[-dz/dx, -dz/dy, 1]||
    
    This convention ensures:
    - Normal points upward (positive z-component)
    - Normal points "away" from the downslope direction
    - Follows right-handed cross product: tangent_x × tangent_y
    
    Parameters
    ----------
    dz_dx : float or np.ndarray
        Elevation gradient in x-direction (Easting).
    dz_dy : float or np.ndarray
        Elevation gradient in y-direction (Northing).
    
    Returns
    -------
    np.ndarray
        Unit normal vector(s). Shape matches input shapes with added dimension 3.
        Components are [nx, ny, nz].
    
    Notes
    -----
    For a surface z = f(x, y), the tangent vectors are:
    - t_x = [1, 0, dz/dx]
    - t_y = [0, 1, dz/dy]
    
    The normal is: t_x × t_y = [-dz/dx, -dz/dy, 1]
    """
    # Build unnormalized normal
    nx = -dz_dx
    ny = -dz_dy
    nz = np.ones_like(dz_dx)
    
    # Stack and normalize
    if np.ndim(dz_dx) == 0:  # Scalar case
        n = np.array([nx, ny, nz])
    else:  # Array case
        n = np.stack([nx, ny, nz], axis=-1)
    
    # Normalize to unit length
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    return n / norm


def gradients_to_dfdx_dfdy(normals):
    """
    Convert normal vectors back to elevation gradients.
    
    Inverse of compute_normal_from_gradients. Given unit normals
    n = [-dz/dx, -dz/dy, 1] / ||...||, recover dz/dx and dz/dy.
    
    Parameters
    ----------
    normals : np.ndarray
        Normal vectors with shape (..., 3) where last dimension is [nx, ny, nz].
    
    Returns
    -------
    dz_dx : np.ndarray
        Gradient in x-direction (Easting).
    dz_dy : np.ndarray
        Gradient in y-direction (Northing).
    
    Notes
    -----
    From n = [-dz/dx, -dz/dy, 1] / sqrt((dz/dx)^2 + (dz/dy)^2 + 1):
        nx / nz = -dz/dx / 1  =>  dz/dx = -nx / nz
        ny / nz = -dz/dy / 1  =>  dz/dy = -ny / nz
    """
    dz_dx = -normals[..., 0] / normals[..., 2]
    dz_dy = -normals[..., 1] / normals[..., 2]
    return dz_dx, dz_dy


# ==============================================================================
# MATRIX CONSTRUCTORS (Finite Difference Operators)
# ==============================================================================

def build_G(n):
    """
    Build vertical (row-wise) first-difference operator.
    
    This operator approximates the derivative in the vertical direction
    of a 2D field. For a field f of height n, G @ f computes:
        (G @ f)[i] ≈ f[i+1] - f[i]
    
    Physical meaning: Vertical gradient operator for DEM elevation fields.
    Used in Sylvester equation for height (North-South) differences.
    
    Parameters
    ----------
    n : int
        Size of the operator (number of rows in the field).
    
    Returns
    -------
    np.ndarray
        Shape (n, n) sparse matrix with -1 on diagonal, +1 on superdiagonal.
        Last row is zeros (boundary condition).
    
    Examples
    --------
    >>> G = build_G(3)
    >>> G
    array([[-1.,  1.,  0.],
           [ 0., -1.,  1.],
           [ 0.,  0.,  0.]])
    """
    G = np.zeros((n, n))
    for i in range(n - 1):
        G[i, i]   = -1.0
        G[i, i+1] =  1.0
    # Last row is zero (boundary condition)
    G[n-1, n-1] = 0.0
    return G


def build_G_transpose(n):
    """
    Build transpose of vertical first-difference operator.
    
    Parameters
    ----------
    n : int
        Size of the operator.
    
    Returns
    -------
    np.ndarray
        Shape (n, n) transpose of G.
    """
    return build_G(n).T


def build_G_prime(m):
    """
    Build horizontal (column-wise) first-difference operator.
    
    This operator approximates the derivative in the horizontal direction
    of a 2D field. For a field f of width m, f @ G' computes:
        (f @ G')[j] ≈ f[j+1] - f[j]
    
    Physical meaning: Horizontal gradient operator for DEM elevation fields.
    Used in Sylvester equation for width (East-West) differences.
    
    Parameters
    ----------
    m : int
        Size of the operator (number of columns in the field).
    
    Returns
    -------
    np.ndarray
        Shape (m, m) sparse matrix with -1 on diagonal, +1 on superdiagonal.
        Last row is zeros (boundary condition).
    
    Notes
    -----
    Naming: "G prime" (G') distinguishes this from G. Both have the same
    structure but operate on different dimensions (height vs width).
    """
    Gp = np.zeros((m, m))
    for j in range(m - 1):
        Gp[j, j]   = -1.0
        Gp[j, j+1] =  1.0
    # Last row is zero (boundary condition)
    Gp[m-1, m-1] = 0.0
    return Gp


def build_G_prime_transpose(m):
    """
    Build transpose of horizontal first-difference operator.
    
    Parameters
    ----------
    m : int
        Size of the operator.
    
    Returns
    -------
    np.ndarray
        Shape (m, m) transpose of G'.
    """
    return build_G_prime(m).T


# ==============================================================================
# PRIOR DEM PROCESSING
# ==============================================================================

def downsample_and_smooth_dem(dem, scale_down_factor=40, smooth_sigma=None):
    """
    Create smoothed, low-resolution prior DEM following Fernandes & Mosegaard (2022).
    
    Process (matches Fig. 1C in the paper):
    1. Downsample DEM by scale_down_factor
    2. Upsample back to original resolution (introduces smoothing)
    3. Apply Gaussian smoothing with sigma ~ scale_down_factor / 2
    
    This creates a "blurred" version of the DEM that preserves large-scale
    topography while removing fine-scale features. Used as prior for Bayesian
    estimation.
    
    Parameters
    ----------
    dem : np.ndarray
        Original DEM with shape (H, W).
    scale_down_factor : int, optional
        Downsampling factor. Default is 40.
    smooth_sigma : float, optional
        Gaussian smoothing sigma. If None, uses scale_down_factor / 2.
    
    Returns
    -------
    dem_low : np.ndarray
        Downsampled DEM (not upsampled).
    dem_prior : np.ndarray
        Smoothed prior DEM with original shape (H, W).
    
    Notes
    -----
    The smoothing removes high-frequency content that would otherwise
    over-constrain the photometric normal estimation.
    """
    H, W = dem.shape
    
    # 1. Downsample
    downsample_factors = (1.0 / scale_down_factor, 1.0 / scale_down_factor)
    dem_low = zoom(dem, downsample_factors, order=1)
    
    # 2. Upsample back to original resolution
    upsample_factors = (H / dem_low.shape[0], W / dem_low.shape[1])
    dem_up = zoom(dem_low, upsample_factors, order=1)
    
    # 3. Smooth
    if smooth_sigma is None:
        smooth_sigma = scale_down_factor / 2.0
    
    dem_prior = gaussian_filter(dem_up, sigma=smooth_sigma)
    
    return dem_low, dem_prior


# ==============================================================================
# GRADIENT FIELD VALIDATION
# ==============================================================================

def validate_normal_field(normals, tol=1e-4):
    """
    Validate that normal vectors are unit-length with positive z-component.
    
    Parameters
    ----------
    normals : np.ndarray
        Normal vectors with shape (..., 3).
    tol : float, optional
        Tolerance for unit length check. Default 1e-4.
    
    Returns
    -------
    dict
        Validation results with keys:
        - 'all_unit': bool, True if all normals are unit length
        - 'all_upward': bool, True if all z-components are positive
        - 'mean_length': float, mean length of normal vectors
        - 'std_length': float, standard deviation of lengths
        - 'min_z': float, minimum z-component
        - 'num_invalid': int, number of non-unit normals
    
    Examples
    --------
    >>> normals = np.array([[0, 0, 1], [0.707, 0, 0.707]])
    >>> result = validate_normal_field(normals)
    >>> result['all_unit']
    True
    >>> result['all_upward']
    True
    """
    lengths = np.linalg.norm(normals, axis=-1)
    z_components = normals[..., 2]
    
    all_unit = np.allclose(lengths, 1.0, atol=tol)
    all_upward = np.all(z_components > 0)
    num_invalid = np.sum(np.abs(lengths - 1.0) > tol)
    
    return {
        'all_unit': all_unit,
        'all_upward': all_upward,
        'mean_length': np.mean(lengths),
        'std_length': np.std(lengths),
        'min_z': np.min(z_components),
        'max_z': np.max(z_components),
        'num_invalid': num_invalid
    }


def compute_angular_error(normals_est, normals_gt):
    """
    Compute angular error between estimated and ground truth normals.
    
    Parameters
    ----------
    normals_est : np.ndarray
        Estimated normal vectors with shape (..., 3).
    normals_gt : np.ndarray
        Ground truth normal vectors with shape (..., 3).
    
    Returns
    -------
    angles_deg : np.ndarray
        Angular errors in degrees. Shape matches input shape without last dimension.
    
    Notes
    -----
    Angular error is computed as:
        theta = arccos(n_est · n_gt)
    Handles numerical issues where dot product slightly exceeds [-1, 1].
    """
    # Compute dot products
    dots = np.sum(normals_est * normals_gt, axis=-1)
    
    # Clip to valid range for arccos
    dots = np.clip(dots, -1.0, 1.0)
    
    # Compute angles
    angles_rad = np.arccos(dots)
    angles_deg = np.rad2deg(angles_rad)
    
    return angles_deg


def remove_outer_n_pixels(field, n, debug = True):
    """
    Remove outer n pixels from all data fields.
    
    Edge pixels have less accurate normals due to boundary effects
    in gradient computation. Remove them before solving.
    
    Parameters
    ----------
    n : int
        Number of pixels to remove from each edge.
    """
    H, W = field.shape
    H_new = H - n
    W_new = W - n
    
    
    if debug:
        print(f"\n--- Removing Outer {n} Pixels ---")
        print(f"Original shape: ({H}, {W})")
        print(f"New shape: ({H_new}, {W_new})")
    
    # Crop all data fields
    new_field = field[n:H-n, n:W-n]
    
    return new_field