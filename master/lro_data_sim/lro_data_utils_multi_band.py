import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift
from scipy.fft import ifftshift
from tqdm import trange
import torch
from joblib import Parallel, delayed
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject, transform_bounds
import rasterio

import matplotlib.gridspec as gridspec

def plot_dem_and_hillshade(dem_array, hillshade_array, metadata, center_lat_deg, center_lon_deg, 
                           proj_type="stere", save_path=None, dpi=300, box_radius_m=None):
    """
    Plot DEM and hillshade side by side.
    
    Parameters
    ----------
    dem_array : np.ndarray
        DEM elevation data
    hillshade_array : np.ndarray
        Hillshade image (uint8)
    metadata : dict
        Metadata from extract_local_dem_subset
    center_lat_deg : float
        Center latitude in degrees
    center_lon_deg : float
        Center longitude in degrees
    proj_type : str, optional
        Projection type for title (default: "stere")
    save_path : str, optional
        Path to save figure. If None, doesn't save.
    dpi : int, optional
        DPI for saved figure (default: 300)
    """
    # Mask nodata values
    z_masked = np.ma.masked_equal(dem_array, metadata['nodata'])
    
    # Desired subplot sizes (width, height) in inches
    subplot_sizes = [(5.5, 5.5), (6, 6)]  # (hillshade, DEM)
    
    # Calculate total figure size
    total_width = subplot_sizes[0][0] + subplot_sizes[1][0]
    total_height = max(subplot_sizes[0][1], subplot_sizes[1][1])
    
    # Set width ratios based on desired subplot widths
    width_ratios = [subplot_sizes[0][0], subplot_sizes[1][0]]
    
    fig = plt.figure(figsize=(total_width, total_height))
    gs = gridspec.GridSpec(1, 2, width_ratios=width_ratios)
    
    if box_radius_m is not None:
        fig.suptitle(f"Local DEM Analysis - {2*box_radius_m/1000:.0f} km × {2*box_radius_m/1000:.0f} km", 
                     fontsize=14, fontweight='bold', y=0.98)

    # Hillshade plot
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(hillshade_array, cmap="gray", origin="upper")
    ax1.set_title(f"Local hillshade ({proj_type}, center {center_lat_deg}°N, {center_lon_deg}°E)", pad=15)
    ax1.set_xlabel('X (pixels)')
    ax1.set_ylabel('Y (pixels)')
    
    # DEM plot
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(z_masked, cmap='terrain', origin='upper')
    ax2.set_title(f"Local DEM ({proj_type}, center {center_lat_deg}°N, {center_lon_deg}°E)", pad=15)
    ax2.set_xlabel('X (pixels)')
    ax2.set_ylabel('Y (pixels)')
    cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label('Elevation (m)')
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=dpi)
        print(f"Saved figure to: {save_path}")
    



# ----------------------------
# CONSTANTS
# ----------------------------
MOON_MEAN_RADIUS_M = 1737400.0  # meters

# ----------------------------
# BUILD LOCAL CRS
# ----------------------------
def build_local_crs(lat0_deg, lon0_deg, proj="stere", moon_radius_m=MOON_MEAN_RADIUS_M):
    """Return Moon local CRS (stere or aeqd) centered at (lat0, lon0)."""
    if proj == "stere":
        proj4 = (
            f"+proj=stere +lat_0={lat0_deg} +lon_0={lon0_deg} "
            f"+k=1 +x_0=0 +y_0=0 +R={moon_radius_m} +units=m +no_defs"
        )
    elif proj == "aeqd":
        proj4 = (
            f"+proj=aeqd +lat_0={lat0_deg} +lon_0={lon0_deg} "
            f"+x_0=0 +y_0=0 +R={moon_radius_m} +units=m +no_defs"
        )
    else:
        raise ValueError("proj must be 'stere' or 'aeqd'")
    return CRS.from_proj4(proj4)



def extract_local_subset_all_bands(
    dem_path,
    center_lat_deg,
    center_lon_deg,
    box_radius_m,
    res_m=100.0,
    local_proj_type="stere",
    out_geotiff=None,
    moon_radius_m=1737400.0,  # meters
    verbose=False
):
    """
    Extract and reproject a local subset of a lunar DEM.
    
    Parameters
    ----------
    dem_path : str
        Path to the source DEM GeoTIFF file
    center_lat_deg : float
        Latitude of the subset center (degrees, planetocentric)
    center_lon_deg : float
        Longitude of the subset center (degrees, east-positive)
    box_radius_m : float
        Half-size of the square output region in meters
        (output will be 2*box_radius_m on each side)
    res_m : float, optional
        Output resolution in meters per pixel (default: 100.0)
    local_proj_type : str, optional
        Local projection type: "stere" (shape-preserving) or 
        "aeqd" (distance-preserving) (default: "stere")
    out_geotiff : str, optional
        Path to save output GeoTIFF. If None, doesn't save to file.
    moon_radius_m : float, optional
        Moon's mean radius in meters (default: 1737400.0)
    
    Returns
    -------
    all_bands_array : np.ndarray
        Reprojected data as 3D array (bands, height, width)
    metadata : dict
        Dictionary containing:
        - 'nodata': nodata value (per band if different, else single value)
        - 'transform': rasterio affine transform
        - 'crs': destination CRS
        - 'width': output width in pixels
        - 'height': output height in pixels
        - 'resolution_m': resolution in meters per pixel
        - 'dtype': dtype (per band if different, else single value)
        - 'band_count': number of bands
    """

    # Build local CRS
    dst_crs = build_local_crs(center_lat_deg, center_lon_deg, 
                               proj=local_proj_type, moon_radius_m=moon_radius_m)

    with rasterio.open(dem_path) as src:
        src_crs = src.crs
        src_transform = src.transform
        src_nodata = src.nodatavals  # tuple, one per band
        dtype = src.dtypes[0]
        band_count = src.count

        # Print info
        if verbose:
            print(f"Source CRS: {src_crs.to_wkt()[:160]}...")
            print(f"Pixel size (m): {src_transform.a}, {abs(src_transform.e)}")
            print(f"Band count: {band_count}")

        # Define output grid
        half = box_radius_m
        width = int(np.ceil((2 * half) / res_m))
        height = int(np.ceil((2 * half) / res_m))
        dst_transform = from_origin(-half, +half, res_m, res_m)

        if verbose:
            print(f"Output grid: {width} x {height} pixels")
            print(f"Output area: {2*half/1000:.1f} x {2*half/1000:.1f} km")

        # Pre-crop source DEM for efficiency
        dst_bounds = (-half, -half, +half, +half)
        src_bounds = transform_bounds(dst_crs, src_crs, *dst_bounds, densify_pts=21)

        window = rasterio.windows.from_bounds(*src_bounds, transform=src_transform)
        window = window.round_offsets(pixel_precision=0)
        window = window.intersection(rasterio.windows.Window.from_slices(
            (0, src.height), (0, src.width)
        ))
        # Prepare output array for all bands
        all_bands_array = np.empty((band_count, height, width), dtype=dtype)

        # Compute the transform for the cropped source window once
        subset_transform = rasterio.windows.transform(window, src_transform)

        # --- Multi-band case: read & reproject all bands in one go ---

        # Read all bands in one call -> shape (band_count, h_src, w_src)
        src_subset = src.read(window=window)

        # Determine nodata handling
        # src_nodata is a tuple of length band_count
        # Check if all bands share the same nodata value
        if all(val == src_nodata[0] for val in src_nodata):
            # Common nodata
            band_nodata = src_nodata[0]
            dst_array = np.full(
                (band_count, height, width),
                band_nodata if band_nodata is not None else 0,
                dtype=dtype,
            )

            reproject(
                source=src_subset,
                destination=dst_array,
                src_transform=subset_transform,
                src_crs=src_crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear,
                src_nodata=band_nodata,
                dst_nodata=band_nodata,
            )

        else:
            # Different nodata per band (less common, but handle it)
            # We rely on rasterio/GDAL supporting sequences for src_nodata/dst_nodata.
            band_nodata = [
                val if val is not None else 0 for val in src_nodata
            ]

            dst_array = np.empty(
                (band_count, height, width),
                dtype=dtype,
            )

            # Initialize each band separately with its own nodata
            for b in range(band_count):
                dst_array[b, :, :] = band_nodata[b]

            reproject(
                source=src_subset,
                destination=dst_array,
                src_transform=subset_transform,
                src_crs=src_crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear,
                src_nodata=band_nodata,
                dst_nodata=band_nodata,
            )

        # Copy into final output array
        all_bands_array[:, :, :] = dst_array

        # Save to file if requested
        if out_geotiff is not None:
            profile = src.profile.copy()
            profile.update({
                "driver": "GTiff",
                "height": height,
                "width": width,
                "count": band_count,
                "crs": dst_crs,
                "transform": dst_transform,
                "dtype": dtype,
                "nodata": src_nodata[0] if all(x == src_nodata[0] for x in src_nodata) else None,
                "compress": "deflate",
                "predictor": 2
            })
            with rasterio.open(out_geotiff, "w", **profile) as dst_ds:
                dst_ds.write(all_bands_array)
            if verbose:
                print(f"Saved to: {out_geotiff}")

        # Prepare metadata
        metadata = {
            'nodata': src_nodata if band_count > 1 else src_nodata[0],
            'transform': dst_transform,
            'crs': dst_crs,
            'width': width,
            'height': height,
            'resolution_m': res_m,
            'dtype': dtype,
            'band_count': band_count
        }

        return all_bands_array, metadata


# ----------------------------
# HILLSHADE FUNCTION
# ----------------------------
def hillshade(z, res, azimuth_deg=315.0, altitude_deg=45.0, z_factor=1.0, nodata=None):
    """Generate hillshade from elevation data."""
    zf = z.astype(np.float32)
    if nodata is not None:
        zf = np.where(zf == nodata, np.nan, zf)
    
    az = np.deg2rad(azimuth_deg)
    alt = np.deg2rad(altitude_deg)
    gy, gx = np.gradient(zf * z_factor, res, res)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gy, gx)
    hs = (np.cos(alt) * np.cos(slope) +
          np.sin(alt) * np.sin(slope) * np.cos(az - aspect))
    hs = np.clip(hs, 0, 1)
    
    # Convert to uint8, handling NaNs to avoid warning
    hs_scaled = hs * 255
    hs_uint8 = np.zeros_like(hs_scaled, dtype=np.uint8)
    valid_mask = ~np.isnan(hs_scaled)
    hs_uint8[valid_mask] = hs_scaled[valid_mask].astype(np.uint8)
    
    return hs_uint8

def radial_profile(autocorr, r_max=None, bin_width=1.0, stat='mean'):
    """
    Compute a radial (circular) average of a 2D autocorrelation map.

    Parameters
    ----------
    autocorr : 2D array
        Autocorrelation map with the zero-lag at the center.
    r_max : float or None
        Maximum radius to compute. If None, uses the largest distance to a corner.
    bin_width : float
        Width of radial bins in pixels. Use <1 for subpixel sampling (slower).
    stat : {'mean', 'median'}
        Aggregation statistic for each annulus.

    Returns
    -------
    r_centers : 1D array
        Radii (in pixels) at the center of each bin.
    profile : 1D array
        Radial-averaged autocorrelation values for each radius bin.
    counts : 1D array
        Number of pixels that contributed to each bin.
    """

    if autocorr.ndim != 2:
        raise ValueError("autocorr must be 2D")

    ny, nx = autocorr.shape
    cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0  # center coordinates (works for odd/even dims)

    # coordinate grids
    y = np.arange(ny) - cy
    x = np.arange(nx) - cx
    X, Y = np.meshgrid(x, y)
    R = np.hypot(X, Y)  # distance from center

    if r_max is None:
        r_max = R.max()

    # define bins
    nbins = int(np.ceil(r_max / bin_width))
    bin_edges = np.linspace(0, nbins * bin_width, nbins + 1)
    r_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # flatten arrays and mask NaNs
    Rf = R.ravel()
    Af = autocorr.ravel()
    valid = ~np.isnan(Af)
    Rf = Rf[valid]
    Af = Af[valid]

    # assign bins
    inds = np.digitize(Rf, bin_edges) - 1  # bin index in [0, nbins-1]
    # remove out-of-range indices (shouldn't happen but safe)
    mask = (inds >= 0) & (inds < nbins)
    inds = inds[mask]
    Af = Af[mask]

    # aggregate
    counts = np.bincount(inds, minlength=nbins)
    if stat == 'mean':
        sums = np.bincount(inds, weights=Af, minlength=nbins)
        # avoid division by zero
        with np.errstate(invalid='ignore', divide='ignore'):
            profile = sums / counts
    elif stat == 'median':
        # median requires grouping — do manual loop (still vectorized-ish)
        profile = np.empty(nbins, dtype=float)
        profile[:] = np.nan
        for b in range(nbins):
            sel = inds == b
            if np.any(sel):
                profile[b] = np.nanmedian(Af[sel])
        # counts already set
    else:
        raise ValueError("stat must be 'mean' or 'median'")

    # mask bins with zero counts
    profile[counts == 0] = np.nan

    return r_centers, profile, counts

def detrend_2d(data, method='linear'):
    """
    Remove large-scale trends from a 2D array (e.g., DEM).
    Parameters
    ----------
    data : 2D numpy array
        Input array (can contain NaNs).
    method : 'linear' or 'planar'
        'linear': fit and subtract a least-squares plane (ax + by + c).
        'planar': same as 'linear'.
    Returns
    -------
    detrended : 2D numpy array
        Array with fitted plane subtracted (NaNs preserved).
    """
    mask = ~np.isnan(data)
    yy, xx = np.indices(data.shape)
    X = xx[mask].ravel()
    Y = yy[mask].ravel()
    Z = data[mask].ravel()
    if method not in ['linear', 'quadratic', 'cubic', 'quartic']:
        raise ValueError("method must be 'linear', 'quadratic', 'cubic', or 'quartic'")
    
    if method == 'linear':
        # Fit plane: Z = a*X + b*Y + c
        A = np.c_[X, Y, np.ones_like(X)]
        coeffs, _, _, _ = np.linalg.lstsq(A, Z, rcond=None)
        plane = coeffs[0] * xx + coeffs[1] * yy + coeffs[2]
        detrended = data - plane

    if method == 'quadratic':
        # Fit quadratic surface: Z = a*X^2 + b*Y^2 + c*X*Y + d*X + e*Y + f
        A = np.c_[X**2, Y**2, X*Y, X, Y, np.ones_like(X)]
        coeffs, _, _, _ = np.linalg.lstsq(A, Z, rcond=None)
        surface = (coeffs[0] * xx**2 + coeffs[1] * yy**2 + coeffs[2] * xx * yy +
                   coeffs[3] * xx + coeffs[4] * yy + coeffs[5])
        detrended = data - surface
    
    if method == 'cubic':
        # Fit cubic surface
        A = np.c_[X**3, Y**3, X**2*Y, X*Y**2, X**2, Y**2, X*Y, X, Y, np.ones_like(X)]
        coeffs, _, _, _ = np.linalg.lstsq(A, Z, rcond=None)
        surface = (coeffs[0] * xx**3 + coeffs[1] * yy**3 + coeffs[2] * xx**2 * yy +
                   coeffs[3] * xx * yy**2 + coeffs[4] * xx**2 + coeffs[5] * yy**2 +
                   coeffs[6] * xx * yy + coeffs[7] * xx + coeffs[8] * yy + coeffs[9])
        detrended = data - surface
    if method == 'quartic':
        # Fit quartic surface
        A = np.c_[X**4, Y**4, X**3*Y, X*Y**3, X**2*Y**2, X**3, Y**3, X**2*Y, X*Y**2,
                  X**2, Y**2, X*Y, X, Y, np.ones_like(X)]
        coeffs, _, _, _ = np.linalg.lstsq(A, Z, rcond=None)
        surface = (coeffs[0] * xx**4 + coeffs[1] * yy**4 + coeffs[2] * xx**3 * yy +
                   coeffs[3] * xx * yy**3 + coeffs[4] * xx**2 * yy**2 +
                   coeffs[5] * xx**3 + coeffs[6] * yy**3 + coeffs[7] * xx**2 * yy +
                   coeffs[8] * xx * yy**2 + coeffs[9] * xx**2 + coeffs[10] * yy**2 +
                   coeffs[11] * xx * yy + coeffs[12] * xx + coeffs[13] * yy + coeffs[14])
        detrended = data - surface

    return detrended