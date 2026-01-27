import matplotlib.pyplot as plt
import os
import os
import re
import torch
import random
import numpy as np
import subprocess
import shutil
import argparse
import glob
import rasterio

from torch.utils.data import Dataset, DataLoader

from master.models.unet import UNet
from master.train.trainer_core import DEMDataset, FluidDEMDataset
from master.train.train_utils import normalize_inputs
from master.train.checkpoints import load_checkpoint, read_file_from_ini
from master.configs.config_utils import load_config_file
from master.train.trainer_new import load_train_objs, prepare_dataloader

def precompute_lowres_DEM(
    input_tif,
    output_tif,
    desired_resolution=5.0
):
    """
    Precompute a low-res DEM by sampling only the needed points, row by row.
    This avoids loading the full raster or all output into memory.
    """
    import rasterio
    import numpy as np

    with rasterio.open(input_tif) as src:
        left, bottom, right, top = src.bounds
        lons = np.arange(left, right, desired_resolution)
        lats = np.arange(bottom, top, desired_resolution)
        width = len(lons)
        height = len(lats)

        # Prepare output file
        profile = src.profile
        profile.update({
            'height': height,
            'width': width,
            'dtype': 'float32',
            'count': 1,
            'compress': 'lzw',
            'transform': rasterio.transform.from_bounds(left, bottom, right, top, width, height)
        })
        with rasterio.open(output_tif, 'w', **profile) as dst:
            min_val, max_val = None, None
            # First pass: find min/max for normalization
            for i, lat in enumerate(lats):
                coords = np.column_stack((lons, np.full_like(lons, lat)))
                vals = np.array([val[0] for val in src.sample(coords)], dtype=np.float32)
                valid = vals[~np.isnan(vals)]
                if valid.size > 0:
                    vmin, vmax = valid.min(), valid.max()
                    min_val = vmin if min_val is None else min(min_val, vmin)
                    max_val = vmax if max_val is None else max(max_val, vmax)
            # Second pass: write normalized rows
            for i, lat in enumerate(lats):
                coords = np.column_stack((lons, np.full_like(lons, lat)))
                vals = np.array([val[0] for val in src.sample(coords)], dtype=np.float32)
                # Normalize
                if min_val is not None and max_val is not None and max_val > min_val:
                    vals = (vals - min_val) / (max_val - min_val)
                else:
                    vals[:] = 0
                dst.write(vals.reshape(1, 1, -1), window=((i, i+1), (0, width)))
    print(f"Low-res DEM saved to {output_tif}")

from rasterio.warp import calculate_default_transform, reproject, Resampling

def reproject_to_latlon(input_tif, output_tif):
    dst_crs = "+proj=longlat +a=1737400 +b=1737400 +no_defs"  # Sphere with Moon radius
    with rasterio.open(input_tif) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        profile = src.profile.copy()
        profile.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open(output_tif, 'w', **profile) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear
            )
    print(f"Reprojected DEM saved to {output_tif}")

def generate_downsampled_DEM(
    input_tif,
    output_tif,
    downsample_factor=100
):
    """
    Generate a downsampled DEM by reading the full raster and resizing.
    This may use more memory but is simpler.
    """
    import rasterio
    from rasterio.enums import Resampling

    with rasterio.open(input_tif) as src:
        data = src.read(
            out_shape=(
                src.count,
                int(src.height / downsample_factor),
                int(src.width / downsample_factor)
            ),
            resampling=Resampling.bilinear
        )
        profile = src.profile
        profile.update({
            'height': data.shape[1],
            'width': data.shape[2],
            'compress': 'lzw',
            'transform': src.transform * src.transform.scale(
            (src.width / data.shape[2]),
            (src.height / data.shape[1])
        )
        })
        with rasterio.open(output_tif, 'w', **profile) as dst:
            dst.write(data)

    # reproject to lat/lon
    temp_tif = output_tif.replace('.tif', '_temp.tif')
    shutil.move(output_tif, temp_tif)
    reproject_to_latlon(temp_tif, output_tif)
    os.remove(temp_tif)
    print(f"Downsampled DEM saved to {output_tif}")

def plot_lro_metadata_distributions(data_dir, data_folder, figures_dir):
    # check that data directory exists
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory {data_dir} does not exist.")

    print(f"Loading data from {data_dir}...")

    # Look for .pt files in the test_dems folder
    candidate_files = sorted(glob.glob(os.path.join(data_dir, '*.pt')))
    if len(candidate_files) > 0:
        test_files = candidate_files
        print(f"Using {len(test_files)} test files from: {data_dir}")
    else:
        print(f"Found '{data_dir}' but no .pt files were present. Falling back to history['test_files'].")

    # loop through files and extract lro_meta information
    all_lro_meta = []
    for file_path in test_files:
        data = torch.load(file_path)
        if 'lro_meta' in data:
            lro_meta = data['lro_meta'].numpy()
            all_lro_meta.append(lro_meta)
        else:
            raise KeyError(f"'lro_meta' not found in file {file_path}.")
        
    all_lro_meta = np.array(all_lro_meta)

    # Create a figure with 2 rows: first row is 1 big plot, second row is 4 histograms
    fig = plt.figure(figsize=(22, 10))
    gs = fig.add_gridspec(2, 4, height_ratios=[3, 1])

    # First row: 1 big scatter plot for (lon, lat) -- flipped axes
    # Make ax_big span all columns to be as wide as the plot
    ax_big = fig.add_subplot(gs[0, :])
    ax_big.set_title('Locations sampled', fontsize=16)
    ax_big.set_xlabel('Longitude (°)')
    ax_big.set_ylabel('Latitude (°)')

    # Moon mean radius in meters
    MOON_RADIUS_M = 1737400.0
    # Example usage:
    moon_tif_file = "master/lro_data_sim/Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014.tif"
    downsample_factor = 100
    moon_tif_downsampled = f"master/lro_data_sim/Lunar_LRO_LOLA_Global_LDEM_downsampled_{downsample_factor}x.tif"

    if not os.path.exists(moon_tif_downsampled):
        print(f"Generating downsampled DEM with factor {downsample_factor} for fast plotting...")
        generate_downsampled_DEM(moon_tif_file, moon_tif_downsampled, downsample_factor=downsample_factor)
    
    print(f"Downsampled DEM exists at {moon_tif_downsampled}")

    try:
        with rasterio.open(moon_tif_downsampled) as src:
            moon_img = src.read(1).astype(np.float32)
            valid = ~np.isnan(moon_img)
            if np.any(valid):
                moon_img[valid] -= np.nanmin(moon_img[valid])
                max_val = np.nanmax(moon_img[valid])
                if max_val > 0:
                    moon_img[valid] /= max_val
                else:
                    print("Warning: max_val is 0 after normalization.")
            else:
                print("Warning: No valid data in DEM!")
            left, bottom, right, top = src.bounds
            ax_big.imshow(
                moon_img,
                cmap='gray',
                extent=[left, right, bottom, top],
                origin='upper',
                alpha=0.7,
                zorder=0
            )

    except Exception as e:
        print(f"Could not load background TIFF: {e}")

    # Plot each sample as a filled rectangle (box) on the map, and plot centers as dots
    import matplotlib.patches as mpatches
    for meta in all_lro_meta:
        lat_c = meta[0]
        lon_c = meta[1]
        box_radius_m = meta[2]
        # Compute box size in degrees
        dlat_deg = (box_radius_m / MOON_RADIUS_M) * (180.0 / np.pi)
        # Avoid division by zero at poles
        cos_lat = np.cos(np.deg2rad(lat_c))
        if np.abs(cos_lat) < 1e-6:
            dlon_deg = 0.0
        else:
            dlon_deg = (box_radius_m / (MOON_RADIUS_M * cos_lat)) * (180.0 / np.pi)
        lon_ll = lon_c - dlon_deg
        lat_ll = lat_c - dlat_deg
        width = 2 * dlon_deg
        height = 2 * dlat_deg
        rect = mpatches.Rectangle(
            (lon_ll, lat_ll), width, height,
            linewidth=0, edgecolor='none', facecolor='red', alpha=0.5
        )
        ax_big.add_patch(rect)
    
    ax_big.set_xlim([-180, 180])
    ax_big.set_ylim([-90, 90])

    # Second row: 4 histograms for each parameter
    param_names = ['Latitude (°)', 'Longitude (°)', 'Box Radius (m)', 'Height Normalization (m)']
    for i in range(4):
        ax = fig.add_subplot(gs[1, i])
        ax.hist(all_lro_meta[:, i], bins=30, color='red', alpha=0.7)
        ax.set_title(f'Distribution of {param_names[i]}')
        ax.set_xlabel(param_names[i])
        ax.set_ylabel('Frequency')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.suptitle('LRO Metadata: Locations sampled and Parameter Distributions', fontsize=18)
    

    filename = "lro_meta_distributions_" + data_folder + ".pdf"
    os.makedirs(figures_dir, exist_ok=True)
    save_path =  os.path.join(figures_dir, filename)
    plt.savefig(save_path)
    print(f"Saved LRO metadata distributions to {save_path}.")
    plt.close()


def plot_two_lro_metadata_distributions(data_dir1, data_folder1, data_dir2, data_folder2, figures_dir):
    all_lro_meta_1 = None
    all_lro_meta_2 = None
    for i, data_dir in enumerate([data_dir1, data_dir2]):
        # check that data directory exists
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory {data_dir} does not exist.")

        print(f"Loading data from {data_dir}...")

        # Look for .pt files in the test_dems folder
        candidate_files = sorted(glob.glob(os.path.join(data_dir, '*.pt')))
        if len(candidate_files) > 0:
            test_files = candidate_files
            print(f"Using {len(test_files)} test files from: {data_dir}")
        else:
            print(f"Found '{data_dir}' but no .pt files were present. Falling back to history['test_files'].")

        # loop through files and extract lro_meta information
        all_lro_meta = []
        for file_path in test_files:
            data = torch.load(file_path)
            if 'lro_meta' in data:
                lro_meta = data['lro_meta'].numpy()
                all_lro_meta.append(lro_meta)
            else:
                raise KeyError(f"'lro_meta' not found in file {file_path}.")
            
        if i == 0:
            all_lro_meta_1 = np.array(all_lro_meta)
        else:
            all_lro_meta_2 = np.array(all_lro_meta)

    # Create plot like big_ax above - but without the histograms!
    fig = plt.figure(figsize=(12, 6))
    ax_big = fig.add_subplot(1, 1, 1)
    ax_big.set_title('Locations sampled', fontsize=16)
    ax_big.set_xlabel('Longitude (°)')
    ax_big.set_ylabel('Latitude (°)')

        # Moon mean radius in meters
    MOON_RADIUS_M = 1737400.0
    # Example usage:
    moon_tif_file = "master/lro_data_sim/Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014.tif"
    downsample_factor = 100
    moon_tif_downsampled = f"master/lro_data_sim/Lunar_LRO_LOLA_Global_LDEM_downsampled_{downsample_factor}x.tif"

    if not os.path.exists(moon_tif_downsampled):
        print(f"Generating downsampled DEM with factor {downsample_factor} for fast plotting...")
        generate_downsampled_DEM(moon_tif_file, moon_tif_downsampled, downsample_factor=downsample_factor)
    
    print(f"Downsampled DEM exists at {moon_tif_downsampled}")

    try:
        with rasterio.open(moon_tif_downsampled) as src:
            moon_img = src.read(1).astype(np.float32)
            valid = ~np.isnan(moon_img)
            if np.any(valid):
                moon_img[valid] -= np.nanmin(moon_img[valid])
                max_val = np.nanmax(moon_img[valid])
                if max_val > 0:
                    moon_img[valid] /= max_val
                else:
                    print("Warning: max_val is 0 after normalization.")
            else:
                print("Warning: No valid data in DEM!")
            left, bottom, right, top = src.bounds
            ax_big.imshow(
                moon_img,
                cmap='gray',
                extent=[left, right, bottom, top],
                origin='upper',
                alpha=0.7,
                zorder=0
            )
            # print("DEM bounds:", left, right, bottom, top)

    except Exception as e:
        print(f"Could not load background TIFF: {e}")

    import matplotlib.patches as mpatches

    # Plot rectangles for each dataset
    for all_lro_meta, color, name in zip([all_lro_meta_1, all_lro_meta_2], ['red', 'blue'], [data_folder1, data_folder2]):
        # print(f"Plotting metadata for dataset: {name}")
        for meta in all_lro_meta:
            print(f"Lat: {meta[0]}, Lon: {meta[1]}, Box Radius (m): {meta[2]}")
            lat_c = meta[0]
            lon_c = meta[1]
            box_radius_m = meta[2]
            # Compute box size in degrees
            dlat_deg = (box_radius_m / MOON_RADIUS_M) * (180.0 / np.pi)
            cos_lat = np.cos(np.deg2rad(lat_c))
            if np.abs(cos_lat) < 1e-6:
                dlon_deg = 0.0
            else:
                dlon_deg = (box_radius_m / (MOON_RADIUS_M * cos_lat)) * (180.0 / np.pi)
            lon_ll = lon_c - dlon_deg
            lat_ll = lat_c - dlat_deg
            width = 2 * dlon_deg
            height = 2 * dlat_deg
            rect = mpatches.Rectangle(
                (lon_ll, lat_ll), width, height,
                linewidth=0, edgecolor='none', facecolor=color, alpha=0.5
            )
            ax_big.add_patch(rect)
        #     print(f"Plotted rectangle at Lon: {lon_ll} to {lon_ll + width}, Lat: {lat_ll} to {lat_ll + height}")
        # print("\n"*20)

    ax_big.set_xlim([-180, 180])
    ax_big.set_ylim([-90, 90])

    # Add legend
    legend_patches = [
        mpatches.Patch(color='red', label=data_folder1),
        mpatches.Patch(color='blue', label=data_folder2),
    ]
    ax_big.legend(handles=legend_patches, loc='upper right')

    plt.suptitle(f"LRO Metadata Distributions: {data_folder} vs {second_data_folder}", fontsize=18)
    
    filename = "lro_meta_distributions_" + data_folder + "_vs_" + second_data_folder + ".pdf"
    os.makedirs(figures_dir, exist_ok=True)
    save_path =  os.path.join(figures_dir, filename)
    plt.savefig(save_path)
    print(f"Saved LRO metadata distributions to {save_path}.")
    plt.close()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="")
    args.add_argument('run_dir', nargs="?" , type=str,
                      help='Directory of the trained model run containing stats and checkpoints.')
    args.add_argument('--run_dir', type=str, dest='run_dir_flag', required=False,
                      help='Directory of the trained model run containing stats and checkpoints.')
    args.add_argument('data_folder', nargs="?" , type=str,
                      help='Directory containing data to analyze.')
    args.add_argument('second_data_folder', nargs="?" , type=str,
                      help='Second directory containing data to analyze.')
    args.add_argument('--variant', type=str, default='first',
                      help="Variant for selecting test sets: 'first' or 'random'.")
    args.add_argument('--use_train_set', action='store_true',
                      help="Use the training set for plotting instead of the test set.")
    args.add_argument('--test_on_separate_data', action='store_true',
                      help="Indicates testing on separate data.")
    args = args.parse_args()

    # Support both positional and flag-based arguments
    run_dir = args.run_dir_flag or args.run_dir

    # if not specified, use 'test' folder
    data_folder = args.data_folder or "test"
    data_dir = os.path.join("runs", run_dir, data_folder)
    figures_dir = os.path.join("runs", run_dir, "figures")

    # Call the function with parsed arguments
    if not args.second_data_folder:
        plot_lro_metadata_distributions(data_dir, data_folder, figures_dir)
    else:
        second_data_folder = args.second_data_folder
        data_dir2 = os.path.join("runs", run_dir, second_data_folder)
        plot_two_lro_metadata_distributions(data_dir, data_folder, data_dir2, second_data_folder, figures_dir)
