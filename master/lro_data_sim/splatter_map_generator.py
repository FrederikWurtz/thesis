import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift
from scipy.fft import ifftshift
from tqdm import tqdm, trange
import torch
from joblib import Parallel, delayed
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject, transform_bounds
import rasterio
import os

import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter

def create_splatter_band(width, height, n_dots, dot_radius):
    """
    GPU-accelerated version using PyTorch.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Creating splatter band of size ({height}, {width}) with {n_dots} dots of radius {dot_radius} using device {device} acceleration.")
    band = torch.zeros((height, width), dtype=torch.float32, device=device)
    cy = torch.randint(0, height, (n_dots,), device=device)
    cx = torch.randint(0, width, (n_dots,), device=device)
    yy = torch.arange(height, device=device).view(-1, 1)
    xx = torch.arange(width, device=device).view(1, -1)

    for i in tqdm(range(n_dots)):
        dy = torch.minimum(torch.abs(yy - cy[i]), height - torch.abs(yy - cy[i]))
        dx = torch.minimum(torch.abs(xx - cx[i]), width - torch.abs(xx - cx[i]))
        mask = (dy**2 + dx**2) <= dot_radius**2
        band += mask.float()

    # Normalize
    if band.max() > 0:
        band /= band.max()
    # Smooth with wrap-around (no direct torch equivalent, so use CPU for this step)
    band_cpu = band.cpu().numpy()
    # from scipy.ndimage import gaussian_filter
    # band_cpu = gaussian_filter(band_cpu, sigma=dot_radius/2, mode='wrap')
    # if band_cpu.max() > 0:
    #     band_cpu /= band_cpu.max()
    return band_cpu

def main():
    dem_path = "master/lro_data_sim/Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014.tif"
    with rasterio.open(dem_path) as dataset:
        dem = dataset.read(1)
        transform = dataset.transform
        crs = dataset.crs
        # print("File info:")
        # print(f"  Path: {dem_path}")
        # print(f"  Width: {dataset.width}")
        # print(f"  Height: {dataset.height}")
        # print(f"  Count (bands): {dataset.count}")
        # print(f"  Dtype: {dataset.dtypes[0]}")
        # print(f"  CRS: {crs}")
        # print(f"  Transform: {transform}")
        # print(f"  Bounds: {dataset.bounds}")
        # print(f"  Driver: {dataset.driver}")
        # print(f"  Nodata: {dataset.nodata}")
        # print(f"  Compression: {dataset.compression}")
        # print(f"  Tiled: {dataset.is_tiled}")
        # print(f"  Block shapes: {dataset.block_shapes}")
        # print(f"  Metadata: {dataset.meta}")

    # create new band with value 1, and add band to dataset (same dataytype as dem)
    test_band_1 = np.full_like(dem, fill_value=0.4, dtype=dem.dtype)
    test_band_2 = np.full_like(dem, fill_value=0.5, dtype=dem.dtype)

    # add to dataset and save to new file
    new_dem_path = "master/lro_data_sim/Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014_with_test_bands.tif"
    with rasterio.open(new_dem_path, 'w', driver='GTiff',
                       height=dem.shape[0], width=dem.shape[1],
                       count=3, dtype=dem.dtype,
                       crs=crs, transform=transform) as new_dataset:
        new_dataset.write(dem, 1)
        new_dataset.write(test_band_1, 2)
        new_dataset.write(test_band_2, 3)
    print(f"New DEM with test bands saved to {new_dem_path}")

    # splatter_map = create_splatter_band(10000, 10000, n_dots=100000, dot_radius=20)
    # plt.imshow(splatter_map, cmap='gray')
    # plt.title("Splatter Map")
    # plt.colorbar()
    # save_dir = "master/lro_data_sim/figures/"
    # os.makedirs(save_dir, exist_ok=True)
    # save_path = os.path.join(save_dir, "splatter_map.png")
    # plt.savefig(save_path)
    # plt.close()
    # print(f"Splatter map saved to {save_path}")

    # # also plot zommed in version of this
    # zoomed_in = splatter_map[4500:5500, 4500:5500]
    # plt.imshow(zoomed_in, cmap='gray')
    # plt.title("Zoomed-in Splatter Map")
    # plt.colorbar()
    # zoomed_save_path = os.path.join(save_dir, "splatter_map_zoomed.png")
    # plt.savefig(zoomed_save_path)
    # plt.close()
    # print(f"Zoomed-in splatter map saved to {zoomed_save_path}")



if __name__ == "__main__":
    main()