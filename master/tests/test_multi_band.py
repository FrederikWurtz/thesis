import math
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.windows import Window
import os
from tqdm import tqdm
import torch

from master.train.trainer_new import FluidDEMDataset
from master.train.trainer_new import load_train_objs, normalize_inputs
from master.train.train_utils import normalize_inputs, compute_input_stats, round_list
from torch.utils.data import DataLoader

from master.configs.config_utils import load_config_file
from master.models.losses import calculate_total_loss, calculate_total_loss_multi_band
from master.models.unet import UNet


from master.configs.config_utils import load_config_file
from master.lro_data_sim.lro_generator_multi_band import generate_and_return_lro_data_multi_band

# -----------------------------
# Device helper
# -----------------------------
def get_best_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def plot_all_in_one_figure(
    images,
    reflectance_maps,
    dem_tensor,
    w_tensor,
    theta_bar_tensor,
    out_path="figures/overview.png",
    max_show=3
):
    """
    Lav én stor figur med:
      - op til max_show images og deres reflectance_maps
      - DEM
      - W-band
      - Theta-band

    images, reflectance_maps: liste af torch.Tensor eller np.ndarray (H, W)
    dem_tensor, w_tensor, theta_bar_tensor: torch.Tensor eller np.ndarray (H, W)
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Konverter til numpy, hvis det er torch tensors
    def to_np(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x

    images_np = [to_np(img) for img in images]
    refls_np = [to_np(r) for r in reflectance_maps]
    dem_np = to_np(dem_tensor)
    w_np = to_np(w_tensor)
    theta_np = to_np(theta_bar_tensor)

    n_images = len(images_np)
    n_show = min(n_images, max_show)

    n_cols = n_show if n_show > 0 else 1
    n_rows = 3  # 1st: images, 2nd: reflectances, 3rd: dem/w/theta

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)
    if n_cols == 1:
        axes = np.expand_dims(axes, axis=1)

    # 1st row: images
    for i in range(n_show):
        ax_img = axes[0, i]
        im1 = ax_img.imshow(images_np[i], cmap="gray")
        ax_img.set_title(f"Image {i}")
        ax_img.axis("off")

    # 2nd row: reflectances
    for i in range(n_show):
        ax_refl = axes[1, i]
        im2 = ax_refl.imshow(refls_np[i], cmap="gray")
        ax_refl.set_title(f"Reflectance {i}")
        ax_refl.axis("off")

    # 3rd row: dem, w, theta (fill first 3 columns)
    band_titles = ["DEM", "W band", "Theta band"]
    band_data = [dem_np, w_np, theta_np]
    band_cmaps = ["terrain", "plasma", "plasma"]
    for j in range(3):
        if j < n_cols:
            ax = axes[2, j]
            im = ax.imshow(band_data[j], cmap=band_cmaps[j])
            ax.set_title(band_titles[j])
            ax.axis("off")
            vmin = np.nanmin(band_data[j])
            if j == 1:
                vmin = 0.1  # for W band, avoid very low values
            vmax = np.nanmax(band_data[j])
            im.set_clim(vmin, vmax)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Turn off any unused axes in 3rd row
    for j in range(3, n_cols):
        axes[2, j].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved overview figure to {out_path}")

def main():
    config = load_config_file()
    device = get_best_device()



    print(f"Use multi band set to {config['USE_MULTI_BAND']}")
    print(f"Use LRO DEMs set to {config['USE_LRO_DEMS']}")

    epoch_shared = 0  # not used in this test
    train_set = FluidDEMDataset(config, epoch_shared=epoch_shared)
    train_data =  DataLoader(
            train_set,
            batch_size=3,
            pin_memory=True,
            shuffle=False,
            sampler=None,
            num_workers=1,
        )

    #print info on output of train_data
    images, reflectance_maps, dem_tensor, metas, w_tensor, theta_bar_tensor, lro_meta = next(iter(train_data))

    print("Generated LRO multi-band data.")

    with torch.no_grad():
        if torch.isnan(images).any():
            nan_mask = torch.isnan(images)
            idx = torch.nonzero(nan_mask, as_tuple=False)[0].tolist()
            b, c, y, x = idx
            print(f"[NaN] images contains NaNs. First at batch={b}, channel={c}, y={y}, x={x}")
            ch_bc = images[b, c]
            ch_bc_masked = ch_bc[~torch.isnan(ch_bc)]
            if ch_bc_masked.numel() > 0:
                print(f"[NaN] images[b,c] stats: min={ch_bc_masked.min().item():.6f}, max={ch_bc_masked.max().item():.6f}")
            else:
                print("[NaN] images[b,c] is all NaN")
        else:
            print("[NaN] images has no NaNs")

        # Per-channel stats (nan-aware)
        for c in range(images.shape[1]):
            ch = images[:, c]
            ch_masked = ch[~torch.isnan(ch)]
            if ch_masked.numel() == 0:
                print(f"[Chan {c}] all NaN")
                continue
            print(
                f"[Chan {c}] min={ch_masked.min().item():.6f}, "
                f"max={ch_masked.max().item():.6f}, "
                f"mean={ch_masked.mean().item():.6f}, "
                f"std={ch_masked.std(unbiased=False).item():.6f}, "
                f"has_nan={torch.isnan(ch).any().item()}"
            )

    #print lro_meta data nicely:
    print("LRO Meta Data:")
    print(lro_meta)

    # # make sure figures directory exists
    # os.makedirs("master/tests/figures", exist_ok=True)

    # # print diagnostics on reflection maps
    # for i, refl_map in enumerate(reflectance_maps):
    #     refl_np = refl_map.detach().cpu().numpy() if isinstance(refl_map, torch.Tensor) else refl_map
    #     print(f"Reflectance Map {i}: min={np.nanmin(refl_np)}, max={np.nanmax(refl_np)}, mean={np.nanmean(refl_np)}")
    #     # Does it contain nans?
    #     print(f"  Contains NaNs: {np.isnan(refl_np).any()}")

    print("Shapes:")
    print(f"images: {images.shape}")
    print(f"reflectance_maps: {reflectance_maps.shape}")
    print(f"dem_tensor: {dem_tensor.shape}")
    print(f"w_tensor: {w_tensor.shape}")
    print(f"theta_bar_tensor: {theta_bar_tensor.shape}")

    # Så kalder du:
    plot_all_in_one_figure(
        images=images[0],
        reflectance_maps=reflectance_maps[0],
        dem_tensor=dem_tensor[0].squeeze(0),
        w_tensor=w_tensor[0].squeeze(0),
        theta_bar_tensor=theta_bar_tensor[0].squeeze(0),
        out_path="master/tests/figures/overview_input.png",
        max_show=3,  # vis op til 3 billeder
    )

    # initialize a model and run a forward pass (just to check everything works)
    
    EPOCH_SHARED = 0

    out_channels = 3 # DEM, w band and theta_bar band
    model = UNet(in_channels=config["IMAGES_PER_DEM"], out_channels=out_channels, w_range=(config["W_MIN"], config["W_MAX"]), theta_range=(config["THETA_BAR_MIN"], config["THETA_BAR_MAX"]))

    model = model.to(device)

    # caluclate mean and std of images
    train_mean, train_std = compute_input_stats(train_data, images_per_dem=config["IMAGES_PER_DEM"])

    print(f"Mean: {train_mean}")
    print(f"Std: {train_std}")

    images_norm = normalize_inputs(images, train_mean, train_std).to(device)

    # move everything to device
    dem_tensor = dem_tensor.to(device)
    w_tensor = w_tensor.to(device)
    theta_bar_tensor = theta_bar_tensor.to(device)
    reflectance_maps = reflectance_maps.to(device)
    images_norm = images_norm.to(device)
    metas = metas.to(device)

    outputs = model(images_norm, metas, target_size=dem_tensor.shape[-2:])

    dem_outputs = outputs[:, 0:1, :, :] 
    w_outputs = outputs[:, 1:2, :, :] 
    theta_outputs = outputs[:, 2:3, :, :]

    # Debug: print shapes to see what we're working with
    print(f"dem_outputs shape: {dem_outputs.shape}")
    print(f"dem_tensor shape: {dem_tensor.shape}")
    print(f"w_outputs shape: {w_outputs.shape}")
    print(f"w_tensor shape: {w_tensor.shape}")

    if dem_outputs.shape != dem_tensor.shape:
        raise ValueError(f"Shape mismatch between dem_outputs {dem_outputs.shape} and dem_targets {dem_tensor.shape}")
    if w_outputs.shape != w_tensor.shape:
        raise ValueError(f"Shape mismatch between w_outputs {w_outputs.shape} and w_targets {w_tensor.shape}")
    if theta_outputs.shape != theta_bar_tensor.shape:
        raise ValueError(f"Shape mismatch between theta_outputs {theta_outputs.shape} and theta_targets {theta_bar_tensor.shape}")


    total_loss = calculate_total_loss_multi_band(
                                                dem_outputs, dem_tensor, reflectance_maps, metas, w_outputs, w_tensor, theta_outputs, theta_bar_tensor,
                                                device=device,
                                                config=config,
                                                return_components=True
                                            )

    loss_mse, loss_grad, loss_refl, loss_w, loss_theta, total_loss = total_loss

    # Check loss component values
    print(f"    Loss components: MSE={loss_mse.item():.6f}, Grad={loss_grad.item():.6f}, Refl={loss_refl.item():.6f}, w_band={loss_w.item():.6f}, theta_band={loss_theta.item():.6f}, Total={total_loss.item():.6f}")

    print(f"images.shape: {images.shape}")
    print(f"reflectance_maps.shape: {reflectance_maps.shape}")
    print(f"dem_tensor.shape: {dem_tensor.shape}")
    print(f"w_tensor.shape: {w_tensor.shape}")
    print(f"theta_bar_tensor.shape: {theta_bar_tensor.shape}")

    plot_all_in_one_figure(
        images=images[0],
        reflectance_maps=reflectance_maps[0],
        dem_tensor=dem_outputs[0].squeeze(0),
        w_tensor=w_outputs[0].squeeze(0),
        theta_bar_tensor=theta_outputs[0].squeeze(0),
        out_path="master/tests/figures/overview_output.png",
        max_show=3,  # vis op til 3 billeder
    )


if __name__ == "__main__":

    # print(get_moon_latlon_bounds("master/lro_data_sim/Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014_with_bands.tif"))

    print("Running multi-band LRO data generation test...")
    main()
    print("Test completed.")