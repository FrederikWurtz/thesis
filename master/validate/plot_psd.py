import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import torch
import shutil
import re

from tqdm import tqdm
from master.train.train_utils import normalize_inputs
from master.train.checkpoints import load_checkpoint, read_file_from_ini
from master.configs.config_utils import load_config_file
from master.train.trainer_core import load_train_objs, DEMDataset


# ============================================================
# ✅ UTILITIES
# ============================================================

def is_latex_available():
    for cmd in ['pdflatex', 'latex', 'xelatex']:
        if shutil.which(cmd) is not None:
            return True
    return False

plt.rcParams.update({
    'text.usetex': is_latex_available(),
    'font.size': 12
})


# Set matplotlib parameters based on LaTeX availability
use_latex = is_latex_available()

plt.rcParams.update({
    'text.usetex': use_latex,
    'font.size': 12,
    'font.family': 'serif' if use_latex else 'DejaVu Serif',
    'axes.labelsize': 12,
    'xtick.labelsize': 12,
    'xtick.top': True,
    'xtick.direction': 'out',
    'ytick.labelsize': 12,
    'ytick.right': True,
    'ytick.direction': 'out',
    'legend.fontsize': 12,
    'xtick.major.size': 10,
    'xtick.minor.size': 5,
    'ytick.major.size': 10,
    'ytick.minor.size': 5
})

if not use_latex:
    print("Warning: LaTeX not found. Using default matplotlib fonts.")


# ============================================================
# ✅ POWER SPECTRUM
# ============================================================

def radial_psd(dem):
    """
    Compute radially averaged power spectral density.
    Returns frequency (1/m) and PSD.
    """

    H, W = dem.shape

    # --- Apply 2D Hann window ---
    wx = np.hanning(W)
    wy = np.hanning(H)
    window = np.outer(wy, wx)
    dem_win = dem * window

    # --- FFT ---
    fft = np.fft.fft2(dem_win)
    fft = np.fft.fftshift(fft)
    psd2 = np.abs(fft) ** 2

    # --- Frequency grid ---
    fx = np.fft.fftfreq(W, d=1.0)
    fy = np.fft.fftfreq(H, d=1.0)
    fx, fy = np.meshgrid(fx, fy)
    fr = np.sqrt(fx**2 + fy**2)

    fr = np.fft.fftshift(fr)

    # --- Radial binning ---
    fr_flat = fr.flatten()
    psd_flat = psd2.flatten()

    # bins: from min to Nyquist
    nbins = min(H, W) // 2
    bins = np.linspace(0, fr_flat.max(), nbins)

    bin_ids = np.digitize(fr_flat, bins)

    psd_radial = np.zeros(nbins)
    fr_radial = np.zeros(nbins)

    for i in range(1, nbins):
        mask = bin_ids == i
        if np.any(mask):
            psd_radial[i] = psd_flat[mask].mean()
            fr_radial[i] = fr_flat[mask].mean()

    # remove zero freq
    mask = fr_radial > 0
    return fr_radial[mask], psd_radial[mask]


# ============================================================
# ✅ VARIOGRAM
# ============================================================

def variogram(dem, max_lag=None):
    """
    Isotropic variogram (radial).
    """

    H, W = dem.shape

    if max_lag is None:
        max_lag = min(H, W) // 4

    lags = np.arange(1, max_lag)
    gamma = np.zeros_like(lags, dtype=float)

    for i, h in enumerate(lags):
        diffs = []

        # shift in x
        dx = dem[:, h:] - dem[:, :-h]
        diffs.append(dx)

        # shift in y
        dy = dem[h:, :] - dem[:-h, :]
        diffs.append(dy)

        diff_all = np.concatenate([d.flatten() for d in diffs])

        gamma[i] = 0.5 * np.mean(diff_all ** 2)

    return lags, gamma


# ============================================================
# ✅ MAIN PLOT FUNCTION
# ============================================================

def plot_analysis(dataset_number=1, use_train_set=False):

    sup_dir = "runs"
    run_path = os.path.join(sup_dir, run_dir)

    checkpoint = load_checkpoint(os.path.join(run_path, 'checkpoints', 'snapshot.pt'), map_location='cpu')
    config = load_config_file(os.path.join(run_path, 'stats', 'config.ini'))
    input_stats = read_file_from_ini(os.path.join(run_path, 'stats', 'input_stats.ini'), ftype=dict)

    dataset_dir = os.path.join(run_path, 'train' if use_train_set else 'test')
    files = sorted(glob.glob(os.path.join(dataset_dir, '*.pt')))
    dataset = DEMDataset(files, config=config)

    sample = dataset[dataset_number]
    images_tensor, _, target_tensor, meta_tensor, _, _, _ = sample

    _, _, _, model, _, _ = load_train_objs(config, run_path)
    model.load_state_dict(checkpoint['MODEL_STATE'])
    model.eval()

    mean = torch.tensor(input_stats['MEAN']).view(1, -1, 1, 1)
    std = torch.tensor(input_stats['STD']).view(1, -1, 1, 1)

    images_batch = images_tensor.unsqueeze(0)
    images_norm = normalize_inputs(images_batch, mean, std)

    with torch.no_grad():
        outputs = model(images_norm, meta_tensor.unsqueeze(0))

    dem_pred = outputs.squeeze(0)[0].cpu().numpy()
    dem_gt = target_tensor.squeeze().numpy()

    # ============================================================
    # ✅ COMPUTE ANALYSIS
    # ============================================================

    f_gt, psd_gt = radial_psd(dem_gt)
    f_pred, psd_pred = radial_psd(dem_pred)

    lag_gt, var_gt = variogram(dem_gt)
    lag_pred, var_pred = variogram(dem_pred)

    # convert freq → wavelength (meters)
    wl_gt = 1.0 / f_gt
    wl_pred = 1.0 / f_pred

    # ============================================================
    # ✅ FIGURE (CLEAN 3x3 GRID)
    # ============================================================

    fig = plt.figure(figsize=(13, 11))

    # ============================================================
    # ✅ DEMs (TOP ROW)
    # ============================================================

    ax1 = plt.subplot(3, 3, 1)
    ax2 = plt.subplot(3, 3, 2)

    vmin = min(dem_gt.min(), dem_pred.min())
    vmax = max(dem_gt.max(), dem_pred.max())

    im1 = ax1.imshow(dem_gt, cmap='terrain', origin='lower', vmin=vmin, vmax=vmax)
    im2 = ax2.imshow(dem_pred, cmap='terrain', origin='lower', vmin=vmin, vmax=vmax)

    ax1.set_title("Ground Truth DEM")
    ax2.set_title("Predicted DEM")

    for ax in [ax1, ax2]:
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")

    # colorbar attached to predicted DEM
    cbar = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label("Elevation [m]")

    # leave (0,2) empty for spacing


    # ============================================================
    # ✅ VARIOGRAM (MIDDLE ROW, FULL WIDTH)
    # ============================================================

    ax_var = plt.subplot(3, 1, 2)

    ax_var.plot(lag_gt, var_gt, label="GT")
    ax_var.plot(lag_pred, var_pred, label="Pred")

    ax_var.set_xlabel("Lag Distance [m]")
    ax_var.set_ylabel("Semivariance")
    ax_var.set_title("Variogram")
    ax_var.legend()
    ax_var.grid(alpha=0.3)


    # ============================================================
    # ✅ POWER SPECTRUM (BOTTOM ROW)
    # ============================================================

    # --- filter usable wavelengths ---
    mask_gt = (wl_gt > 2) & (wl_gt < min(dem_gt.shape))
    mask_pred = (wl_pred > 2) & (wl_pred < min(dem_pred.shape))

    wl_gt_plot = wl_gt[mask_gt]
    psd_gt_plot = psd_gt[mask_gt]

    wl_pred_plot = wl_pred[mask_pred]
    psd_pred_plot = psd_pred[mask_pred]

    max_wl = min(dem_gt.shape) / 2

    # --- interpolate to same wavelength grid ---
    common_wl = np.interp(wl_gt, wl_pred[::-1], wl_pred[::-1])  # simple alignment trick

    # safer: interpolate PSD onto GT wavelengths
    psd_pred_interp = np.interp(wl_gt, wl_pred[::-1], psd_pred[::-1])

    ratio = psd_pred_interp / psd_gt

    # --- find cutoff ---
    threshold = 0.5

    valid = ratio > threshold

    if np.any(valid):
        idx = np.where(valid)[0][-1]  # last point where still OK
        resolution_limit = wl_gt[idx]
    else:
        resolution_limit = np.nan

    print(f"Resolution limit ≈ {resolution_limit:.2f} m")


    # --- LOG-LOG ---
    ax_psd1 = plt.subplot(3, 3, 7)

    ax_psd1.loglog(wl_gt_plot, psd_gt_plot, label="GT")
    ax_psd1.loglog(wl_pred_plot, psd_pred_plot, label="Pred")

    ax_psd1.set_title("PSD (log-log)")

    ax_psd1.set_xlabel("Wavelength [m]")
    ax_psd1.set_ylabel("Power")
    ax_psd1.invert_xaxis()
    ax_psd1.set_xlim(max_wl, wl_gt_plot.min())

    # ✅ add resolution limit line
    if not np.isnan(resolution_limit):
        ax_psd1.axvline(
            resolution_limit,
            color='red',
            linestyle=':',
            linewidth=1.0,
            label=f'Resolution ≈ {resolution_limit:.1f} m'
        )

    ax_psd1.legend()
    ax_psd1.grid(alpha=0.3)


    # --- LOG-X ---
    ax_psd2 = plt.subplot(3, 3, 8)

    ax_psd2.semilogx(wl_gt_plot, psd_gt_plot, label="GT")
    ax_psd2.semilogx(wl_pred_plot, psd_pred_plot, label="Pred")

    ax_psd2.set_title("PSD (log-x)")
    ax_psd2.set_xlabel("Wavelength [m]")
    ax_psd2.set_ylabel("Power")
    ax_psd2.invert_xaxis()
    ax_psd2.legend()
    ax_psd2.grid(alpha=0.3)
    ax_psd2.set_xlim(max_wl, wl_gt_plot.min())



    # --- LINEAR ---
    ax_psd3 = plt.subplot(3, 3, 9)

    ax_psd3.plot(wl_gt_plot, psd_gt_plot, label="GT")
    ax_psd3.plot(wl_pred_plot, psd_pred_plot, label="Pred")

    ax_psd3.set_title("PSD (linear)")
    ax_psd3.set_xlabel("Wavelength [m]")
    ax_psd3.set_ylabel("Power")
    ax_psd3.invert_xaxis()
    ax_psd3.set_xlim(max_wl, wl_gt_plot.min())
    ax_psd3.legend()
    ax_psd3.grid(alpha=0.3)


    # ============================================================
    # ✅ FINALIZE
    # ============================================================

    plt.tight_layout()
    plt.show()

    out_dir = os.path.join(run_path, 'figures')
    os.makedirs(out_dir, exist_ok=True)

    tag = "train" if use_train_set else "test"
    path = os.path.join(out_dir, f'fineness_{tag}_{dataset_number}.pdf')

    plt.savefig(path, bbox_inches='tight', dpi=300)
    print(f"Saved figure to {path}")


def plot_psd_loglog_only(run_dir, dataset_number=1, use_train_set=False):
    sup_dir = "runs"
    run_path = os.path.join(sup_dir, run_dir)

    # --- load ---
    checkpoint = load_checkpoint(os.path.join(run_path, 'checkpoints', 'snapshot.pt'), map_location='cpu')
    config = load_config_file(os.path.join(run_path, 'stats', 'config.ini'))
    input_stats = read_file_from_ini(os.path.join(run_path, 'stats', 'input_stats.ini'), ftype=dict)

    dataset_dir = os.path.join(run_path, 'train' if use_train_set else 'test')
    files = sorted(glob.glob(os.path.join(dataset_dir, '*.pt')))
    dataset = DEMDataset(files, config=config)

    sample = dataset[dataset_number]
    images_tensor, _, target_tensor, meta_tensor, _, _, _ = sample

    _, _, _, model, _, _ = load_train_objs(config, run_path)
    model.load_state_dict(checkpoint['MODEL_STATE'])
    model.eval()

    # --- normalize ---
    mean = torch.tensor(input_stats['MEAN']).view(1, -1, 1, 1)
    std = torch.tensor(input_stats['STD']).view(1, -1, 1, 1)

    images_batch = images_tensor.unsqueeze(0)
    images_norm = normalize_inputs(images_batch, mean, std)

    # --- inference ---
    with torch.no_grad():
        outputs = model(images_norm, meta_tensor.unsqueeze(0))

    dem_pred = outputs.squeeze(0)[0].cpu().numpy()
    dem_gt = target_tensor.squeeze().numpy()

    # ============================================================
    # ✅ PSD
    # ============================================================

    f_gt, psd_gt = radial_psd(dem_gt)
    f_pred, psd_pred = radial_psd(dem_pred)

    wl_gt = 1.0 / f_gt
    wl_pred = 1.0 / f_pred

    max_wl = min(dem_gt.shape) / 2

    mask_gt = (wl_gt > 2) & (wl_gt < max_wl)
    mask_pred = (wl_pred > 2) & (wl_pred < max_wl)

    wl_gt = wl_gt[mask_gt]
    psd_gt = psd_gt[mask_gt]

    wl_pred = wl_pred[mask_pred]
    psd_pred = psd_pred[mask_pred]

    # --- resolution limit (same logic) ---
    psd_pred_interp = np.interp(wl_gt, wl_pred[::-1], psd_pred[::-1])
    ratio = psd_pred_interp / psd_gt

    threshold = 0.5
    resolution_limit = np.nan

    valid = ratio > threshold
    if np.any(valid):
        resolution_limit = wl_gt[np.where(valid)[0][-1]]

    # ============================================================
    # ✅ PLOT (ONLY THIS)
    # ============================================================

    plt.figure(figsize=(7, 6))

    plt.loglog(wl_gt, psd_gt, label="Ground Truth")
    plt.loglog(wl_pred, psd_pred, label="Prediction")

    if not np.isnan(resolution_limit):
        plt.axvline(
            resolution_limit,
            color='red',
            linestyle=':',
            linewidth=1.0,
            label=f'Resolution ≈ {resolution_limit:.1f} m'
        )

    plt.xlabel("Wavelength [m]")
    plt.ylabel("Power")
    #plt.title("Power Spectrum Density")
    plt.gca().invert_xaxis()
    plt.xlim(max_wl, wl_gt.min())

    plt.grid(alpha=0.3)
    plt.legend()

    # ============================================================
    # ✅ FINALIZE
    # ============================================================

    plt.tight_layout()
    plt.show()

    out_dir = os.path.join(run_path, 'figures')
    os.makedirs(out_dir, exist_ok=True)

    tag = "train" if use_train_set else "test"
    path = os.path.join(out_dir, f'psd_{tag}_{dataset_number}.pdf')

    plt.savefig(path, bbox_inches='tight', dpi=300)
    print(f"Saved figure to {path}")

def plot_psd_loglog_average(run_dir, use_train_set=False):
    sup_dir = "runs"
    run_path = os.path.join(sup_dir, run_dir)

    # --- load ---
    checkpoint = load_checkpoint(os.path.join(run_path, 'checkpoints', 'snapshot_best.pt'), map_location='cpu')
    config = load_config_file(os.path.join(run_path, 'stats', 'config.ini'))
    input_stats = read_file_from_ini(os.path.join(run_path, 'stats', 'input_stats.ini'), ftype=dict)

    dataset_dir = os.path.join(run_path, 'train' if use_train_set else 'test')
    files = sorted(glob.glob(os.path.join(dataset_dir, '*.pt')))
    dataset = DEMDataset(files, config=config)

    _, _, _, model, _, _ = load_train_objs(config, run_path)
    model.load_state_dict(checkpoint['MODEL_STATE'])
    model.eval()

    # --- normalization ---
    mean = torch.tensor(input_stats['MEAN']).view(1, -1, 1, 1)
    std = torch.tensor(input_stats['STD']).view(1, -1, 1, 1)

    # --- common wavelength grid ---
    sample = dataset[0]
    _, _, target_tensor, _, _, _, _ = sample
    dem_shape = target_tensor.squeeze().numpy().shape

    max_wl = min(dem_shape) / 2
    wl_common = np.logspace(np.log10(2), np.log10(max_wl), 200)

    psd_gt_all = []
    psd_pred_all = []
    resolutions = []

    # ============================================================
    # ✅ LOOP
    # ============================================================
    for sample in tqdm(dataset, total=len(dataset), desc="PSD"):
        images_tensor, _, target_tensor, meta_tensor, _, _, _ = sample

    #for i in tqdm(range(10), desc="PSD"):
    #    sample = dataset[i]
    #    images_tensor, _, target_tensor, meta_tensor, _, _, _ = sample


        images_batch = images_tensor.unsqueeze(0)
        images_norm = normalize_inputs(images_batch, mean, std)

        with torch.no_grad():
            outputs = model(images_norm, meta_tensor.unsqueeze(0))

        dem_pred = outputs.squeeze(0)[0].cpu().numpy()
        dem_gt = target_tensor.squeeze().numpy()

        # --- PSD ---
        f_gt, psd_gt = radial_psd(dem_gt)
        f_pred, psd_pred = radial_psd(dem_pred)

        wl_gt = 1.0 / f_gt
        wl_pred = 1.0 / f_pred

        mask_gt = (wl_gt > 2) & (wl_gt < max_wl)
        mask_pred = (wl_pred > 2) & (wl_pred < max_wl)

        wl_gt_f = wl_gt[mask_gt]
        psd_gt_f = psd_gt[mask_gt]

        wl_pred_f = wl_pred[mask_pred]
        psd_pred_f = psd_pred[mask_pred]

        # ========================================================
        # ✅ Resolution per sample (FIX #2)
        # ========================================================

        if len(wl_gt_f) > 0 and len(wl_pred_f) > 0:
            psd_pred_interp = np.interp(
                wl_gt_f,
                wl_pred_f[::-1],
                psd_pred_f[::-1]
            )

            ratio = psd_pred_interp / psd_gt_f
            valid = ratio > 0.5

            if np.any(valid):
                res = wl_gt_f[np.where(valid)[0][-1]]
                resolutions.append(res)

        # ========================================================
        # ✅ interpolation for averaging
        # ========================================================

        wl_gt_i = wl_gt_f[::-1]
        psd_gt_i = psd_gt_f[::-1]

        wl_pred_i = wl_pred_f[::-1]
        psd_pred_i = psd_pred_f[::-1]

        psd_gt_interp = np.interp(wl_common, wl_gt_i, psd_gt_i, left=np.nan, right=np.nan)
        psd_pred_interp = np.interp(wl_common, wl_pred_i, psd_pred_i, left=np.nan, right=np.nan)

        psd_gt_all.append(psd_gt_interp)
        psd_pred_all.append(psd_pred_interp)

    psd_gt_all = np.array(psd_gt_all)
    psd_pred_all = np.array(psd_pred_all)
    resolutions = np.array(resolutions)


    # ============================================================
    # ✅ REMOVE INVALID WAVELENGTHS (critical fix)
    # ============================================================
    valid_counts = np.sum(~np.isnan(psd_gt_all), axis=0)

    # require at least 5% of samples to contribute
    n_samples = len(psd_gt_all)

    # adaptive threshold (works for both small + large datasets)
    min_count = max(2, int(0.1 * n_samples))


    valid_mask = valid_counts >= min_count
    print(f"Minimum count, in order to keep {min_count}")
    print(f"Kept {len(wl_common)} / {len(valid_counts)} wavelengths")

    # apply mask
    wl_common = wl_common[valid_mask]
    psd_gt_all = psd_gt_all[:, valid_mask]
    psd_pred_all = psd_pred_all[:, valid_mask]


    # ============================================================
    # ✅ PSD statistics (log-space)
    # ============================================================

    psd_gt_log = np.log(psd_gt_all)
    psd_pred_log = np.log(psd_pred_all)

    psd_gt_mean = np.exp(np.nanmean(psd_gt_log, axis=0))
    psd_pred_mean = np.exp(np.nanmean(psd_pred_log, axis=0))

    psd_gt_std = np.nanstd(psd_gt_log, axis=0)
    psd_pred_std = np.nanstd(psd_pred_log, axis=0)

    psd_gt_lower = np.exp(np.nanmean(psd_gt_log, axis=0) - psd_gt_std)
    psd_gt_upper = np.exp(np.nanmean(psd_gt_log, axis=0) + psd_gt_std)

    psd_pred_lower = np.exp(np.nanmean(psd_pred_log, axis=0) - psd_pred_std)
    psd_pred_upper = np.exp(np.nanmean(psd_pred_log, axis=0) + psd_pred_std)

    # ============================================================
    # ✅ Resolution statistics
    # ============================================================

    median_res = np.median(resolutions)
    p16 = np.percentile(resolutions, 16)
    p84 = np.percentile(resolutions, 84)

    print(f"Resolution ≈ {median_res:.2f} m (+{p84 - median_res:.2f} / -{median_res - p16:.2f})")

    # ============================================================
    # ✅ FIGURE (two panels)
    # ============================================================

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # ============================================================
    # ✅ PSD plot
    # ============================================================

    ax1.loglog(wl_common, psd_gt_mean, label="Ground Truth")
    ax1.loglog(wl_common, psd_pred_mean, label="Prediction")

    ax1.fill_between(wl_common, psd_gt_lower, psd_gt_upper, alpha=0.2)
    ax1.fill_between(wl_common, psd_pred_lower, psd_pred_upper, alpha=0.2)

    ax1.axvline(
        median_res,
        color='red',
        linestyle=':',
        linewidth=1.5,
        label=f'Resolution ≈ {median_res:.1f} m'
    )

    ax1.axvspan(p16, p84, color='red', alpha=0.15)

    ax1.set_xlabel("Wavelength [m]")
    ax1.set_ylabel("Power")
    ax1.invert_xaxis()
    ax1.set_xlim(max_wl, wl_common.min())
    ax1.grid(alpha=0.3)
    ax1.legend()
    ax1.set_title("A) Power Spectrum Density (mean ± spread)", pad = 14)

    # ============================================================
    # ✅ Histogram
    # ============================================================

    ax2.hist(resolutions, bins=40, alpha=0.7)

    ax2.axvline(median_res, color='red', linestyle='--', label='Median')
    ax2.axvspan(p16, p84, color='red', alpha=0.2, label='16–84%')

    ax2.set_xlabel("Resolution [m]")
    ax2.set_ylabel("Count")
    ax2.set_title("B) Resolution Distribution", pad = 14)
    ax2.legend()

    # ============================================================
    # ✅ FINALIZE
    # ============================================================

    plt.tight_layout()
    plt.show()

    # optional save
    out_dir = os.path.join(run_path, 'figures')
    os.makedirs(out_dir, exist_ok=True)

    tag = "train" if use_train_set else "test"
    path = os.path.join(out_dir, f'psd_{tag}_average_full.pdf')

    plt.savefig(path, bbox_inches='tight', dpi=300)
    print(f"Saved figure to {path}")


# ============================================================
# ✅ CLI
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir', type=str)
    parser.add_argument('--dataset_number', type=int, default=1)
    parser.add_argument('--use_train_set', action='store_true')

    args = parser.parse_args()

    run_dir = args.run_dir

#    plot_analysis(
#        dataset_number=args.dataset_number,
#        use_train_set=args.use_train_set
#    )

#    plot_psd_loglog_only(
#        run_dir = run_dir,
#        dataset_number=args.dataset_number,
#        use_train_set=args.use_train_set
#        )
        
    plot_psd_loglog_average(
        run_dir = run_dir,
        use_train_set=args.use_train_set
        )