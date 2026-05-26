import matplotlib.pyplot as plt
import os
import os
import re
import torch
import random
import numpy as np
import subprocess
import shutil

from torch.utils.data import Dataset, DataLoader

from master.models.unet import UNet
from master.train.train_utils import normalize_inputs
from master.train.checkpoints import load_checkpoint, read_file_from_ini
from master.configs.config_utils import load_config_file
from master.train.trainer_core import load_train_objs, prepare_dataloader, DEMDataset, FluidDEMDataset
import glob

from master.lro_data_sim.lro_data_utils_multi_band import detrend_2d, extract_local_subset_all_bands
from master.lro_data_sim.lro_data_utils import extract_local_dem_subset
from master.render.dem_utils import DEM
from master.render.hapke_model import FullHapkeModel
from master.render.camera import Camera
from master.render.renderer import Renderer
from master.lro_data_sim.noise_map_generator import random_blob_field_tile
from master.data_sim.generator import _render_single_image



def ax_format(ax,
           xlim = None, 
           ylim = None, 
           title = None,
           title_pad = 15,
           xlabel = None, 
           ylabel = None, 
           legend = False, 
           grid = False,
           majorlocater = None,
           minorlocater = None,
           n_minors = None,
           aspect_equal = False,
           xy_cross = False):

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if aspect_equal is True:
        ax.set_aspect("equal")
    if grid is True:
        ax.grid(True)
    if xy_cross is True:
        ax.axhline(0, color='black', lw=1)
        ax.axvline(0, color='black', lw=1)
    if majorlocater is not None:
        ax.xaxis.set_major_locator(plt.MultipleLocator(majorlocater[0]))
        ax.yaxis.set_major_locator(plt.MultipleLocator(majorlocater[1]))
    if n_minors is not None:
        ax.xaxis.set_minor_locator(plt.MultipleLocator(majorlocater[0]/(n_minors[0]+1)))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(majorlocater[1]/(n_minors[1]+1)))
    if minorlocater is not None:
        ax.xaxis.set_minor_locator(plt.MultipleLocator(minorlocater[0]))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(minorlocater[1]))
    if legend is True:
        ax.legend()
    if title is not None:
        ax.set_title(title, pad = title_pad)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

# Check if LaTeX is available
def is_latex_available():
    """Check if LaTeX is installed and accessible."""
    # Check for common LaTeX executables
    latex_commands = ['pdflatex', 'latex', 'xelatex']
    for cmd in latex_commands:
        if shutil.which(cmd) is not None:
            return True
    return False

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

def extract_dataset_number(filename):
    """Extract dataset number from filename."""
    match = re.search(r'dataset_(\d+)', filename)
    return int(match.group(1)) if match else None
import matplotlib.patches as patches

import matplotlib.patches as patches

def plot_zoom(dataset_number=1, use_train_set=None, figsize=(13, 10), save_fig=True):

    sup_dir = "runs"
    run_path = os.path.join(sup_dir, run_dir)

    snapshot_path = os.path.join(run_path, 'checkpoints', 'snapshot_best.pt')
    input_stats_path = os.path.join(run_path, 'stats', 'input_stats.ini')
    config_path = os.path.join(run_path, 'stats', 'config.ini')

    config = load_config_file(config_path)
    checkpoint = load_checkpoint(snapshot_path, map_location='cpu')
    input_stats = read_file_from_ini(input_stats_path, ftype=dict)

    dataset_dir = os.path.join(run_path, 'train' if use_train_set else 'test')
    files = sorted(glob.glob(os.path.join(dataset_dir, '*.pt')))
    dataset = DEMDataset(files, config=config)

    sample = dataset[dataset_number]
    images_tensor, _, target_tensor, meta_tensor, w_tensor, theta_bar_tensor, _ = sample

    _, _, _, model, _, _ = load_train_objs(config, run_path)
    model.load_state_dict(checkpoint['MODEL_STATE'])
    model.eval()

    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    model.to(device)

    train_mean = torch.tensor(input_stats['MEAN'], dtype=torch.float32).view(1, -1, 1, 1)
    train_std = torch.tensor(input_stats['STD'], dtype=torch.float32).view(1, -1, 1, 1)

    images_batch = images_tensor.unsqueeze(0)
    images_norm = normalize_inputs(images_batch, train_mean, train_std)

    with torch.no_grad():
        outputs = model(images_norm.to(device), meta_tensor.unsqueeze(0).to(device))

    dem_pred = outputs.squeeze(0)[0].cpu().numpy()
    dem_gt = target_tensor.squeeze().numpy()
    images = images_tensor.numpy()

    H, W = dem_gt.shape


    # ✅ move everything needed for rendering to device
    meta_tensor = meta_tensor.to(device)
    w_tensor = w_tensor.to(device)
    theta_bar_tensor = theta_bar_tensor.to(device)

    dem_obj = DEM(dem_pred, cellsize=1, x0=0, y0=0)
    # ✅ move ALL DEM tensors to device
    if hasattr(dem_obj, "world_points"):
        dem_obj.world_points = dem_obj.world_points.to(device)

    if hasattr(dem_obj, "normals"):
        dem_obj.normals = dem_obj.normals.to(device)

    if hasattr(dem_obj, "normals_flat"):
        dem_obj.normals_flat = dem_obj.normals_flat.to(device)

    if hasattr(dem_obj, "dem"):
        dem_obj.dem = dem_obj.dem.to(device)


    #hapke = FullHapkeModel(w=w_tensor, theta_bar_rad=theta_bar_tensor)
    hapke = FullHapkeModel(w=w_tensor, b=theta_bar_tensor)
    hapke.eval()  # Set to evaluation mode to disable smooth transition at shadow boundaries for rendering
    camera = Camera(image_width=config["IMAGE_W"],
                    image_height=config["IMAGE_H"],
                    focal_length=config["FOCAL_LENGTH"],
                    device=device)
    
    renderer = Renderer(dem_obj, hapke, camera)
    images_pred = []
    

    # Render images + reflectance maps
    for i in range(config["IMAGES_PER_DEM"]):
        params = meta_tensor[i]
        img, reflectance_map = _render_single_image(renderer=renderer, params=params, image_w=config["IMAGE_W"], image_h=config["IMAGE_H"])
        images_pred.append(img)


    # ============================================================
    # ✅ ZOOM CONTROL (EDIT THESE)
    # (cx, cy, size) all in fraction of full DEM
    # ============================================================
    zoom_specs = [
        (0.40, 0.50, 0.1),
        (0.8, 0.8, 0.15),
        (0.4, 0.35, 0.20),
        (0.2, 0.73, 0.15),
        (0.76, 0.055, 0.05),
    ]

    zoom_regions = []

    for cx, cy, size in zoom_specs:

        half_h = int(size * H / 2)
        half_w = int(size * W / 2)

        center_y = int(cy * H)
        center_x = int(cx * W)

        y1 = max(0, center_y - half_h)
        y2 = min(H, center_y + half_h)
        x1 = max(0, center_x - half_w)
        x2 = min(W, center_x + half_w)

        zoom_regions.append((y1, y2, x1, x2))

    # ============================================================
    fig = plt.figure(figsize=figsize)

    # --- Layout tuning (IMPORTANT knobs) ---
    left = 0.06
    full_w = 0.28
    zoom_w = 0.14
    zoom_gap = 0.05   # ✅ increased spacing
    col_gap = 0.03

    row_h_full = 0.28
    row_h_zoom = 0.20
    row_h_img = 0.18

    y_full = 0.68
    y_gt = 0.40 + 0.01
    row_h_zoom = 0.20
    row_gap = 0.01

    y_pred = y_gt - row_h_zoom - row_gap -0.02
    y_img = y_pred - row_h_zoom - row_gap - 0.02
    y_img_pred = y_img - row_h_img - 0.04   # below real images


    # ============================================================
    # ✅ FULL DEMS (ROW 1)
    # ============================================================
    top_total_width = 2 * full_w + col_gap
    top_left = 0.5 - top_total_width / 2  # center horizontally

    ax_gt_full = fig.add_axes([top_left, y_full, full_w, row_h_full])
    ax_pred_full = fig.add_axes([top_left + full_w + col_gap, y_full, full_w, row_h_full])


    vmin_full = min(dem_gt.min(), dem_pred.min())
    vmax_full = max(dem_gt.max(), dem_pred.max())

    ax_gt_full.imshow(dem_gt, cmap='terrain', origin='lower', vmin=vmin_full, vmax=vmax_full)
    ax_pred_full.imshow(dem_pred, cmap='terrain', origin='lower', vmin=vmin_full, vmax=vmax_full)

    im_pred = ax_pred_full.imshow(
                                dem_pred,
                                cmap='terrain',
                                origin='lower',
                                vmin=vmin_full,
                                vmax=vmax_full
    )

    # Add colorbar to the right of predicted DEM
    cbar = fig.colorbar(
        im_pred,
        ax=ax_pred_full,
        fraction=0.046,   # size of colorbar
        pad=0.07          # distance from plot
    )

    cbar.set_label("Elevation [m]")

    ax_gt_full.set_title("Ground Truth DEM", pad=14)
    ax_pred_full.set_title("Predicted DEM", pad=14)

    for ax in [ax_gt_full, ax_pred_full]:
        ax.set_xlabel("x [m]", labelpad=2)
        ax.set_ylabel("y [m]", labelpad=2)
        ax.tick_params(labelsize=8)

    for i, (y1, y2, x1, x2) in enumerate(zoom_regions):

        for ax in [ax_gt_full, ax_pred_full]:

            # --- draw box ---
            rect = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=1.8,
                edgecolor='red',
                facecolor='none'
            )
            ax.add_patch(rect)

            # --- add number label ---
            ax.text(
                x1-0.01, y2,                 # position: top-left corner of box
                f"{i+1}",
                color='red',
                fontsize=10,
                fontweight='bold',
                va='bottom',
                ha='left',
                bbox=dict(
                    facecolor='white',
                    edgecolor='none',
                    alpha=0.7,
                    pad=1
                )
            )
        



    # ============================================================
    # ✅ ZOOMS (ROWS 2 & 3)
    # ============================================================
    n_ticks = 4  # number of ticks per axis (adjust as needed)

    for i, (y1, y2, x1, x2) in enumerate(zoom_regions):

        x_pos = left + i*(zoom_w + zoom_gap)

        gt_patch = dem_gt[y1:y2, x1:x2]
        pred_patch = dem_pred[y1:y2, x1:x2]

        #vmin = min(gt_patch.min(), pred_patch.min())
        #vmax = max(gt_patch.max(), pred_patch.max())


        h = y2 - y1
        w = x2 - x1

        # ✅ define tick positions (same for all)
        xticks = np.linspace(0, w-1, n_ticks).astype(int)
        yticks = np.linspace(0, h-1, n_ticks).astype(int)

        xtick_labels = (x1 + xticks).astype(int)
        ytick_labels = (y1 + yticks).astype(int)

        # --- GT row ---
        ax_gt = fig.add_axes([x_pos, y_gt, zoom_w, row_h_zoom])
        ax_gt.imshow(gt_patch, cmap='terrain', origin='lower', vmin=vmin_full, vmax=vmax_full)
        ax_gt.set_title(f"Zoom {i+1}", fontsize=10, pad=12)
        #ax_gt.set_xlabel("x [m]", fontsize=8, labelpad=2)
        
        ax_gt.set_xticks(xticks)
        ax_gt.set_yticks(yticks)
        ax_gt.set_xticklabels(xtick_labels)
        ax_gt.set_yticklabels(ytick_labels)

        ax_gt.tick_params(labelsize=7, pad=2)

        if i == 0:
            ax_gt.text(
                -0.25, 0.5,          # position (slightly left of axis)
                "Ground Truth\n y [m]",
                transform=ax_gt.transAxes,
                rotation=90,
                va='center',
                ha='center',
                fontsize=9
            )
        #else:
        #    ax_gt.set_yticklabels([])



        # --- Pred row ---
        ax_pred = fig.add_axes([x_pos, y_pred, zoom_w, row_h_zoom])
        ax_pred.imshow(pred_patch, cmap='terrain', origin='lower', vmin=vmin_full, vmax=vmax_full)
        #ax_pred.set_title(f"Pred {i+1}", fontsize=10, pad=6)
        ax_pred.set_xlabel("x [m]", fontsize=8, labelpad=2)

        ax_pred.set_xticks(xticks)
        ax_pred.set_yticks(yticks)
        ax_pred.set_xticklabels(xtick_labels)
        ax_pred.set_yticklabels(ytick_labels)

        ax_pred.tick_params(labelsize=7, pad=2)


        if i == 0:
            ax_pred.text(
                -0.25, 0.5,
                "Prediction\n y [m]",
                transform=ax_pred.transAxes,
                rotation=90,
                va='center',
                ha='center',
                fontsize=9
            )
        #else:
         #   ax_pred.set_yticklabels([])
          #  ax_pred.set_xticklabels([])


    # ============================================================
    # ✅ IMAGE ROW (ROW 4)
    # ============================================================

    img_w = zoom_w
    img_gap = zoom_gap

    for i in range(min(5, images.shape[0])):

        x_pos = left + i*(img_w + img_gap)

        img = images[i]
        h, w = img.shape

        xticks = np.linspace(0, w-1, n_ticks).astype(int)
        yticks = np.linspace(0, h-1, n_ticks).astype(int)

        ax_img = fig.add_axes([x_pos, y_img, img_w, row_h_img])
        ax_img.imshow(img, cmap='gray', origin='lower')

        ax_img.set_xticks(xticks)
        ax_img.set_yticks(yticks)

        ax_img.set_xticklabels(xticks)
        ax_img.set_yticklabels(yticks)

        if i != 0:
            ax_img.set_yticklabels([])
        #    ax_img.set_ylabel("y [pix]", fontsize=8, labelpad=4)
        #else:
        ax_img.set_xticklabels([])

        #ax_img.set_xlabel("x [pix]", fontsize=8, labelpad=2)
        ax_img.tick_params(labelsize=7, pad=2)

        # ✅ Larger rotated row label
        if i == 0:
            ax_img.text(
                -0.25, 0.5,   # pushed further left
                "Ground Truth Images\n y [pix]",
                transform=ax_img.transAxes,
                rotation=90,
                va='center',
                ha='center',
                fontsize=9,   # ✅ larger
            )

    # ============================================================
    # ✅ PREDICTED IMAGE ROW (ROW 5)
    # ============================================================
    for i in range(min(5, len(images_pred))):

        x_pos = left + i*(img_w + img_gap)

        img = images_pred[i]
        h, w = img.shape

        xticks = np.linspace(0, w-1, n_ticks).astype(int)
        yticks = np.linspace(0, h-1, n_ticks).astype(int)

        ax_img_pred = fig.add_axes([x_pos, y_img_pred, img_w, row_h_img])
        ax_img_pred.imshow(img, cmap='gray', origin='lower')

        ax_img_pred.set_xticks(xticks)
        ax_img_pred.set_yticks(yticks)

        ax_img_pred.set_xticklabels(xticks)
        ax_img_pred.set_yticklabels(yticks)

        if i != 0:
            ax_img_pred.set_yticklabels([]) 
        #    ax_img_pred.set_ylabel("y [pix]", fontsize=8, labelpad=4)
        #else:
        #ax_img_pred.set_xticklabels([])

        ax_img_pred.set_xlabel("x [pix]", fontsize=8, labelpad=2)
        ax_img_pred.tick_params(labelsize=7, pad=2)

        # ✅ Larger rotated row label
        if i == 0:
            ax_img_pred.text(
                -0.25, 0.5,
                "Predicted Images \n y [pix]",
                transform=ax_img_pred.transAxes,
                rotation=90,
                va='center',
                ha='center',
                fontsize=9,
            )




    # ============================================================
    # ✅ FINALIZE
    # ============================================================
    plt.subplots_adjust(hspace=0.3)

    if save_fig:
        out_dir = os.path.join(run_path, 'figures')
        os.makedirs(out_dir, exist_ok=True)

        tag = "train" if use_train_set else "test"
        path = os.path.join(out_dir, f'zoom_{tag}_{dataset_number}.pdf')

        plt.savefig(path, bbox_inches='tight', dpi=300)
        print(f"Saved figure to {path}")

    plt.show()

import argparse

if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Plot comprehensive test set predictions for a trained UNet model.")
    args.add_argument('run_dir', nargs="?" , type=str,
                    help='Directory of the trained model run containing stats and checkpoints.')
    args.add_argument('--run_dir', type=str, required=False, dest='run_dir_flag',
                      help='Directory of the trained model run containing stats and checkpoints.')
    args.add_argument('--dataset_number', type=int, required=False, dest='dataset_number', default=1,
                      help='The index for the desired dataset to use in zoom.')
    args.add_argument(
        '--use_train_set',
        action='store_true',
        help="Use the training set for plotting instead of the test set."
    )

    args = args.parse_args()
    
    # Support both positional and flag-based arguments
    run_dir = args.run_dir_flag or args.run_dir
    


    print(f"Plotting zoom set for dataset {args.dataset_number}...")

    config = load_config_file(os.path.join("runs", run_dir, 'stats', 'config.ini'))

    plot_zoom(dataset_number=args.dataset_number, use_train_set=args.use_train_set)

