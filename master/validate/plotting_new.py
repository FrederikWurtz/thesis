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
from master.train.trainer_core import DEMDataset, FluidDEMDataset
from master.train.train_utils import normalize_inputs
from master.train.checkpoints import load_checkpoint, read_file_from_ini
from master.configs.config_utils import load_config_file
from master.train.trainer_new import load_train_objs, prepare_dataloader
import glob


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



def plot_comprehensive_pt(
    run_dir=None,
    n_test_sets=5, figsize=(24, 18), return_fig=False, 
    save_fig=True,
    same_scale=False, variant='random', use_train_set = False, filename=None, test_on_separate_data=False
):

    # Load config to find paths
    sup_dir = "runs/"
    config = load_config_file(os.path.join(sup_dir, run_dir, 'stats', 'config.ini'))
    snapshot_path = os.path.join(sup_dir, run_dir, 'checkpoints', 'snapshot.pt')
    train_losses_path = os.path.join(sup_dir, run_dir, 'checkpoints', 'train_losses.csv')
    val_losses_path = os.path.join(sup_dir, run_dir, 'checkpoints', 'val_losses.csv')
    input_stats_path = os.path.join(sup_dir, run_dir, 'stats', 'input_stats.ini')
    train_timings_path = os.path.join(sup_dir, run_dir, 'checkpoints', 'train_timings.csv')
    val_timings_path = os.path.join(sup_dir, run_dir, 'checkpoints', 'val_timings.csv')
    if not test_on_separate_data:
        test_results_path = os.path.join(sup_dir, run_dir, 'stats', 'test_results.ini')
    else:
        test_results_path = os.path.join(sup_dir, run_dir, 'stats', 'alt_test_results.ini')
    equipment_info_path = os.path.join(sup_dir, run_dir, 'stats', 'equipment_info.ini')
    
    # Load model, history, and test dataset
    checkpoint = load_checkpoint(snapshot_path, map_location='cpu')
    train_data = np.genfromtxt(train_losses_path, delimiter=',', skip_header=1)
    val_data = np.genfromtxt(val_losses_path, delimiter=',', skip_header=1)

    # Handle case where there's only one epoch (1D array instead of 2D)
    if train_data.ndim == 1:
        train_data = train_data.reshape(1, -1)
    if val_data.ndim == 1:
        val_data = val_data.reshape(1, -1)

    train_epochs = train_data[:, 0].astype(int).tolist()
    train_losses = train_data[:, 1].tolist()
    val_epochs = val_data[:, 0].astype(int).tolist()
    val_losses = val_data[:, 1].tolist()
    input_stats = read_file_from_ini(input_stats_path, ftype=dict)
    if os.path.exists(train_timings_path):
        train_timings_data = np.genfromtxt(train_timings_path, delimiter=',', skip_header=1)
        val_timings_data = np.genfromtxt(val_timings_path, delimiter=',', skip_header=1)

        # Handle case where there's only one epoch (1D array instead of 2D)
        if train_timings_data.ndim == 1:
            train_timings_data = train_timings_data.reshape(1, -1)
        if val_timings_data.ndim == 1:
            val_timings_data = val_timings_data.reshape(1, -1)

        avg_train_time_per_epoch = np.mean(train_timings_data[:, 1])
        avg_val_time_per_epoch = np.mean(val_timings_data[:, 1])
        if os.path.exists(test_results_path):
            test_results = read_file_from_ini(test_results_path, ftype=dict)
        else:
            raise FileNotFoundError(f"Found training snapshot, but no test results file found at '{test_results_path}'. Please run testing first.")
        equipment_info = read_file_from_ini(equipment_info_path, ftype=dict)
        n_gpus = int(equipment_info.get('NUM_GPUS', ['1'])[0])

    total_time_logged = False
    total_time_path = os.path.join(sup_dir, run_dir, 'stats', 'total_time.ini')
    if os.path.exists(total_time_path):
        total_time_logged = True
        timing_info = read_file_from_ini(total_time_path, ftype=dict)
        total_time_seconds = timing_info["TOTAL_TIME_SECONDS"]
        total_time_hrs = timing_info["TOTAL_TIME_HRS"]
        total_time_minutes = timing_info["TOTAL_TIME_MINUTES"]

    # run_stats = read_file_from_ini(run_stats_path, ftype=dict)
    # timing_info = read_file_from_ini(timing_info, ftype=dict)
    # lr_changes = read_file_from_ini(lr_changes_path, ftype=dict)
    test_dems_dir = os.path.join(sup_dir, run_dir, 'test')
    print(f"Use train set: {use_train_set}")
    print(f"Use separate test data: {test_on_separate_data}")
    if not use_train_set and not test_on_separate_data:
        test_files = None
        if os.path.isdir(test_dems_dir):
            # Look for .pt files in the test_dems folder
            candidate_files = sorted(glob.glob(os.path.join(test_dems_dir, '*.pt')))
            if len(candidate_files) > 0:
                test_files = candidate_files
                print(f"Using {len(test_files)} test files from: {test_dems_dir}")
            else:
                print(f"Found '{test_dems_dir}' but no .pt files were present. Falling back to history['test_files'].")
        else:
            raise FileNotFoundError(f"No test directory found at '{test_dems_dir}'")

        test_dataset = DEMDataset(test_files)
    
    if use_train_set and not test_on_separate_data:
        run_path = os.path.join(sup_dir, run_dir)

        train_set = FluidDEMDataset(config)
        train_loader = DataLoader(
            train_set,
            batch_size=config["BATCH_SIZE"],
            shuffle=False,
            num_workers=config["NUM_WORKERS_DATALOADER"],
            pin_memory=True,
            prefetch_factor=config["PREFETCH_FACTOR"]
        )
        test_dataset = train_set
    
    if test_on_separate_data:
        alt_test_dir = os.path.join(sup_dir, run_dir, 'alt_test')
        test_files = sorted(glob.glob(os.path.join(alt_test_dir, '*.pt')))
        test_dataset = DEMDataset(test_files)
        print(f"Using {len(test_files)} alternative test files from: {alt_test_dir}")

    model = UNet(in_channels=config["IMAGES_PER_DEM"], out_channels=1)
    model.load_state_dict(checkpoint['MODEL_STATE'])
    model.eval()

    train_mean = torch.tensor(input_stats['MEAN'], dtype=torch.float32)
    train_std = torch.tensor(input_stats['STD'], dtype=torch.float32)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model.to(device)
    
    # Select test sets based on variant
    print(f"Selecting {n_test_sets} test sets using variant: {variant}")
    available_indices = list(range(len(test_dataset)))
    
    if variant == 'first':
        # Use the first n_test_sets from the test pool
        selected_indices = list(range(min(n_test_sets, len(test_dataset))))
    elif variant == 'random':
        # Randomly sample n_test_sets and sort them by index
        selected_indices = sorted(random.sample(available_indices, min(n_test_sets, len(test_dataset))))
        print("Selected test set indices:", selected_indices)
    else:
        raise ValueError(f"Unknown variant: {variant}. Use 'first' or 'random'.")
    
    # ========================================================================
    # First pass: Generate all predictions and find global min/max
    # ========================================================================
    all_targets = []
    all_preds = []
    all_images_norm = []
    
    for test_idx in selected_indices:
        images, reflectance_maps, target, meta = test_dataset[test_idx]
        
        with torch.no_grad():
            images_norm = normalize_inputs(images.unsqueeze(0), train_mean, train_std)
            pred = model(images_norm.to(device), meta.unsqueeze(0).to(device))
            pred = pred.squeeze().cpu().numpy()
        
        target_np = target.squeeze().numpy()
        all_targets.append(target_np)
        all_preds.append(pred)
        all_images_norm.append(images_norm.squeeze(0).cpu().numpy())
    
    # Find global min/max for DEMs (ground truth and predictions)
    global_dem_min = min(min(t.min() for t in all_targets), min(p.min() for p in all_preds))
    global_dem_max = max(max(t.max() for t in all_targets), max(p.max() for p in all_preds))
    
    # Find global min/max for differences
    all_diffs = [all_preds[i] - all_targets[i] for i in range(len(all_preds))]
    global_diff_max = max(abs(d.min()) for d in all_diffs + [abs(d.max()) for d in all_diffs])
    global_diff_min = -global_diff_max  # Symmetric scale for bwr colormap
    
    # Calculate global min/max for images if same_scale='all'
    if same_scale == 'all':
        global_img_min = min(img.min() for img in all_images_norm)
        global_img_max = max(img.max() for img in all_images_norm)
    else:
        global_img_min = None
        global_img_max = None
    
    # ========================================================================
    # Create figure
    # ========================================================================
    fig = plt.figure(figsize=figsize)
    
    # ========================================================================
    # Top row: Loss plot (left) + 4 info boxes (right)
    # ========================================================================
    # Loss plot on the left (50% of the width)
    loss_start_x = 0.13
    loss_width = 0.45  # 50% of usable space
    loss_height = 0.10
    loss_start_y = 0.88
    
    ax_loss = fig.add_axes([loss_start_x, loss_start_y, loss_width, loss_height])

    ax_loss.plot(train_epochs, train_losses, label='Training Loss', linewidth=2)
    ax_loss.plot(val_epochs, val_losses, label='Validation Loss', linewidth=2)
    ax_loss.set_yscale('log')
    ax_loss.set_xlabel('Epoch', fontsize=12)
    ax_loss.set_ylabel('Loss', fontsize=12)
    ax_loss.set_title('Training and Validation Loss', fontsize=14, pad=14)
    ax_loss.legend(fontsize=10, loc='lower left')
    ax_loss.grid(True, alpha=0.3)
    
    # Set x-axis to show only integer ticks
    from matplotlib.ticker import MaxNLocator
    ax_loss.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Position for 2 text boxes in a row on the right (remaining 50%)
    info_width = 0.215  # Wider boxes with minimal gap
    info_height = 0.1   # Height of each text box (taller to fit 10 lines)
    if use_latex:
        info_gap_h = -0.08  # Minimal/overlapping gap between boxes
    else:
        info_gap_h = 0.0  # Minimal/overlapping gap between boxes
    info_start_x = loss_start_x + loss_width + 0.01  # Start after loss plot with small gap
    info_start_y = 0.88  # Starting y position (top row)
    
    fontsize_textbox = 13

    if not os.path.exists(train_timings_path):
        test_results = {"TEST_LOSS": 0.0, "TEST_AME": 0.0}

    infobox_1_text = ("Unet DL Network \n"
                        f"Epochs trained: {int(checkpoint.get('EPOCHS_RUN', 'N/A'))}\n"
                        f"Test Loss: {float(test_results['TEST_LOSS']):.3f}\n"
                        f"Test AME: {float(test_results['TEST_AME']):.3f}\n"
                        f"LR: {float(config['LR']):.2e}"

                        )
    
    # Info box 1: 
    ax_model = fig.add_axes([info_start_x, info_start_y, info_width, info_height])
    ax_model.axis('off')
    ax_model.text(0.05, 0.95, infobox_1_text,
                  transform=ax_model.transAxes,
                  fontsize=fontsize_textbox,
                  verticalalignment='top',
                  family='monospace',
                  bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    

    if not os.path.exists(train_timings_path):
        avg_train_time_per_epoch = 0.0
        avg_val_time_per_epoch = 0.0
        n_gpus = 1

    infobox_2_text = (  
                        f"Total Epochs: {checkpoint.get('EPOCHS_RUN', 'N/A')}\n"
                        f"Train Time Avg.: {avg_train_time_per_epoch:.2f} s/epoch\n"
                        f"Val Time Avg.: {avg_val_time_per_epoch:.2f} s/epoch\n"
                        f"Number GPUs: {n_gpus}\n")

    # Info box 2: 
    ax_training = fig.add_axes([info_start_x + info_width + info_gap_h, info_start_y, info_width, info_height])
    ax_training.axis('off')
    ax_training.text(0.05, 0.95, infobox_2_text,
                     transform=ax_training.transAxes,
                     fontsize=fontsize_textbox,
                     verticalalignment='top',
                     family='monospace',
                     bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    

    
    # ========================================================================
    # Rows 2-6: Test set predictions with manual positioning for tight spacing
    # ========================================================================
    row_height = 0.16  # Increased from 0.13
    row_spacing = 0.013
    start_y = 0.80  # Top of first row
    
    # Column widths and positions (extended to use more horizontal space)
    diff_width = 0.13  # Wider to account for colorbar
    dem_width = 0.11
    pred_width = 0.125  # Wider to account for colorbar
    img_width = 0.11
    
    # Horizontal positions (centered better with equal margins)
    diff_x = 0.07  # Shifted right slightly for better balance
    gt_x = diff_x + diff_width + 0.029  # Small gap after diff
    pred_x = gt_x + dem_width - 0.007   # Small gap between DEMs
    img_x_start = pred_x + pred_width + 0.037  # Small gap before images
    img_gap = 0.0005  # Reduced gap between images
    
    # Store the last image objects for colorbars
    last_diff_im = None
    last_pred_im = None
    
    for row_idx, test_idx in enumerate(selected_indices):
        # Get pre-computed data
        target_np = all_targets[row_idx]
        pred = all_preds[row_idx]
        images_norm = all_images_norm[row_idx]
        diff = pred - target_np
        
        # Calculate y position for this row (bottom of the axes)
        # First row: bottom at (start_y - row_height), subsequent rows below with spacing
        y_pos = (start_y - row_height) - row_idx * (row_height + row_spacing)
        
        # Column 0: Difference (with colorbar using global scale)
        ax_diff = fig.add_axes([diff_x, y_pos, diff_width, row_height])
        im_diff = ax_diff.imshow(diff, cmap='bwr', origin='lower', vmin=global_diff_min, vmax=global_diff_max)
        ax_diff.set_xticks([])
        ax_diff.set_yticks([])
        if row_idx == 0:
            ax_diff.set_title('Diff (Pred-True)', fontsize=12, pad=5)
        ax_diff.set_ylabel(f'Test set {test_idx}', fontsize=12, rotation=90, labelpad=5)
        last_diff_im = im_diff
        
        # Column 1: Ground Truth DEM (using global scale)
        ax_gt = fig.add_axes([gt_x, y_pos, dem_width, row_height])
        ax_gt.imshow(target_np, cmap='terrain', origin='lower', vmin=global_dem_min, vmax=global_dem_max)
        ax_gt.set_xticks([])
        ax_gt.set_yticks([])
        if row_idx == 0:
            ax_gt.set_title('Ground Truth', fontsize=12, pad=5)
        
        # Column 2: Predicted DEM (using global scale, with colorbar)
        ax_pred = fig.add_axes([pred_x, y_pos, pred_width, row_height])
        im_pred = ax_pred.imshow(pred, cmap='terrain', origin='lower', vmin=global_dem_min, vmax=global_dem_max)
        ax_pred.set_xticks([])
        ax_pred.set_yticks([])
        if row_idx == 0:
            ax_pred.set_title('Predicted', fontsize=12, pad=5)
        last_pred_im = im_pred
        
        # Columns 3-7: 5 images with different scaling options
        # Calculate per-row min/max if same_scale='row'
        if same_scale == 'row':
            row_img_min = images_norm.min()
            row_img_max = images_norm.max()
        
        for img_idx in range(5):
            img_x = img_x_start + img_idx * (img_width + img_gap)
            ax_img = fig.add_axes([img_x, y_pos, img_width, row_height])
            
            if same_scale == 'all':
                # Use same scale for all images across all rows
                ax_img.imshow(images_norm[img_idx], 
                             cmap='gray', origin='lower', vmin=global_img_min, vmax=global_img_max)
            elif same_scale == 'row':
                # Use same scale for all images in this row
                ax_img.imshow(images_norm[img_idx], 
                             cmap='gray', origin='lower', vmin=row_img_min, vmax=row_img_max)
            else:  # same_scale=False
                # Use individual scale for each image
                ax_img.imshow(images_norm[img_idx], 
                             cmap='gray', origin='lower')
            
            ax_img.set_xticks([])
            ax_img.set_yticks([])
            if row_idx == 0:
                ax_img.set_title(f'Img {img_idx+1}', fontsize=12, pad=5)
    
    # Calculate total height for colorbar spanning all 5 rows
    total_height = n_test_sets * row_height + (n_test_sets - 1) * row_spacing
    bottom_y = (start_y - row_height) - (n_test_sets - 1) * (row_height + row_spacing)
    
    # Add shared colorbar for differences (positioned at right edge of diff column)
    cbar_diff_width = 0.007
    cbar_diff_x = diff_x + diff_width - 0.008  # Positioned with space from figure
    cbar_diff_ax = fig.add_axes([cbar_diff_x, bottom_y, cbar_diff_width, total_height])
    cbar_diff = plt.colorbar(last_diff_im, cax=cbar_diff_ax)
    cbar_diff.ax.tick_params(labelsize=10)
    
    # Add shared colorbar for DEMs (predicted) (positioned at right edge of pred column)
    cbar_dem_width = 0.007
    cbar_dem_x = pred_x + pred_width - 0.0055  # Positioned with space from figure
    cbar_dem_ax = fig.add_axes([cbar_dem_x, bottom_y, cbar_dem_width, total_height])
    cbar_dem = plt.colorbar(last_pred_im, cax=cbar_dem_ax)
    cbar_dem.ax.tick_params(labelsize=10)


    # Save?
    if save_fig:
        figures_dir = os.path.join(sup_dir, run_dir, 'figures')
        os.makedirs(figures_dir, exist_ok=True)
        output_path = os.path.join(figures_dir, filename if filename is not None else 'comprehensive_plot.pdf')
        fig.savefig(output_path, format='pdf', bbox_inches='tight', dpi=150)
        print(f"Figure saved to: {output_path}")
    
    if return_fig:
        return fig


def plot_comprehensive_multi_band(
    run_dir=None,
    n_test_sets=5,
    figsize=(11*3, 6*3),
    return_fig=False,
    save_fig=True,
    same_scale=False,
    variant='first',
    use_train_set=False,
    filename=None,
    test_on_separate_data=False
):
    sup_dir = "runs/"
    config = load_config_file(os.path.join(sup_dir, run_dir, 'stats', 'config.ini'))
    if not config.get("USE_MULTI_BAND"):
        raise ValueError("plot_comprehensive_multi_band requires a multi-band run directory.")

    snapshot_path = os.path.join(sup_dir, run_dir, 'checkpoints', 'snapshot.pt')
    train_losses_path = os.path.join(sup_dir, run_dir, 'checkpoints', 'train_losses.csv')
    val_losses_path = os.path.join(sup_dir, run_dir, 'checkpoints', 'val_losses.csv')
    input_stats_path = os.path.join(sup_dir, run_dir, 'stats', 'input_stats.ini')
    train_timings_path = os.path.join(sup_dir, run_dir, 'checkpoints', 'train_timings.csv')
    val_timings_path = os.path.join(sup_dir, run_dir, 'checkpoints', 'val_timings.csv')
    if not test_on_separate_data:
        test_results_path = os.path.join(sup_dir, run_dir, 'stats', 'test_results.ini')
    else:
        test_results_path = os.path.join(sup_dir, run_dir, 'stats', 'alt_test_results.ini')
    equipment_info_path = os.path.join(sup_dir, run_dir, 'stats', 'equipment_info.ini')

    checkpoint = load_checkpoint(snapshot_path, map_location='cpu')
    train_data = np.genfromtxt(train_losses_path, delimiter=',', skip_header=1)
    val_data = np.genfromtxt(val_losses_path, delimiter=',', skip_header=1)

    if train_data.ndim == 1:
        train_data = train_data.reshape(1, -1)
    if val_data.ndim == 1:
        val_data = val_data.reshape(1, -1)

    train_epochs = train_data[:, 0].astype(int).tolist()
    train_losses = train_data[:, 1].tolist()
    val_epochs = val_data[:, 0].astype(int).tolist()
    val_losses = val_data[:, 1].tolist()
    # check for extreme outliers in first 10 data points 
    first_10_val_losses = val_losses[:10]
    mean_first_10 = np.mean(first_10_val_losses)
    for i, loss in enumerate(first_10_val_losses):
        if loss > 5 * mean_first_10:
            print(f"Warning: Extreme outlier, more than 5 times the mean of first 10 epochs detected in validation loss at epoch {val_epochs[i]}: {loss}. Removing from validation losses.")
            val_losses.pop(i)
            val_epochs.pop(i)
    
    input_stats = read_file_from_ini(input_stats_path, ftype=dict)

    if os.path.exists(train_timings_path):
        train_timings_data = np.genfromtxt(train_timings_path, delimiter=',', skip_header=1)
        val_timings_data = np.genfromtxt(val_timings_path, delimiter=',', skip_header=1)

        if train_timings_data.ndim == 1:
            train_timings_data = train_timings_data.reshape(1, -1)
        if val_timings_data.ndim == 1:
            val_timings_data = val_timings_data.reshape(1, -1)

        avg_train_time_per_epoch = np.mean(train_timings_data[:, 1])
        avg_val_time_per_epoch = np.mean(val_timings_data[:, 1])
        if os.path.exists(test_results_path):
            test_results = read_file_from_ini(test_results_path, ftype=dict)
        else:
            raise FileNotFoundError(f"Found training snapshot, but no test results file found at '{test_results_path}'. Please run testing first.")
        equipment_info = read_file_from_ini(equipment_info_path, ftype=dict)
        n_gpus = int(equipment_info.get('NUM_GPUS', ['1'])[0])
    else:
        test_results = {"TEST_LOSS": 0.0, "TEST_AME": 0.0}
        avg_train_time_per_epoch = 0.0
        avg_val_time_per_epoch = 0.0
        n_gpus = 1

    total_time_logged = False
    total_time_path = os.path.join(sup_dir, run_dir, 'stats', 'total_time.ini')
    if os.path.exists(total_time_path):
        total_time_logged = True
        timing_info = read_file_from_ini(total_time_path, ftype=dict)
        total_time_seconds = float(timing_info["TOTAL_TIME_SECONDS"])
        total_time_hours = float(timing_info["TOTAL_TIME_HRS"])
        total_time_minutes = float(timing_info["TOTAL_TIME_MINUTES"])

    print(f"Use train set: {use_train_set}")
    print(f"Use separate test data: {test_on_separate_data}")
    if use_train_set and test_on_separate_data:
        raise ValueError("Cannot use both training set and separate test data simultaneously.")

    if not use_train_set and not test_on_separate_data:
        test_dir = os.path.join(sup_dir, run_dir, 'test')
        if not os.path.isdir(test_dir):
            raise FileNotFoundError(f"No test directory found at '{test_dir}'")
        candidate_files = sorted(glob.glob(os.path.join(test_dir, '*.pt')))
        if len(candidate_files) == 0:
            raise FileNotFoundError(f"Found '{test_dir}' but no .pt files were present.")
        test_dataset = DEMDataset(candidate_files, config=config)
    elif use_train_set:
        run_path = os.path.join(sup_dir, run_dir)
        train_set = FluidDEMDataset(config=config)
        test_dataset = train_set
    else:
        alt_test_dir = os.path.join(sup_dir, run_dir, 'alt_test')
        candidate_files = sorted(glob.glob(os.path.join(alt_test_dir, '*.pt')))
        if len(candidate_files) == 0:
            raise FileNotFoundError(f"No .pt files found in alternate test directory '{alt_test_dir}'.")
        test_dataset = DEMDataset(candidate_files, config=config)

    model = UNet(
        in_channels=config["IMAGES_PER_DEM"],
        out_channels=3,
        w_range=(config["W_MIN"], config["W_MAX"]),
        theta_range=(config["THETA_BAR_MIN"], config["THETA_BAR_MAX"]),
    )
    model.load_state_dict(checkpoint['MODEL_STATE'])
    model.eval()

    train_mean = torch.tensor(input_stats['MEAN'], dtype=torch.float32)
    train_std = torch.tensor(input_stats['STD'], dtype=torch.float32)

    if train_mean.ndim == 0:
        train_mean_view = train_mean.view(1, 1, 1, 1)
    elif train_mean.ndim == 1:
        train_mean_view = train_mean.view(1, -1, 1, 1)
    else:
        train_mean_view = train_mean

    if train_std.ndim == 0:
        train_std_view = train_std.view(1, 1, 1, 1)
    elif train_std.ndim == 1:
        train_std_view = train_std.view(1, -1, 1, 1)
    else:
        train_std_view = train_std

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    model.to(device)

    print(f"Selecting {n_test_sets} test sets using variant: {variant}")
    available_indices = list(range(len(test_dataset)))
    if len(available_indices) == 0:
        raise ValueError("No samples available in the selected dataset.")

    if variant == 'first':
        selected_indices = list(range(min(n_test_sets, len(test_dataset))))
    elif variant == 'random':
        selected_indices = sorted(random.sample(available_indices, min(n_test_sets, len(test_dataset))))
        print("Selected test set indices:", selected_indices)
    else:
        raise ValueError(f"Unknown variant: {variant}. Use 'first' or 'random'.")

    all_dem_targets = []
    all_dem_preds = []
    all_w_targets = []
    all_w_preds = []
    all_theta_targets = []
    all_theta_preds = []
    all_images_norm = []
    index_labels = []

    for test_idx in selected_indices:
        sample = test_dataset[test_idx]
        if len(sample) == 7:
            images_tensor, reflectance_maps_tensor, target_tensor, meta_tensor, w_tensor, theta_tensor, _ = sample
        else:
            images_tensor, reflectance_maps_tensor, target_tensor, meta_tensor, w_tensor, theta_tensor = sample

        images_tensor = images_tensor.float()
        meta_tensor = meta_tensor.float()
        target_tensor = target_tensor.float()
        w_tensor = w_tensor.float()
        theta_tensor = theta_tensor.float()

        images_batch = images_tensor.unsqueeze(0)
        images_norm = normalize_inputs(images_batch, train_mean_view, train_std_view)
        meta_batch = meta_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(images_norm.to(device), meta_batch, target_size=target_tensor.shape[-2:])

        outputs_np = outputs.squeeze(0).cpu().numpy()
        dem_pred = outputs_np[0]
        w_pred = outputs_np[1]
        theta_pred = outputs_np[2]

        dem_target = target_tensor.squeeze().numpy()
        w_target = w_tensor.squeeze().numpy()
        theta_target = theta_tensor.squeeze().numpy()

        all_dem_targets.append(dem_target)
        all_dem_preds.append(dem_pred)
        all_w_targets.append(w_target)
        all_w_preds.append(w_pred)
        all_theta_targets.append(theta_target)
        all_theta_preds.append(theta_pred)
        all_images_norm.append(images_norm.squeeze(0).cpu().numpy())
        index_labels.append(test_idx)

    if len(selected_indices) == 0:
        raise ValueError("No test indices selected for plotting.")

    global_dem_min = min(min(d.min() for d in all_dem_targets), min(d.min() for d in all_dem_preds))
    global_dem_max = max(max(d.max() for d in all_dem_targets), max(d.max() for d in all_dem_preds))
    global_w_min = min(min(w.min() for w in all_w_targets), min(w.min() for w in all_w_preds))
    global_w_max = max(max(w.max() for w in all_w_targets), max(w.max() for w in all_w_preds))
    global_theta_min = min(min(t.min() for t in all_theta_targets), min(t.min() for t in all_theta_preds))
    global_theta_max = max(max(t.max() for t in all_theta_targets), max(t.max() for t in all_theta_preds))

    if same_scale == 'all':
        global_img_min = min(img.min() for img in all_images_norm)
        global_img_max = max(img.max() for img in all_images_norm)
    else:
        global_img_min = None
        global_img_max = None

    fig = plt.figure(figsize=figsize)

    # Top row layout
    loss_start_x = 0.10
    loss_width = 0.44
    loss_height = 0.25
    loss_start_y = 0.88

    ax_loss = fig.add_axes([loss_start_x, loss_start_y, loss_width, loss_height])
    ax_loss.plot(train_epochs, train_losses, label='Training Loss', linewidth=2)
    ax_loss.plot(val_epochs, val_losses, label='Validation Loss', linewidth=2)
    ax_loss.set_yscale('log')
    ax_loss.set_xlabel('Epoch', fontsize=12)
    ax_loss.set_ylabel('Loss', fontsize=12)
    ax_loss.set_title('Training and Validation Loss', fontsize=14, pad=14)
    ax_loss.legend(fontsize=10, loc='lower left')
    ax_loss.grid(True, alpha=0.3)

    from matplotlib.ticker import MaxNLocator
    ax_loss.xaxis.set_major_locator(MaxNLocator(integer=True))

    info_width = 0.12
    info_height = 0.25
    info_gap_h = 0.0001
    info_start_x = loss_start_x + loss_width + 0.005
    info_start_y = loss_start_y
    fontsize_textbox = 16
    
    theta_min_rad = float(config["THETA_BAR_MIN"])
    theta_max_rad = float(config["THETA_BAR_MAX"])
    theta_min_deg = np.degrees(theta_min_rad)
    theta_max_deg = np.degrees(theta_max_rad)

    infobox_entries = [
        (
            "Multi-band UNet\n"
            f"Epochs trained: {int(checkpoint.get('EPOCHS_RUN', 'N/A'))}\n"
            f"Test Loss: {float(test_results['TEST_LOSS']):.3f}\n"
            f"Test DEM AME: {float(test_results['DEM_AME']):.3f}\n"
            f"Test W AME: {float(test_results['W_AME']):.3f}\n"
            f"Test Theta AME: {float(test_results['THETA_AME']):.3f}\n"
            f"LR: {float(config['LR']):.2e}",
            dict(boxstyle='round', facecolor='lightblue', alpha=0.3),
        ),
        (
            f"Total Epochs: {checkpoint.get('EPOCHS_RUN', 'N/A')}\n"
            f"Train Time Avg.: \n"
            f"{avg_train_time_per_epoch:.2f} s/epoch\n"
            f"Val Time Avg.: \n"
            f"{avg_val_time_per_epoch:.2f} s/epoch\n"
            f"Number GPUs: {n_gpus}",
            dict(boxstyle='round', facecolor='lightgreen', alpha=0.3),
        ),
        (
            "Multi-band Config\n"
            f"Albedo:\n"
            f"w: {float(config['W_MIN']):.3f}–{float(config['W_MAX']):.3f}\n"
            f"w blobs: r={config['W_BLOB_RADIUS_PX']} px, ρ={float(config['W_BLOB_DENSITY']):.2f}\n"
            f"Macroscopic Roughness:\n"
            rf"$\bar{{\theta}}$: {theta_min_rad:.1f}–{theta_max_rad:.1f} rad" "\n"
            rf"$\bar{{\theta}}$: {theta_min_deg:.1f}–{theta_max_deg:.1f}°" "\n"
            rf"$\bar{{\theta}}$ blobs: r={config['THETA_BAR_BLOB_RADIUS_PX']} px, ρ={float(config['THETA_BAR_BLOB_DENSITY']):.2f}",
            dict(boxstyle='round', facecolor='lavender', alpha=0.3),
        ),
    ]

        
    if not config["USE_SEMIFLUID"]: 
        infobox_entries.append(
            (
            "Training Config\n"
            f"Batch Size: {config['BATCH_SIZE']}\n"
            f"Save every: {config['SAVE_EVERY']} epochs\n"
            f"Use semifluid: {config['USE_SEMIFLUID']}\n",
            dict(boxstyle='round', facecolor='lightyellow', alpha=0.3)
            )
    )
    else:  
        infobox_entries.append(
            (
                f"Training Config\n"
                f"Batch Size: {config['BATCH_SIZE']}\n"
                f"Save every: {config['SAVE_EVERY']} epochs\n"
                f"Use semifluid: {config['USE_SEMIFLUID']}\n"
                f"New data every: {config['NEW_FLUID_DATA_EVERY']} epochs\n",
                dict(boxstyle='round', facecolor='lightyellow', alpha=0.3)
            )
        )
        
    if total_time_logged:
        infobox_entries.append(
            (
                f"Total Train Time:\n"
                f"{total_time_hours:.1f} h {total_time_minutes:.1f} m\n"
                f"{total_time_seconds:.1f} s total",
                dict(boxstyle='round', facecolor='lightyellow', alpha=0.3),
            )
        )

    for idx, (text, bbox_kwargs) in enumerate(infobox_entries):
        ax_box = fig.add_axes([info_start_x + idx * (info_width + info_gap_h), info_start_y, info_width, info_height])
        ax_box.axis('off')
        ax_box.text(
            0.05,
            0.95,
            text,
            transform=ax_box.transAxes,
            fontsize=fontsize_textbox,
            verticalalignment='top',
            family='monospace',
            bbox=bbox_kwargs,
        )

    # Data rows layout (5 datasets by default)
    row_height = 0.13
    row_spacing = 0.012
    start_y = 0.83

    band_width = 0.075
    band_gap = 0.002
    images_width = 0.070
    images_gap = 0.005
    start_x = 0.06
    
    color_bar_extra_space = 0.03
    space_before_color_bar = 0.009

    band_positions = []
    current_x = start_x
    for _ in range(6):
        band_positions.append(current_x)
        current_x += band_width + band_gap
        if _ in [1, 3, 5]:  # After predicted w and predicted theta
            current_x += color_bar_extra_space
    current_x += 0.005 # Extra gap before images

    image_positions = []
    for _ in range(5):
        image_positions.append(current_x)
        current_x += images_width + images_gap

    last_w_im = None
    last_theta_im = None
    last_dem_im = None

    for row_idx, test_idx in enumerate(selected_indices):
        dem_target = all_dem_targets[row_idx]
        dem_pred = all_dem_preds[row_idx]
        w_target = all_w_targets[row_idx]
        w_pred = all_w_preds[row_idx]
        theta_target = all_theta_targets[row_idx]
        theta_pred = all_theta_preds[row_idx]
        images_norm = all_images_norm[row_idx]

        if same_scale == 'row':
            row_img_min = images_norm.min()
            row_img_max = images_norm.max()

        y_pos = (start_y - row_height) - row_idx * (row_height + row_spacing)

        ax_gt_w = fig.add_axes([band_positions[0], y_pos, band_width, row_height])
        ax_gt_w.imshow(w_target, cmap='viridis', origin='lower', vmin=global_w_min, vmax=global_w_max)
        ax_gt_w.set_xticks([])
        ax_gt_w.set_yticks([])
        if row_idx == 0:
            ax_gt_w.set_title('GT w', fontsize=12, pad=5)
        ax_gt_w.set_ylabel(f'Set {index_labels[row_idx]}', fontsize=11, rotation=90, labelpad=5)

        ax_pred_w = fig.add_axes([band_positions[1], y_pos, band_width, row_height])
        last_w_im = ax_pred_w.imshow(w_pred, cmap='viridis', origin='lower', vmin=global_w_min, vmax=global_w_max)
        ax_pred_w.set_xticks([])
        ax_pred_w.set_yticks([])
        if row_idx == 0:
            ax_pred_w.set_title('Pred w', fontsize=12, pad=5)

        ax_gt_theta = fig.add_axes([band_positions[2], y_pos, band_width, row_height])
        ax_gt_theta.imshow(theta_target, cmap='plasma', origin='lower', vmin=global_theta_min, vmax=global_theta_max)
        ax_gt_theta.set_xticks([])
        ax_gt_theta.set_yticks([])
        if row_idx == 0:
            ax_gt_theta.set_title('GT theta', fontsize=12, pad=5)

        ax_pred_theta = fig.add_axes([band_positions[3], y_pos, band_width, row_height])
        last_theta_im = ax_pred_theta.imshow(theta_pred, cmap='plasma', origin='lower', vmin=global_theta_min, vmax=global_theta_max)
        ax_pred_theta.set_xticks([])
        ax_pred_theta.set_yticks([])
        if row_idx == 0:
            ax_pred_theta.set_title('Pred theta', fontsize=12, pad=5)

        ax_gt_dem = fig.add_axes([band_positions[4], y_pos, band_width, row_height])
        ax_gt_dem.imshow(dem_target, cmap='terrain', origin='lower', vmin=global_dem_min, vmax=global_dem_max)
        ax_gt_dem.set_xticks([])
        ax_gt_dem.set_yticks([])
        if row_idx == 0:
            ax_gt_dem.set_title('GT DEM', fontsize=12, pad=5)

        ax_pred_dem = fig.add_axes([band_positions[5], y_pos, band_width, row_height])
        last_dem_im = ax_pred_dem.imshow(dem_pred, cmap='terrain', origin='lower', vmin=global_dem_min, vmax=global_dem_max)
        ax_pred_dem.set_xticks([])
        ax_pred_dem.set_yticks([])
        if row_idx == 0:
            ax_pred_dem.set_title('Pred DEM', fontsize=12, pad=5)

        for img_idx, img_x in enumerate(image_positions):
            ax_img = fig.add_axes([img_x, y_pos, images_width, row_height])

            if same_scale == 'all':
                ax_img.imshow(images_norm[img_idx], cmap='gray', origin='lower', vmin=global_img_min, vmax=global_img_max)
            elif same_scale == 'row':
                ax_img.imshow(images_norm[img_idx], cmap='gray', origin='lower', vmin=row_img_min, vmax=row_img_max)
            else:
                ax_img.imshow(images_norm[img_idx], cmap='gray', origin='lower')

            ax_img.set_xticks([])
            ax_img.set_yticks([])
            if row_idx == 0:
                ax_img.set_title(f'Img {img_idx+1}', fontsize=12, pad=5)

    n_rows = len(selected_indices)
    total_height = n_rows * row_height + max(0, (n_rows - 1)) * row_spacing
    bottom_y = (start_y - row_height) - max(0, (n_rows - 1)) * (row_height + row_spacing)

    if last_w_im is not None:
        cbar_w_ax = fig.add_axes([band_positions[1] + band_width - 0.004 + space_before_color_bar, bottom_y, 0.006, total_height])
        cbar_w = plt.colorbar(last_w_im, cax=cbar_w_ax)
        cbar_w.ax.tick_params(labelsize=10)
        cbar_w.set_label('w', fontsize=10)

    if last_theta_im is not None:
        cbar_theta_ax = fig.add_axes([band_positions[3] + band_width - 0.004 + space_before_color_bar, bottom_y, 0.006, total_height])
        cbar_theta = plt.colorbar(last_theta_im, cax=cbar_theta_ax)
        cbar_theta.ax.tick_params(labelsize=10)
        cbar_theta.set_label('theta (rad)', fontsize=10)

    if last_dem_im is not None:
        cbar_dem_ax = fig.add_axes([band_positions[5] + band_width - 0.004 + space_before_color_bar, bottom_y, 0.006, total_height])
        cbar_dem = plt.colorbar(last_dem_im, cax=cbar_dem_ax)
        cbar_dem.ax.tick_params(labelsize=10)
        cbar_dem.set_label('DEM (m)', fontsize=10)

    if save_fig:
        figures_dir = os.path.join(sup_dir, run_dir, 'figures')
        os.makedirs(figures_dir, exist_ok=True)
        output_path = os.path.join(figures_dir, filename if filename is not None else 'comprehensive_plot_multi_band.pdf')
        fig.savefig(output_path, format='pdf', bbox_inches='tight', dpi=150)
        print(f"Figure saved to: {output_path}")

    if return_fig:
        return fig


def plot_comprehensive_multi_band_with_dif(
    run_dir=None,
    n_test_sets=5,
    figsize=(14*3, 6*3),
    return_fig=False,
    save_fig=True,
    same_scale=False,
    variant='first',
    use_train_set=False,
    filename=None,
    test_on_separate_data=False
):
    sup_dir = "runs/"
    config = load_config_file(os.path.join(sup_dir, run_dir, 'stats', 'config.ini'))
    if not config.get("USE_MULTI_BAND"):
        raise ValueError("plot_comprehensive_multi_band_with_dif requires a multi-band run directory.")

    snapshot_path = os.path.join(sup_dir, run_dir, 'checkpoints', 'snapshot.pt')
    train_losses_path = os.path.join(sup_dir, run_dir, 'checkpoints', 'train_losses.csv')
    val_losses_path = os.path.join(sup_dir, run_dir, 'checkpoints', 'val_losses.csv')
    input_stats_path = os.path.join(sup_dir, run_dir, 'stats', 'input_stats.ini')
    train_timings_path = os.path.join(sup_dir, run_dir, 'checkpoints', 'train_timings.csv')
    val_timings_path = os.path.join(sup_dir, run_dir, 'checkpoints', 'val_timings.csv')
    if not test_on_separate_data:
        test_results_path = os.path.join(sup_dir, run_dir, 'stats', 'test_results.ini')
    else:
        test_results_path = os.path.join(sup_dir, run_dir, 'stats', 'alt_test_results.ini')
    equipment_info_path = os.path.join(sup_dir, run_dir, 'stats', 'equipment_info.ini')

    checkpoint = load_checkpoint(snapshot_path, map_location='cpu')
    train_data = np.genfromtxt(train_losses_path, delimiter=',', skip_header=1)
    val_data = np.genfromtxt(val_losses_path, delimiter=',', skip_header=1)

    if train_data.ndim == 1:
        train_data = train_data.reshape(1, -1)
    if val_data.ndim == 1:
        val_data = val_data.reshape(1, -1)

    train_epochs = train_data[:, 0].astype(int).tolist()
    train_losses = train_data[:, 1].tolist()
    val_epochs = val_data[:, 0].astype(int).tolist()
    val_losses = val_data[:, 1].tolist()
    # check for extreme outliers in first 10 data points 
    first_10_val_losses = val_losses[:10]
    mean_first_10 = np.mean(first_10_val_losses)
    for i, loss in enumerate(first_10_val_losses):
        if loss > 5 * mean_first_10:
            print(f"Warning: Extreme outlier, more than 5 times the mean of first 10 epochs detected in validation loss at epoch {val_epochs[i]}: {loss}. Removing from validation losses.")
            val_losses.pop(i)
            val_epochs.pop(i)
    
    input_stats = read_file_from_ini(input_stats_path, ftype=dict)

    if os.path.exists(train_timings_path):
        train_timings_data = np.genfromtxt(train_timings_path, delimiter=',', skip_header=1)
        val_timings_data = np.genfromtxt(val_timings_path, delimiter=',', skip_header=1)

        if train_timings_data.ndim == 1:
            train_timings_data = train_timings_data.reshape(1, -1)
        if val_timings_data.ndim == 1:
            val_timings_data = val_timings_data.reshape(1, -1)

        avg_train_time_per_epoch = np.mean(train_timings_data[:, 1])
        avg_val_time_per_epoch = np.mean(val_timings_data[:, 1])
        if os.path.exists(test_results_path):
            test_results = read_file_from_ini(test_results_path, ftype=dict)
        else:
            raise FileNotFoundError(f"Found training snapshot, but no test results file found at '{test_results_path}'. Please run testing first.")
        equipment_info = read_file_from_ini(equipment_info_path, ftype=dict)
        n_gpus = int(equipment_info.get('NUM_GPUS', ['1'])[0])
    else:
        test_results = {"TEST_LOSS": 0.0, "TEST_AME": 0.0}
        avg_train_time_per_epoch = 0.0
        avg_val_time_per_epoch = 0.0
        n_gpus = 1

    total_time_logged = False
    total_time_path = os.path.join(sup_dir, run_dir, 'stats', 'total_time.ini')
    if os.path.exists(total_time_path):
        total_time_logged = True
        timing_info = read_file_from_ini(total_time_path, ftype=dict)
        total_time_seconds = float(timing_info["TOTAL_TIME_SECONDS"])
        total_time_hours = float(timing_info["TOTAL_TIME_HRS"])
        total_time_minutes = float(timing_info["TOTAL_TIME_MINUTES"])

    print(f"Use train set: {use_train_set}")
    print(f"Use separate test data: {test_on_separate_data}")
    if use_train_set and test_on_separate_data:
        raise ValueError("Cannot use both training set and separate test data simultaneously.")

    if not use_train_set and not test_on_separate_data:
        test_dir = os.path.join(sup_dir, run_dir, 'test')
        if not os.path.isdir(test_dir):
            raise FileNotFoundError(f"No test directory found at '{test_dir}'")
        candidate_files = sorted(glob.glob(os.path.join(test_dir, '*.pt')))
        if len(candidate_files) == 0:
            raise FileNotFoundError(f"Found '{test_dir}' but no .pt files were present.")
        test_dataset = DEMDataset(candidate_files, config=config)
    elif use_train_set:
        run_path = os.path.join(sup_dir, run_dir)
        train_set = FluidDEMDataset(config=config)
        test_dataset = train_set
    else:
        alt_test_dir = os.path.join(sup_dir, run_dir, 'alt_test')
        candidate_files = sorted(glob.glob(os.path.join(alt_test_dir, '*.pt')))
        if len(candidate_files) == 0:
            raise FileNotFoundError(f"No .pt files found in alternate test directory '{alt_test_dir}'.")
        test_dataset = DEMDataset(candidate_files, config=config)

    model = UNet(
        in_channels=config["IMAGES_PER_DEM"],
        out_channels=3,
        w_range=(config["W_MIN"], config["W_MAX"]),
        theta_range=(config["THETA_BAR_MIN"], config["THETA_BAR_MAX"]),
    )
    model.load_state_dict(checkpoint['MODEL_STATE'])
    model.eval()

    train_mean = torch.tensor(input_stats['MEAN'], dtype=torch.float32)
    train_std = torch.tensor(input_stats['STD'], dtype=torch.float32)

    if train_mean.ndim == 0:
        train_mean_view = train_mean.view(1, 1, 1, 1)
    elif train_mean.ndim == 1:
        train_mean_view = train_mean.view(1, -1, 1, 1)
    else:
        train_mean_view = train_mean

    if train_std.ndim == 0:
        train_std_view = train_std.view(1, 1, 1, 1)
    elif train_std.ndim == 1:
        train_std_view = train_std.view(1, -1, 1, 1)
    else:
        train_std_view = train_std

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    model.to(device)

    print(f"Selecting {n_test_sets} test sets using variant: {variant}")
    available_indices = list(range(len(test_dataset)))
    if len(available_indices) == 0:
        raise ValueError("No samples available in the selected dataset.")

    if variant == 'first':
        selected_indices = list(range(min(n_test_sets, len(test_dataset))))
    elif variant == 'random':
        selected_indices = sorted(random.sample(available_indices, min(n_test_sets, len(test_dataset))))
        print("Selected test set indices:", selected_indices)
    else:
        raise ValueError(f"Unknown variant: {variant}. Use 'first' or 'random'.")

    all_dem_targets = []
    all_dem_preds = []
    all_w_targets = []
    all_w_preds = []
    all_theta_targets = []
    all_theta_preds = []
    all_images_norm = []
    index_labels = []

    for test_idx in selected_indices:
        sample = test_dataset[test_idx]
        if len(sample) == 7:
            images_tensor, reflectance_maps_tensor, target_tensor, meta_tensor, w_tensor, theta_tensor, _ = sample
        else:
            images_tensor, reflectance_maps_tensor, target_tensor, meta_tensor, w_tensor, theta_tensor = sample

        images_tensor = images_tensor.float()
        meta_tensor = meta_tensor.float()
        target_tensor = target_tensor.float()
        w_tensor = w_tensor.float()
        theta_tensor = theta_tensor.float()

        images_batch = images_tensor.unsqueeze(0)
        images_norm = normalize_inputs(images_batch, train_mean_view, train_std_view)
        meta_batch = meta_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(images_norm.to(device), meta_batch, target_size=target_tensor.shape[-2:])

        outputs_np = outputs.squeeze(0).cpu().numpy()
        dem_pred = outputs_np[0]
        w_pred = outputs_np[1]
        theta_pred = outputs_np[2]

        dem_target = target_tensor.squeeze().numpy()
        w_target = w_tensor.squeeze().numpy()
        theta_target = theta_tensor.squeeze().numpy()

        all_dem_targets.append(dem_target)
        all_dem_preds.append(dem_pred)
        all_w_targets.append(w_target)
        all_w_preds.append(w_pred)
        all_theta_targets.append(theta_target)
        all_theta_preds.append(theta_pred)
        all_images_norm.append(images_norm.squeeze(0).cpu().numpy())
        index_labels.append(test_idx)

    if len(selected_indices) == 0:
        raise ValueError("No test indices selected for plotting.")

    global_dem_min = min(min(d.min() for d in all_dem_targets), min(d.min() for d in all_dem_preds))
    global_dem_max = max(max(d.max() for d in all_dem_targets), max(d.max() for d in all_dem_preds))
    global_w_min = min(min(w.min() for w in all_w_targets), min(w.min() for w in all_w_preds))
    global_w_max = max(max(w.max() for w in all_w_targets), max(w.max() for w in all_w_preds))
    global_theta_min = min(min(t.min() for t in all_theta_targets), min(t.min() for t in all_theta_preds))
    global_theta_max = max(max(t.max() for t in all_theta_targets), max(t.max() for t in all_theta_preds))

    # Compute difference min/max for consistent coloring of difference maps
    all_dem_diffs = [gt - pred for gt, pred in zip(all_dem_targets, all_dem_preds)]
    all_w_diffs = [gt - pred for gt, pred in zip(all_w_targets, all_w_preds)]
    all_theta_diffs = [gt - pred for gt, pred in zip(all_theta_targets, all_theta_preds)]
    
    global_dem_dif_min = min(d.min() for d in all_dem_diffs)
    global_dem_dif_max = max(d.max() for d in all_dem_diffs)
    global_w_dif_min = min(d.min() for d in all_w_diffs)
    global_w_dif_max = max(d.max() for d in all_w_diffs)
    global_theta_dif_min = min(d.min() for d in all_theta_diffs)
    global_theta_dif_max = max(d.max() for d in all_theta_diffs)

    if same_scale == 'all':
        global_img_min = min(img.min() for img in all_images_norm)
        global_img_max = max(img.max() for img in all_images_norm)
    else:
        global_img_min = None
        global_img_max = None

    fig = plt.figure(figsize=figsize)

    # Top row layout
    loss_start_x = 0.08
    loss_width = 0.6
    loss_height = 0.2
    loss_start_y = 0.88

    ax_loss = fig.add_axes([loss_start_x, loss_start_y, loss_width, loss_height])
    ax_loss.plot(train_epochs, train_losses, label='Training Loss', linewidth=2)
    ax_loss.plot(val_epochs, val_losses, label='Validation Loss', linewidth=2)
    ax_loss.set_yscale('log')
    ax_loss.set_xlabel('Epoch', fontsize=12)
    ax_loss.set_ylabel('Loss', fontsize=12)
    ax_loss.set_title('Training and Validation Loss', fontsize=14, pad=14)
    ax_loss.legend(fontsize=10, loc='lower left')
    ax_loss.grid(True, alpha=0.3)

    from matplotlib.ticker import MaxNLocator
    ax_loss.xaxis.set_major_locator(MaxNLocator(integer=True))

    info_width = 0.1
    info_height = 0.2
    info_gap_h = 0.0001
    info_start_x = loss_start_x + loss_width + 0.005
    info_start_y = loss_start_y
    fontsize_textbox = 16
    
    theta_min_rad = float(config["THETA_BAR_MIN"])
    theta_max_rad = float(config["THETA_BAR_MAX"])
    theta_min_deg = np.degrees(theta_min_rad)
    theta_max_deg = np.degrees(theta_max_rad)

    infobox_entries = [
        (
            "Multi-band UNet\n"
            f"Epochs trained: {int(checkpoint.get('EPOCHS_RUN', 'N/A'))}\n"
            f"Test Loss: {float(test_results['TEST_LOSS']):.3f}\n"
            f"Test DEM AME: {float(test_results['DEM_AME']):.3f}\n"
            f"Test W AME: {float(test_results['W_AME']):.3f}\n"
            f"Test Theta AME: {float(test_results['THETA_AME']):.3f}\n"
            f"LR: {float(config['LR']):.2e}",
            dict(boxstyle='round', facecolor='lightblue', alpha=0.3),
        ),
        (
            f"Total Epochs: {checkpoint.get('EPOCHS_RUN', 'N/A')}\n"
            f"Train Time Avg.: \n"
            f"{avg_train_time_per_epoch:.2f} s/epoch\n"
            f"Val Time Avg.: \n"
            f"{avg_val_time_per_epoch:.2f} s/epoch\n"
            f"Number GPUs: {n_gpus}",
            dict(boxstyle='round', facecolor='lightgreen', alpha=0.3),
        ),
        (
            "Multi-band Config\n"
            f"Albedo:\n"
            f"w: {float(config['W_MIN']):.3f}–{float(config['W_MAX']):.3f}\n"
            f"w blobs: r={config['W_BLOB_RADIUS_PX']} px, ρ={float(config['W_BLOB_DENSITY']):.2f}\n"
            f"Macroscopic Roughness:\n"
            rf"$\bar{{\theta}}$: {theta_min_rad:.1f}–{theta_max_rad:.1f} rad" "\n"
            rf"$\bar{{\theta}}$: {theta_min_deg:.1f}–{theta_max_deg:.1f}°" "\n"
            rf"$\bar{{\theta}}$ blobs: r={config['THETA_BAR_BLOB_RADIUS_PX']} px, ρ={float(config['THETA_BAR_BLOB_DENSITY']):.2f}",
            dict(boxstyle='round', facecolor='lavender', alpha=0.3),
        ),
    ]

        
    if not config["USE_SEMIFLUID"]: 
        infobox_entries.append(  
                    (
                    "Training Config\n"
                    f"Batch Size: {config['BATCH_SIZE']}\n"
                    f"Save every: {config['SAVE_EVERY']} epochs\n"
                    f"Use semifluid: {config['USE_SEMIFLUID']}\n",
                    dict(boxstyle='round', facecolor='lightyellow', alpha=0.3)
                    )
    )
    else:  
        infobox_entries.append(
            (
                f"Training Config\n"
                f"Batch Size: {config['BATCH_SIZE']}\n"
                f"Save every: {config['SAVE_EVERY']} epochs\n"
                f"Use semifluid: {config['USE_SEMIFLUID']}\n"
                f"New data every: {config['NEW_FLUID_DATA_EVERY']} epochs\n",
                dict(boxstyle='round', facecolor='lightyellow', alpha=0.3)
            )
        )
        
    if total_time_logged:
        infobox_entries.append(
            (
                f"Total Train Time:\n"
                f"{total_time_hours:.1f} h {total_time_minutes:.1f} m\n"
                f"{total_time_seconds:.1f} s total",
                dict(boxstyle='round', facecolor='lightyellow', alpha=0.3),
            )
        )

    for idx, (text, bbox_kwargs) in enumerate(infobox_entries):
        ax_box = fig.add_axes([info_start_x + idx * (info_width + info_gap_h), info_start_y, info_width, info_height])
        ax_box.axis('off')
        ax_box.text(
            0.05,
            0.95,
            text,
            transform=ax_box.transAxes,
            fontsize=fontsize_textbox,
            verticalalignment='top',
            family='monospace',
            bbox=bbox_kwargs,
        )

    # Data rows layout (5 datasets by default)
    row_height = 0.13
    row_spacing = 0.012
    start_y = 0.83

    band_width = 0.065  # Slightly smaller to fit 9 bands (6 original + 3 diff)
    band_gap = 0.002
    images_width = 0.060
    images_gap = 0.005
    start_x = 0.04
    
    color_bar_extra_space = 0.025
    space_before_color_bar = 0.009

    # Update band positions to include 9 bands (GT W, Pred W, Diff W, GT Theta, Pred Theta, Diff Theta, GT DEM, Pred DEM, Diff DEM)
    band_positions = []
    current_x = start_x
    for _ in range(9):
        band_positions.append(current_x)
        current_x += band_width + band_gap
        if _ in [1, 2, 4, 5, 7, 8]:  # After pred and after diff
            current_x += color_bar_extra_space
    current_x += 0.005 # Extra gap before images

    image_positions = []
    for _ in range(5):
        image_positions.append(current_x)
        current_x += images_width + images_gap

    last_w_im = None
    last_theta_im = None
    last_dem_im = None
    last_w_dif_im = None
    last_theta_dif_im = None
    last_dem_dif_im = None

    for row_idx, test_idx in enumerate(selected_indices):
        dem_target = all_dem_targets[row_idx]
        dem_pred = all_dem_preds[row_idx]
        w_target = all_w_targets[row_idx]
        w_pred = all_w_preds[row_idx]
        theta_target = all_theta_targets[row_idx]
        theta_pred = all_theta_preds[row_idx]
        images_norm = all_images_norm[row_idx]
        
        # Compute differences
        dem_dif = dem_target - dem_pred
        w_dif = w_target - w_pred
        theta_dif = theta_target - theta_pred

        if same_scale == 'row':
            row_img_min = images_norm.min()
            row_img_max = images_norm.max()

        y_pos = (start_y - row_height) - row_idx * (row_height + row_spacing)

        # GT W
        ax_gt_w = fig.add_axes([band_positions[0], y_pos, band_width, row_height])
        ax_gt_w.imshow(w_target, cmap='viridis', origin='lower', vmin=global_w_min, vmax=global_w_max)
        ax_gt_w.set_xticks([])
        ax_gt_w.set_yticks([])
        if row_idx == 0:
            ax_gt_w.set_title('GT w', fontsize=12, pad=5)
        ax_gt_w.set_ylabel(f'Set {index_labels[row_idx]}', fontsize=11, rotation=90, labelpad=5)

        # Pred W
        ax_pred_w = fig.add_axes([band_positions[1], y_pos, band_width, row_height])
        last_w_im = ax_pred_w.imshow(w_pred, cmap='viridis', origin='lower', vmin=global_w_min, vmax=global_w_max)
        ax_pred_w.set_xticks([])
        ax_pred_w.set_yticks([])
        if row_idx == 0:
            ax_pred_w.set_title('Pred w', fontsize=12, pad=5)

        # Diff W (GT - Pred)
        ax_dif_w = fig.add_axes([band_positions[2], y_pos, band_width, row_height])
        last_w_dif_im = ax_dif_w.imshow(w_dif, cmap='RdBu_r', origin='lower', vmin=global_w_dif_min, vmax=global_w_dif_max)
        ax_dif_w.set_xticks([])
        ax_dif_w.set_yticks([])
        if row_idx == 0:
            ax_dif_w.set_title('GT-Pred w', fontsize=12, pad=5)

        # GT Theta
        ax_gt_theta = fig.add_axes([band_positions[3], y_pos, band_width, row_height])
        ax_gt_theta.imshow(theta_target, cmap='plasma', origin='lower', vmin=global_theta_min, vmax=global_theta_max)
        ax_gt_theta.set_xticks([])
        ax_gt_theta.set_yticks([])
        if row_idx == 0:
            ax_gt_theta.set_title('GT theta', fontsize=12, pad=5)

        # Pred Theta
        ax_pred_theta = fig.add_axes([band_positions[4], y_pos, band_width, row_height])
        last_theta_im = ax_pred_theta.imshow(theta_pred, cmap='plasma', origin='lower', vmin=global_theta_min, vmax=global_theta_max)
        ax_pred_theta.set_xticks([])
        ax_pred_theta.set_yticks([])
        if row_idx == 0:
            ax_pred_theta.set_title('Pred theta', fontsize=12, pad=5)

        # Diff Theta (GT - Pred)
        ax_dif_theta = fig.add_axes([band_positions[5], y_pos, band_width, row_height])
        last_theta_dif_im = ax_dif_theta.imshow(theta_dif, cmap='RdBu_r', origin='lower', vmin=global_theta_dif_min, vmax=global_theta_dif_max)
        ax_dif_theta.set_xticks([])
        ax_dif_theta.set_yticks([])
        if row_idx == 0:
            ax_dif_theta.set_title('GT-Pred theta', fontsize=12, pad=5)

        # GT DEM
        ax_gt_dem = fig.add_axes([band_positions[6], y_pos, band_width, row_height])
        ax_gt_dem.imshow(dem_target, cmap='terrain', origin='lower', vmin=global_dem_min, vmax=global_dem_max)
        ax_gt_dem.set_xticks([])
        ax_gt_dem.set_yticks([])
        if row_idx == 0:
            ax_gt_dem.set_title('GT DEM', fontsize=12, pad=5)

        # Pred DEM
        ax_pred_dem = fig.add_axes([band_positions[7], y_pos, band_width, row_height])
        last_dem_im = ax_pred_dem.imshow(dem_pred, cmap='terrain', origin='lower', vmin=global_dem_min, vmax=global_dem_max)
        ax_pred_dem.set_xticks([])
        ax_pred_dem.set_yticks([])
        if row_idx == 0:
            ax_pred_dem.set_title('Pred DEM', fontsize=12, pad=5)

        # Diff DEM (GT - Pred)
        ax_dif_dem = fig.add_axes([band_positions[8], y_pos, band_width, row_height])
        last_dem_dif_im = ax_dif_dem.imshow(dem_dif, cmap='RdBu_r', origin='lower', vmin=global_dem_dif_min, vmax=global_dem_dif_max)
        ax_dif_dem.set_xticks([])
        ax_dif_dem.set_yticks([])
        if row_idx == 0:
            ax_dif_dem.set_title('GT-Pred DEM', fontsize=12, pad=5)

        # Plot input images
        for img_idx, img_x in enumerate(image_positions):
            ax_img = fig.add_axes([img_x, y_pos, images_width, row_height])

            if same_scale == 'all':
                ax_img.imshow(images_norm[img_idx], cmap='gray', origin='lower', vmin=global_img_min, vmax=global_img_max)
            elif same_scale == 'row':
                ax_img.imshow(images_norm[img_idx], cmap='gray', origin='lower', vmin=row_img_min, vmax=row_img_max)
            else:
                ax_img.imshow(images_norm[img_idx], cmap='gray', origin='lower')

            ax_img.set_xticks([])
            ax_img.set_yticks([])
            if row_idx == 0:
                ax_img.set_title(f'Img {img_idx+1}', fontsize=12, pad=5)

    n_rows = len(selected_indices)
    total_height = n_rows * row_height + max(0, (n_rows - 1)) * row_spacing
    bottom_y = (start_y - row_height) - max(0, (n_rows - 1)) * (row_height + row_spacing)

    # Colorbars for GT/Pred (same as original)
    if last_w_im is not None:
        cbar_w_ax = fig.add_axes([band_positions[1] + band_width - 0.004 + space_before_color_bar, bottom_y, 0.006, total_height])
        cbar_w = plt.colorbar(last_w_im, cax=cbar_w_ax)
        cbar_w.ax.tick_params(labelsize=10)
        cbar_w.set_label('w', fontsize=10)

    if last_theta_im is not None:
        cbar_theta_ax = fig.add_axes([band_positions[4] + band_width - 0.004 + space_before_color_bar, bottom_y, 0.006, total_height])
        cbar_theta = plt.colorbar(last_theta_im, cax=cbar_theta_ax)
        cbar_theta.ax.tick_params(labelsize=10)
        cbar_theta.set_label('theta (rad)', fontsize=10)

    if last_dem_im is not None:
        cbar_dem_ax = fig.add_axes([band_positions[7] + band_width - 0.004 + space_before_color_bar, bottom_y, 0.006, total_height])
        cbar_dem = plt.colorbar(last_dem_im, cax=cbar_dem_ax)
        cbar_dem.ax.tick_params(labelsize=10)
        cbar_dem.set_label('DEM (m)', fontsize=10)

    # Colorbars for differences
    if last_w_dif_im is not None:
        cbar_w_dif_ax = fig.add_axes([band_positions[2] + band_width - 0.004 + space_before_color_bar, bottom_y, 0.006, total_height])
        cbar_w_dif = plt.colorbar(last_w_dif_im, cax=cbar_w_dif_ax)
        cbar_w_dif.ax.tick_params(labelsize=10)
        cbar_w_dif.set_label('Δw', fontsize=10)

    if last_theta_dif_im is not None:
        cbar_theta_dif_ax = fig.add_axes([band_positions[5] + band_width - 0.004 + space_before_color_bar, bottom_y, 0.006, total_height])
        cbar_theta_dif = plt.colorbar(last_theta_dif_im, cax=cbar_theta_dif_ax)
        cbar_theta_dif.ax.tick_params(labelsize=10)
        cbar_theta_dif.set_label('Δtheta (rad)', fontsize=10)

    if last_dem_dif_im is not None:
        cbar_dem_dif_ax = fig.add_axes([band_positions[8] + band_width - 0.004 + space_before_color_bar, bottom_y, 0.006, total_height])
        cbar_dem_dif = plt.colorbar(last_dem_dif_im, cax=cbar_dem_dif_ax)
        cbar_dem_dif.ax.tick_params(labelsize=10)
        cbar_dem_dif.set_label('ΔDEM (m)', fontsize=10)

    if save_fig:
        figures_dir = os.path.join(sup_dir, run_dir, 'figures')
        os.makedirs(figures_dir, exist_ok=True)
        output_path = os.path.join(figures_dir, filename if filename is not None else 'comprehensive_plot_multi_band_with_dif.pdf')
        fig.savefig(output_path, format='pdf', bbox_inches='tight', dpi=150)
        print(f"Figure saved to: {output_path}")

    if return_fig:
        return fig


def plot_data_pt(run_dir, n_sets=5, n_images=5, save_fig=True, return_fig=False, same_scale=False, variant='random', use_train_set = False):
    """
    Plot DEMs and images from .pt files (PyTorch format).
    Similar to plot_data but works with .pt files instead of .npz files.
    """

    # Ensure figures directory exists
    figures_dir = os.path.join(run_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    test_dems_dir = os.path.join(run_dir, 'test')
    test_files = None

    print(f"Use train set: {use_train_set}")
    if not use_train_set:
        test_files = None
        if os.path.isdir(test_dems_dir):
            # Look for .pt files in the test_dems folder
            candidate_files = sorted(glob.glob(os.path.join(test_dems_dir, '*.pt')))
            if len(candidate_files) > 0:
                test_files = candidate_files
                print(f"Using {len(test_files)} test files from: {test_dems_dir}")
            else:
                print(f"Found '{test_dems_dir}' but no .pt files were present.")
                raise FileNotFoundError(f"No .pt files found in {test_dems_dir}")
        else:
            raise FileNotFoundError(f"No test directory found at '{test_dems_dir}'")

        test_dataset = DEMDataset(test_files)
    
    config = load_config_file(os.path.join(run_dir, 'stats', 'config.ini'))
    if use_train_set:
        print("Using training set for plotting instead of test set.")

        train_set = FluidDEMDataset(config)
        test_dataset = train_set
    
    # Select test sets based on variant
    print(f"Selecting {n_sets} test sets using variant: {variant}")
    available_indices = list(range(len(test_dataset)))
    
    if variant == 'first':
        selected_indices = list(range(min(n_sets, len(test_dataset))))
    elif variant == 'random':
        selected_indices = sorted(random.sample(available_indices, min(n_sets, len(test_dataset))))
        print("Selected test set indices:", selected_indices)
    else:
        raise ValueError(f"Unknown variant: {variant}. Use 'first' or 'random'.")
    
    # Load all data and find global min/max
    all_dems = []
    all_images = []
    
    for idx in selected_indices:
        if not config["SAVE_LRO_METAS"]:
            images, reflectance_maps, target, meta = test_dataset[idx]
        elif use_train_set:
            images, reflectance_maps, target, meta = test_dataset[idx]
        elif config["SAVE_LRO_METAS"]:
            images, reflectance_maps, target, meta, lro_metas = test_dataset[idx]

        dem = target.squeeze().numpy()
        all_dems.append(dem)
        for img_idx in range(n_images):
            img = images[img_idx].numpy()
            all_images.append(img)
    
    all_dems = np.array(all_dems)
    all_images = np.array(all_images)
    
    # Find global min/max for DEMs and images
    dem_vmin = np.min(all_dems)
    dem_vmax = np.max(all_dems)
    img_vmin = np.min(all_images)
    img_vmax = np.max(all_images)

    # Create figure with manual positioning for precise control
    fig = plt.figure(figsize=(3.2*(n_images+1), 2.5*n_sets))
    
    # Layout parameters
    row_height = 0.85 / n_sets  # Total height available divided by rows
    row_spacing = 0.005  # Small gap between rows
    
    # Column widths and positions
    dem_width = 0.13  # Width for DEM column
    dem_x = 0.08  # Left position for DEM
    
    img_width = 0.11  # Width for each image
    img_x_start = dem_x + dem_width + 0.03  # Start after DEM with larger gap
    img_gap = 0.002  # Tiny gap between images
    
    # Starting y position (from top)
    start_y = 0.92
    
    # Store the last DEM image object for colorbar
    last_dem_im = None
    
    for i, test_idx in enumerate(selected_indices):
        # Load data
        if not config["SAVE_LRO_METAS"]:
            images, reflectance_maps, target, meta = test_dataset[test_idx]
        else:
            images, reflectance_maps, target, meta, lro_metas = test_dataset[test_idx]
        dem = target.squeeze().numpy()
        dem_num = test_idx
        images = np.array([images[j].numpy() for j in range(n_images)])
        # Calculate per-row min/max if same_scale='row'
        if same_scale == 'row':
            row_img_min = images.min()
            row_img_max = images.max()
        
        # Calculate y position for this row
        y_pos = start_y - i * (row_height + row_spacing)
        
        # Column 0: DEM (without individual colorbar)
        ax_dem = fig.add_axes([dem_x, y_pos - row_height, dem_width, row_height])
        im_dem = ax_dem.imshow(dem, cmap='terrain', vmin=dem_vmin, vmax=dem_vmax, origin='lower')
        
        # Store the image object for the colorbar
        last_dem_im = im_dem
        
        # Remove ticks and tick labels
        ax_dem.set_xticks([])
        ax_dem.set_yticks([])
        
        # Title only on first row
        if i == 0:
            title_str = 'DEM'
            ax_dem.set_title(title_str, fontsize=13, pad=5)
        
        # Y-label showing DEM number
        if dem_num is not None:
            ax_dem.set_ylabel(f'DEM {dem_num}', fontsize=12, rotation=90, labelpad=5)
        
        # Columns 1-5: Images with different scaling options
        for j in range(n_images):
            img_x = img_x_start + j * (img_width + img_gap)
            ax_img = fig.add_axes([img_x, y_pos - row_height, img_width, row_height])
            
            if same_scale == 'all':
                # Use same scale for all images across all rows
                ax_img.imshow(images[j], cmap='gray', vmin=img_vmin, vmax=img_vmax, origin='lower')
            elif same_scale == 'row':
                # Use same scale for all images in this row
                ax_img.imshow(images[j], cmap='gray', vmin=row_img_min, vmax=row_img_max, origin='lower')
            else:  # same_scale=False
                # Use individual scale for each image
                ax_img.imshow(images[j], cmap='gray', origin='lower')
            
            # Remove ticks and tick labels
            ax_img.set_xticks([])
            ax_img.set_yticks([])
            
            # Title only on first row
            if i == 0:
                ax_img.set_title(f'Image {j+1}', fontsize=13, pad=5)
    
    # Add one tall colorbar spanning all rows, close to the DEMs
    # Calculate the total height of all DEM plots
    total_height = n_sets * row_height + (n_sets - 1) * row_spacing
    bottom_y = start_y - total_height
    
    # Create a dummy axis for the colorbar (reduced width from 0.015 to 0.01)
    cbar_ax = fig.add_axes([dem_x + dem_width - 0.006, bottom_y, 0.008, total_height])
    cbar = plt.colorbar(last_dem_im, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=8)
    
    output_name = "data_summary_test.pdf" if not use_train_set else "data_summary_train.pdf"

    # Save or show
    if save_fig:
        figures_dir = os.path.join(run_dir, 'figures')
        os.makedirs(figures_dir, exist_ok=True)
        output_path = os.path.join(figures_dir, output_name)
        fig.savefig(output_path, format='pdf', bbox_inches='tight', dpi=150)
        print(f"Figure saved to: {output_path}")

    if return_fig:
        return fig


def plot_data_multi_band(run_dir, n_sets=5, n_images=5, save_fig=True, return_fig=False, same_scale=False, variant='random', use_train_set=False):
    """
    Plot multi-band data (W, Theta_bar, DEM) and images from .pt files (PyTorch format).
    Similar to plot_data_pt but includes W and Theta_bar bands.
    
    Layout: 8 columns per row
    - Columns 0-2: W band, Theta_bar band, DEM
    - Columns 3-7: 5 input images
    """

    # read config file
    config = load_config_file(os.path.join(run_dir, 'stats', 'config.ini'))

    if not config["USE_MULTI_BAND"]:
        raise ValueError("The provided run_dir does not correspond to a multi-band model. Please use plot_data_pt instead.")

    # Ensure figures directory exists
    figures_dir = os.path.join(run_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    test_dems_dir = os.path.join(run_dir, 'test')
    test_files = None

    print(f"Use train set: {use_train_set}")
    if not use_train_set:
        test_files = None
        if os.path.isdir(test_dems_dir):
            # Look for .pt files in the test_dems folder
            candidate_files = sorted(glob.glob(os.path.join(test_dems_dir, '*.pt')))
            if len(candidate_files) > 0:
                test_files = candidate_files
                print(f"Using {len(test_files)} test files from: {test_dems_dir}")
            else:
                print(f"Found '{test_dems_dir}' but no .pt files were present.")
                raise FileNotFoundError(f"No .pt files found in {test_dems_dir}")
        else:
            raise FileNotFoundError(f"No test directory found at '{test_dems_dir}'")

        test_dataset = DEMDataset(test_files, config=config)
    
    if use_train_set:
        print("Using training set for plotting instead of test set.")
        train_set = FluidDEMDataset(config=config)
        test_dataset = train_set
    
    # Select test sets based on variant
    print(f"Selecting {n_sets} test sets using variant: {variant}")
    available_indices = list(range(len(test_dataset)))
    
    if variant == 'first':
        selected_indices = list(range(min(n_sets, len(test_dataset))))
    elif variant == 'random':
        selected_indices = sorted(random.sample(available_indices, min(n_sets, len(test_dataset))))
        print("Selected test set indices:", selected_indices)
    else:
        raise ValueError(f"Unknown variant: {variant}. Use 'first' or 'random'.")
    
    # Load all data and find global min/max
    all_w_bands = []
    all_theta_bands = []
    all_dems = []
    all_images = []
    
    for idx in selected_indices:
        images_tensor, reflectance_maps_tensor, target_tensor, meta_tensor, w_tensor, theta_bar_tensor, lro_meta_tensor = test_dataset[idx]

        dem = target_tensor.squeeze().numpy()
        all_dems.append(dem)
        
        # Extract W and Theta_bar 
        w_band = w_tensor.numpy()
        theta_band = theta_bar_tensor.numpy()
        all_w_bands.append(w_band)
        all_theta_bands.append(theta_band)
        
        for img_idx in range(n_images):
            img = images_tensor[img_idx].numpy()
            all_images.append(img)
    
    all_w_bands = np.array(all_w_bands)
    all_theta_bands = np.array(all_theta_bands)
    all_dems = np.array(all_dems)
    all_images = np.array(all_images)
    
    # Find global min/max for all data
    w_vmin = np.min(all_w_bands)
    w_vmax = np.max(all_w_bands)
    theta_vmin = np.min(all_theta_bands)
    theta_vmax = np.max(all_theta_bands)
    dem_vmin = np.min(all_dems)
    dem_vmax = np.max(all_dems)
    img_vmin = np.min(all_images)
    img_vmax = np.max(all_images)

    # Create figure with manual positioning for precise control
    # 8 columns: W, Theta, DEM, + 5 images
    fig = plt.figure(figsize=(3.6*8, 2.5*n_sets))
    
    # Layout parameters
    row_height = 0.85 / n_sets  # Total height available divided by rows
    row_spacing = 0.005  # Small gap between rows
    
    # Column widths and positions for 8 columns
    band_width = 0.09  # Width for W, Theta, DEM columns
    dem_width = 0.09
    img_width = 0.09  # Width for each image
    
    # Horizontal positions
    w_x = 0.05          # W band
    theta_x = w_x + band_width + 0.015     # Theta band
    dem_x = theta_x + band_width + 0.01   # DEM
    img_x_start = dem_x + dem_width + 0.02  # Start of images
    img_gap = 0.0005     # Tiny gap between images
    
    # Starting y position (from top)
    start_y = 0.92
    
    # Store the last image objects for colorbars
    last_w_im = None
    last_theta_im = None
    last_dem_im = None
    
    for i, test_idx in enumerate(selected_indices):
        # Load data
        images_tensor, reflectance_maps_tensor, target_tensor, meta_tensor, w_tensor, theta_bar_tensor, lro_meta_tensor = test_dataset[test_idx]

        
        dem = target_tensor.squeeze().numpy()
        w_band = w_tensor.squeeze().numpy()
        theta_band = theta_bar_tensor.squeeze().numpy()
        dem_num = test_idx
        images = np.array([images_tensor[j].numpy() for j in range(n_images)])
        
        # Calculate per-row min/max if same_scale='row'
        if same_scale == 'row':
            row_img_min = images.min()
            row_img_max = images.max()
        
        # Calculate y position for this row
        y_pos = start_y - i * (row_height + row_spacing)
        
        # Column 0: W band
        ax_w = fig.add_axes([w_x, y_pos - row_height, band_width, row_height])
        im_w = ax_w.imshow(w_band, cmap='viridis', vmin=w_vmin, vmax=w_vmax, origin='lower')
        ax_w.set_xticks([])
        ax_w.set_yticks([])
        if i == 0:
            ax_w.set_title('W Band', fontsize=12, pad=5)
        if i == 0:
            ax_w.set_ylabel(f'Dataset {dem_num}', fontsize=11, rotation=90, labelpad=5)
        else:
            ax_w.set_ylabel(f'Dataset {dem_num}', fontsize=11, rotation=90, labelpad=5)
        last_w_im = im_w
        
        # Column 1: Theta band
        ax_theta = fig.add_axes([theta_x, y_pos - row_height, band_width, row_height])
        im_theta = ax_theta.imshow(theta_band, cmap='plasma', vmin=theta_vmin, vmax=theta_vmax, origin='lower')
        ax_theta.set_xticks([])
        ax_theta.set_yticks([])
        if i == 0:
            ax_theta.set_title('Theta Band', fontsize=12, pad=5)
        last_theta_im = im_theta
        
        # Column 2: DEM
        ax_dem = fig.add_axes([dem_x, y_pos - row_height, dem_width, row_height])
        im_dem = ax_dem.imshow(dem, cmap='terrain', vmin=dem_vmin, vmax=dem_vmax, origin='lower')
        ax_dem.set_xticks([])
        ax_dem.set_yticks([])
        if i == 0:
            ax_dem.set_title('DEM', fontsize=12, pad=5)
        last_dem_im = im_dem
        
        # Columns 3-7: Images with different scaling options
        for j in range(n_images):
            img_x = img_x_start + j * (img_width + img_gap)
            ax_img = fig.add_axes([img_x, y_pos - row_height, img_width, row_height])
            
            if same_scale == 'all':
                # Use same scale for all images across all rows
                ax_img.imshow(images[j], cmap='gray', vmin=img_vmin, vmax=img_vmax, origin='lower')
            elif same_scale == 'row':
                # Use same scale for all images in this row
                ax_img.imshow(images[j], cmap='gray', vmin=row_img_min, vmax=row_img_max, origin='lower')
            else:  # same_scale=False
                # Use individual scale for each image
                ax_img.imshow(images[j], cmap='gray', origin='lower')
            
            # Remove ticks and tick labels
            ax_img.set_xticks([])
            ax_img.set_yticks([])
            
            # Title only on first row
            if i == 0:
                ax_img.set_title(f'Img {j+1}', fontsize=12, pad=5)
    
    # Add colorbars spanning all rows
    total_height = n_sets * row_height + (n_sets - 1) * row_spacing
    bottom_y = start_y - total_height
    
    # Colorbar for W band
    cbar_w_width = 0.006
    cbar_w_x = w_x + band_width - 0.004
    cbar_w_ax = fig.add_axes([cbar_w_x, bottom_y, cbar_w_width, total_height])
    cbar_w = plt.colorbar(last_w_im, cax=cbar_w_ax)
    cbar_w.ax.tick_params(labelsize=8)
    
    # Colorbar for Theta band
    cbar_theta_width = 0.006
    cbar_theta_x = theta_x + band_width - 0.004
    cbar_theta_ax = fig.add_axes([cbar_theta_x, bottom_y, cbar_theta_width, total_height])
    
    # Convert theta to degrees for colorbar display
    last_theta_im.set_array(last_theta_im.get_array())
    last_theta_im.set_clim(vmin=theta_vmin, vmax=theta_vmax)
    
    cbar_theta = plt.colorbar(last_theta_im, cax=cbar_theta_ax)
    cbar_theta.ax.tick_params(labelsize=8)
    
    # Colorbar for DEM
    cbar_dem_width = 0.006
    cbar_dem_x = dem_x + dem_width - 0.004
    cbar_dem_ax = fig.add_axes([cbar_dem_x, bottom_y, cbar_dem_width, total_height])
    cbar_dem = plt.colorbar(last_dem_im, cax=cbar_dem_ax)
    cbar_dem.ax.tick_params(labelsize=8)
    
    output_name = "data_summary_multi_band_test.pdf" if not use_train_set else "data_summary_multi_band_train.pdf"

    # Save or show
    if save_fig:
        figures_dir = os.path.join(run_dir, 'figures')
        os.makedirs(figures_dir, exist_ok=True)
        output_path = os.path.join(figures_dir, output_name)
        fig.savefig(output_path, format='pdf', bbox_inches='tight', dpi=150)
        print(f"Figure saved to: {output_path}")

    if return_fig:
        return fig


def plot_data(run_dir, n_sets=5, n_images=5, save_fig=True, return_fig=False,
                   fig_path='data_grid.pdf', variant='random', 
                   same_scale=False, output_name='data_grid.pdf'):


    #ensure figures directory exists
    figures_dir = os.path.join(run_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    test_dems_dir = os.path.join(run_dir, 'test')
    test_files = None
    if os.path.isdir(test_dems_dir):
        # Look for .npz files in the test_dems folder
        candidate_files = sorted(glob.glob(os.path.join(test_dems_dir, '*.npz')))
        if len(candidate_files) > 0:
            test_files = candidate_files
            print(f"Using {len(test_files)} test files from: {test_dems_dir}")
        else:
            raise ValueError(f"Found '{test_dems_dir}' but no .npz files were present.")
    
    if len(test_files) < n_sets:
        raise ValueError(f"Not enough .npz files in provided folder to plot {n_sets} sets. Found only {len(test_files)}.") 
    
    # Select files based on variant.
    files = []
    if variant in ('first'):
        files = test_files[:n_sets]
    elif variant in ('random'):
        files = random.sample(test_files, n_sets)
    else:
        raise ValueError(f"Unknown variant: {variant}")
    
    # find dataset number from and use to sort files
    for i in range(len(files)):
        files[i] = (extract_dataset_number(os.path.basename(files[i])), files[i])
    files = sorted(files, key=lambda x: x[0])
    files = [f[1] for f in files]

    # Find global min/max for all DEMs and images
    all_dems = []
    all_images = []
    for f in files:
        d = np.load(f)
        all_dems.append(d['dem'])
        all_images.append(d['data'])
    all_dems = np.stack(all_dems, axis=0)
    all_images = np.concatenate(all_images, axis=0)
    dem_vmin = np.min(all_dems)
    dem_vmax = np.max(all_dems)
    img_vmin = np.min(all_images)
    img_vmax = np.max(all_images)

    # Create figure with manual positioning for precise control
    fig = plt.figure(figsize=(3.2*(n_images+1), 2.5*n_sets))
    
    # Layout parameters
    row_height = 0.85 / n_sets  # Total height available divided by rows
    row_spacing = 0.005  # Small gap between rows
    
    # Column widths and positions
    dem_width = 0.13  # Width for DEM column
    dem_x = 0.08  # Left position for DEM
    
    img_width = 0.11  # Width for each image
    img_x_start = dem_x + dem_width + 0.03  # Start after DEM with larger gap
    img_gap = 0.002  # Tiny gap between images
    
    # Starting y position (from top)
    start_y = 0.92
    
    # Store the last DEM image object for colorbar
    last_dem_im = None
    
    for i, f in enumerate(files):
        d = np.load(f)
        dem = d['dem']
        images = d['data']
        dem_num = extract_dataset_number(os.path.basename(f))
        
        # Calculate per-row min/max if same_scale='row'
        if same_scale == 'row':
            row_img_min = images.min()
            row_img_max = images.max()
        
        # Calculate y position for this row
        y_pos = start_y - i * (row_height + row_spacing)
        
        # Column 0: DEM (without individual colorbar)
        ax_dem = fig.add_axes([dem_x, y_pos - row_height, dem_width, row_height])
        im_dem = ax_dem.imshow(dem, cmap='terrain', vmin=dem_vmin, vmax=dem_vmax, origin='lower')
        
        # Store the image object for the colorbar
        last_dem_im = im_dem
        
        # Remove ticks and tick labels
        ax_dem.set_xticks([])
        ax_dem.set_yticks([])
        
        # Title only on first row
        if i == 0:
            title_str = 'DEM'
            ax_dem.set_title(title_str, fontsize=13, pad=5)
        
        # Y-label showing DEM number
        if dem_num is not None:
            ax_dem.set_ylabel(f'DEM {dem_num}', fontsize=12, rotation=90, labelpad=5)
        
        # Columns 1-5: Images with different scaling options
        for j in range(n_images):
            img_x = img_x_start + j * (img_width + img_gap)
            ax_img = fig.add_axes([img_x, y_pos - row_height, img_width, row_height])
            
            if same_scale == 'all':
                # Use same scale for all images across all rows
                ax_img.imshow(images[j], cmap='gray', vmin=img_vmin, vmax=img_vmax, origin='lower')
            elif same_scale == 'row':
                # Use same scale for all images in this row
                ax_img.imshow(images[j], cmap='gray', vmin=row_img_min, vmax=row_img_max, origin='lower')
            else:  # same_scale=False
                # Use individual scale for each image
                ax_img.imshow(images[j], cmap='gray', origin='lower')
            
            # Remove ticks and tick labels
            ax_img.set_xticks([])
            ax_img.set_yticks([])
            
            # Title only on first row
            if i == 0:
                ax_img.set_title(f'Image {j+1}', fontsize=13, pad=5)
    
    # Add one tall colorbar spanning all rows, close to the DEMs
    # Calculate the total height of all DEM plots
    total_height = n_sets * row_height + (n_sets - 1) * row_spacing
    bottom_y = start_y - total_height
    
    # Create a dummy axis for the colorbar (reduced width from 0.015 to 0.01)
    cbar_ax = fig.add_axes([dem_x + dem_width - 0.006, bottom_y, 0.008, total_height])
    cbar = plt.colorbar(last_dem_im, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=8)
    
    # Save or show
    if save_fig:
        figures_dir = os.path.join(run_dir, 'figures')
        os.makedirs(figures_dir, exist_ok=True)
        output_path = os.path.join(figures_dir, output_name)
        fig.savefig(output_path, format='pdf', bbox_inches='tight', dpi=150)
        print(f"Figure saved to: {output_path}")

    if return_fig:
        return fig


def extract_dataset_number(filename):
    """Extract dataset number from filename."""
    match = re.search(r'dataset_(\d+)', filename)
    return int(match.group(1)) if match else None




import argparse

if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Plot comprehensive test set predictions for a trained UNet model.")
    args.add_argument('run_dir', nargs="?" , type=str,
                    help='Directory of the trained model run containing stats and checkpoints.')
    args.add_argument('--run_dir', type=str, required=False, dest='run_dir_flag',
                      help='Directory of the trained model run containing stats and checkpoints.')
    args.add_argument('--variant', type=str, default='first',
                      help="Variant for selecting test sets: 'first' or 'random'.")
    args.add_argument('--use_train_set', action='store_true',
                      help="Use the training set for plotting instead of the test set.")
    args.add_argument('--test_on_separate_data', action='store_true',
                      help="Test on a separate dataset if available.")
    args.add_argument('--diff', action='store_true',
                      help="Include difference images in the plot.")
    args = args.parse_args()
    
    # Support both positional and flag-based arguments
    run_dir = args.run_dir_flag or args.run_dir

    print("Plotting test set predictions...")

    config = load_config_file(os.path.join("runs", run_dir, 'stats', 'config.ini'))

    if not os.path.exists(os.path.join("runs", run_dir, 'checkpoints', 'snapshot.pt')):
        #no training exists, only plot data
        print("No training snapshot found, plotting data only...")
        if config["USE_MULTI_BAND"]:
            print("Detected multi-band model, plotting multi-band data...")
            plot_data_multi_band(run_dir=os.path.join("runs", run_dir),
                                 n_sets=5,
                                 variant=args.variant,  # 'first' or 'random'
                                 same_scale=False,  # False, 'row', or 'all'
                                 save_fig=True,
                                 return_fig=False,
                                 use_train_set = args.use_train_set)
        else:
            print("Detected single-band model, plotting single-band data...")
            plot_data_pt(run_dir=os.path.join("runs", run_dir),
                            n_sets=5,
                            variant=args.variant,  # 'first' or 'random'
                            same_scale=False,  # False, 'row', or 'all'
                            save_fig=True,
                            return_fig=False,
                            use_train_set = args.use_train_set)
    else:
        filename = "comprehensive"
        if not args.use_train_set:
            filename += "_testset"
        if args.use_train_set:
            filename += "_trainset"
        if args.test_on_separate_data:
            filename += "_alt"
        if args.variant == 'first':
            filename += "_first"
        if args.variant == 'random':
            filename += "_random"
        if args.diff:
            filename += "_with_diff"
        filename += ".pdf"
        print(f"Output filename: {filename}")
        print("Training snapshot found, plotting predictions...")
        print(f"Config USE_MULTI_BAND: {config['USE_MULTI_BAND']}")
        if config["USE_MULTI_BAND"]:
            if args.diff:
                print("Difference images will be included in the plot.")
                plot_comprehensive_multi_band_with_dif(run_dir=run_dir,
                                n_test_sets=5,
                                variant=args.variant,
                                same_scale=False,
                                save_fig=True,
                                return_fig=False,
                                use_train_set=args.use_train_set)
            else:
                print("Detected multi-band model, plotting multi-band predictions...")
                plot_comprehensive_multi_band(run_dir=run_dir,
                                            n_test_sets=5,
                                            variant=args.variant,
                                            same_scale=False,
                                            save_fig=True,
                                            return_fig=False,
                                            use_train_set=args.use_train_set)
        else:
            print("Detected single-band model, plotting single-band predictions...")
            plot_comprehensive_pt(run_dir=run_dir,
                                  n_test_sets=5,
                                  variant=args.variant,
                                  same_scale=False,
                                  figsize=(15, 10),
                                  save_fig=True,
                                  return_fig=False,
                                  use_train_set=args.use_train_set)