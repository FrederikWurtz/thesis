import matplotlib.pyplot as plt
import os
import os
import re
import torch
import random
import numpy as np

from master.models.unet import UNet
from master.train.trainer_core import DEMDataset
from master.train.train_utils import normalize_inputs
from master.train.checkpoints import load_checkpoint, read_file_from_ini
from master.configs.config_utils import load_config_file
import glob

# hej!

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

plt.rcParams.update({
    'text.usetex': True,
    'font.size': 12,
    'font.family': 'serif',
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


def plot_predictions(run_dir,
                     n_test_sets=5,
                     variant='first',  # 'first' or 'random'
                     same_scale=False,  # False, 'row', or 'all'
                     figsize=(15, 10),
                     save_fig=True,
                     output_name='predictions_summary.pdf',
                     config_path=None,
                     snapshot_path=None,
                     return_fig=False):

    # Load model, history, and test dataset
    snapshot = torch.load(snapshot_path, map_location='cpu')["MODEL_STATE"]
    config = load_config_file(config_path)

    test_dems_dir = os.path.join(run_dir, 'test')
    test_files = None
    if os.path.isdir(test_dems_dir):
        # Look for .npz files in the test_dems folder
        candidate_files = sorted(glob.glob(os.path.join(test_dems_dir, '*.npz')))
        if len(candidate_files) > 0:
            test_files = candidate_files
            print(f"Using {len(test_files)} test files from: {test_dems_dir}")
        else:
            print(f"Found '{test_dems_dir}' but no .npz files were present. Falling back to history['test_files'].")

    model = UNet(in_channels=config["IMAGES_PER_DEM"], out_channels=1)
    model.load_state_dict(snapshot)
    model.eval()


    test_dataset = DEMDataset(test_files)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model.to(device)
    
    # Select test sets based on variant
    available_indices = list(range(len(test_dataset)))
    
    if variant == 'first':
        # Use the first n_test_sets from the test pool
        selected_indices = list(range(min(n_test_sets, len(test_dataset))))
    elif variant == 'random':
        # Randomly sample n_test_sets and sort them by index
        selected_indices = sorted(random.sample(available_indices, min(n_test_sets, len(test_dataset))))
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
            #images_norm = normalize_inputs(images.unsqueeze(0), train_mean, train_std)
            #pred = model(images_norm.to(device), meta.unsqueeze(0).to(device))
            pred = model(images.to(device), meta.unsqueeze(0).to(device))
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
    print("Creating prediction summary figure...")
    # ========================================================================
    # Top row: Loss plot (left) + 4 info boxes (right)
    # ========================================================================
    # Loss plot on the left (50% of the width)
    # loss_start_x = 0.1
    # loss_width = 0.45  # 50% of usable space
    # loss_height = 0.10
    # loss_start_y = 0.88
    
    # ax_loss = fig.add_axes([loss_start_x, loss_start_y, loss_width, loss_height])
    
    # epochs = list(range(1, len(train_losses) + 1))

    # ax_loss.plot(epochs, train_losses, label='Training Loss', linewidth=2)
    # ax_loss.plot(epochs, val_losses, label='Validation Loss', linewidth=2)
    # ax_loss.set_yscale('log')
    # ax_loss.set_xlabel('Epoch', fontsize=12)
    # ax_loss.set_ylabel('Loss', fontsize=12)
    # ax_loss.set_title('Training and Validation Loss', fontsize=14, pad=14)
    # ax_loss.legend(fontsize=10, loc='best')
    # ax_loss.grid(True, alpha=0.3)
    
    # # Set x-axis to show only integer ticks
    # from matplotlib.ticker import MaxNLocator
    # ax_loss.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # # Position for 4 text boxes in a row on the right (remaining 50%)
    # info_width = 0.1125  # (0.45 / 4) - split remaining 50% into 4 equal boxes
    # info_height = 0.10   # Height of each text box (taller to fit 10 lines)
    # info_gap_h = -0.007  # Slight overlap to eliminate visual gaps from rounded corners
    # info_start_x = loss_start_x + loss_width + 0.02  # Start after loss plot with small gap
    # info_start_y = 0.88  # Starting y position (top row)
    
    # fontsize_textbox = 13

    # infobox_1_text = ("Unet DL Network \n"
    #                     f"Best epoch: {int(run_stats.get('best_epoch', 'N/A'))}\n"
    #                     f"Test Loss: {float(run_stats.get('test_loss', 'N/A')):.3f}\n"
    #                     f"Test AME: {float(run_stats.get('test_ame', 'N/A')):.3f}\n"
    #                     f"Final LR: {float(checkpoint['learning_rate']):.3e}\n"

    #                     )
    
    # # Info box 1: 
    # ax_model = fig.add_axes([info_start_x, info_start_y, info_width, info_height])
    # ax_model.axis('off')
    # ax_model.text(0.05, 0.95, infobox_1_text,
    #               transform=ax_model.transAxes,
    #               fontsize=fontsize_textbox,
    #               verticalalignment='top',
    #               family='monospace',
    #               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    # # # Info box 2: Loss Function
    # # ax_loss_info = fig.add_axes([info_start_x + 1*info_width + 1*info_gap_h, info_start_y, info_width, info_height])
    # # ax_loss_info.axis('off')
    # # ax_loss_info.text(0.05, 0.95, loss_summary,
    # #                   transform=ax_loss_info.transAxes,
    # #                   fontsize=fontsize_textbox,
    # #                   verticalalignment='top',
    # #                   family='monospace',
    # #                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # total_epochs = int(timing_info['epochs'])
    # wall_hours = float(timing_info['wall_hours'])
    # gpu_seconds_per_epoch = float(timing_info['gpu_seconds_per_epoch'])
    # cpu_seconds_per_epoch = float(timing_info['cpu_seconds_per_epoch'])
    # infobox_3_text = (  
    #                     f"Total Epochs: {total_epochs}\n"
    #                     f"Total Time: {wall_hours:.2f} hrs\n"
    #                     f"GPU: {gpu_seconds_per_epoch:.2f} seconds/epoch\n"
    #                     f"CPU: {cpu_seconds_per_epoch:.2f} seconds/epoch\n")

    # # Info box 3: 
    # ax_training = fig.add_axes([info_start_x + 2*info_width + 2*info_gap_h, info_start_y, info_width, info_height])
    # ax_training.axis('off')
    # ax_training.text(0.05, 0.95, infobox_3_text,
    #                  transform=ax_training.transAxes,
    #                  fontsize=fontsize_textbox,
    #                  verticalalignment='top',
    #                  family='monospace',
    #                  bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    # # # Info box 4: Training Data
    # # ax_data = fig.add_axes([info_start_x + 3*info_width + 3*info_gap_h - 0.014, info_start_y, info_width, info_height])
    # # ax_data.axis('off')
    # # ax_data.text(0.05, 0.95, data_summary,
    # #              transform=ax_data.transAxes,
    # #              fontsize=fontsize_textbox,
    # #              verticalalignment='top',
    # #              family='monospace',
    # #              bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.3))
    
    # ========================================================================
    # Rows 2-6: Test set predictions with manual positioning for tight spacing
    # ========================================================================
    row_height = 0.13
    row_spacing = 0.01
    start_y = 0.84  # Top of first row
    
    # Column widths and positions (extended to use more horizontal space)
    diff_width = 0.13  # Wider to account for colorbar
    dem_width = 0.105
    pred_width = 0.125  # Wider to account for colorbar
    img_width = 0.105
    
    # Horizontal positions (centered better with equal margins)
    diff_x = 0.075  # Shifted right slightly for better balance
    gt_x = diff_x + diff_width + 0.015  # Small gap after diff
    pred_x = gt_x + dem_width - 0.01   # Tiny gap between DEMs
    img_x_start = pred_x + pred_width + 0.015  # Small gap before images
    img_gap = 0.002  # Tiny gap between images
    
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
    cbar_diff_x = diff_x + diff_width - 0.013  # Positioned further left
    cbar_diff_ax = fig.add_axes([cbar_diff_x, bottom_y, cbar_diff_width, total_height])
    cbar_diff = plt.colorbar(last_diff_im, cax=cbar_diff_ax)
    cbar_diff.ax.tick_params(labelsize=10)
    
    # Add shared colorbar for DEMs (predicted) (positioned at right edge of pred column)
    cbar_dem_width = 0.007
    cbar_dem_x = pred_x + pred_width - 0.01  # Positioned further left
    cbar_dem_ax = fig.add_axes([cbar_dem_x, bottom_y, cbar_dem_width, total_height])
    cbar_dem = plt.colorbar(last_pred_im, cax=cbar_dem_ax)
    cbar_dem.ax.tick_params(labelsize=10)
    
    
    # Save?
    if save_fig:
        figures_dir = "master/simple_setup_res"
        os.makedirs(figures_dir, exist_ok=True)
        output_path = os.path.join(figures_dir, output_name)
        fig.savefig(output_path, format='pdf', bbox_inches='tight', dpi=150)
        print(f"Figure saved to: {output_path}")
    
        if return_fig:
            return fig
    
    
if __name__ == "__main__":
    print("Plotting test set predictions...")

    snapshot_path = "simple_setup_res/snapshot.pt"
    run_dir = "../runs/run_1"
    test_dems_dir = os.path.join(run_dir, 'test')
    config_path = os.path.join(run_dir, "stats", 'config.ini')

    plot_predictions(run_dir=run_dir,
                        n_test_sets=5,
                        variant='first',  # 'first' or 'random'
                        same_scale=False,  # False, 'row', or 'all'
                        figsize=(15, 10),
                        save_fig=True,
                        config_path=config_path,
                        snapshot_path=snapshot_path,
                        output_name='predictions_summary.pdf',
                        return_fig=False)

