import matplotlib.pyplot as plt
import os
import re
import torch
import random
import numpy as np
import subprocess
import shutil
import glob

from master.models.unet import UNet
from master.train.trainer_core import DEMDataset
from master.train.train_utils import normalize_inputs
from master.train.checkpoints import load_checkpoint, read_file_from_ini
from master.configs.config_utils import load_config_file


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


def plot_dataset(run_dir, n_sets=5, variant='random', save_fig=True, 
                 output_name='data_grid.pdf', return_fig=False):
    """
    Plot DEMs and their corresponding images from test datasets.
    
    Parameters:
    -----------
    run_dir : str
        Name of the run directory (e.g., 'test_lro')
    n_sets : int
        Number of datasets to plot (number of rows)
    variant : str
        Selection method: 'first' or 'random'
    save_fig : bool
        Whether to save the figure
    output_name : str
        Name of the output file
    return_fig : bool
        Whether to return the figure object
    """
    
    # Setup paths similar to plot_comprehensive_pt
    sup_dir = "runs/"
    run_path = os.path.join(sup_dir, run_dir)
    test_dems_dir = os.path.join(run_path, 'test')
    figures_dir = os.path.join(run_path, 'figures')
    
    # Ensure figures directory exists
    os.makedirs(figures_dir, exist_ok=True)
    
    # Find test files
    test_files = None
    if os.path.isdir(test_dems_dir):
        candidate_files = sorted(glob.glob(os.path.join(test_dems_dir, '*.pt')))
        if len(candidate_files) > 0:
            test_files = candidate_files
        else:
            raise FileNotFoundError(f"No .pt files found in '{test_dems_dir}'")
    else:
        raise FileNotFoundError(f"No test directory found at '{test_dems_dir}'")
    
    # Create dataset
    test_dataset = DEMDataset(test_files)
    
    # Select datasets based on variant
    print(f"Selecting {n_sets} datasets using variant: {variant}")
    available_indices = list(range(len(test_dataset)))
    
    if variant == 'first':
        selected_indices = list(range(min(n_sets, len(test_dataset))))
    elif variant == 'random':
        selected_indices = sorted(random.sample(available_indices, min(n_sets, len(test_dataset))))
        print("Selected dataset indices:", selected_indices)
    else:
        raise ValueError(f"Unknown variant: {variant}. Use 'first' or 'random'.")
    
    # Dynamic figure sizing: each subplot should be reasonably sized
    # Width: 6 columns (1 DEM + 5 images) * base_width
    # Height: n_sets rows * base_height
    base_width = 3.0  # Width per column in inches
    base_height = 3.0  # Height per row in inches
    fig_width = 6 * base_width
    fig_height = n_sets * base_height
    
    # Create figure with tight layout
    fig, axes = plt.subplots(n_sets, 6, figsize=(fig_width, fig_height))
    
    # Handle case where n_sets=1 (axes won't be 2D)
    if n_sets == 1:
        axes = axes.reshape(1, -1)
    
    # Plot each dataset
    for row_idx, test_idx in enumerate(selected_indices):
        # Load data
        images, reflectance_maps, target, meta = test_dataset[test_idx]
        
        # Convert to numpy
        target_np = target.squeeze().numpy()
        images_np = images.numpy()
        
        # Plot DEM (column 0)
        ax_dem = axes[row_idx, 0]
        ax_dem.imshow(target_np, cmap='terrain', aspect='auto')
        ax_dem.axis('off')
        
        # Plot images (columns 1-5)
        for img_idx in range(5):
            ax_img = axes[row_idx, img_idx + 1]
            if img_idx < images_np.shape[0]:
                ax_img.imshow(images_np[img_idx], cmap='gray', aspect='auto')
            ax_img.axis('off')
    
    # Remove spacing between subplots
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, 
                        wspace=0.02, hspace=0.02)
    
    # Save figure
    if save_fig:
        output_path = os.path.join(figures_dir, output_name)
        plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.05)
        print(f"Figure saved to: {output_path}")
    
    if return_fig:
        return fig
    else:
        plt.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Plot datasets with DEMs and images.")
    parser.add_argument('--run_dir', type=str, required=True,
                        help='Directory of the run (e.g., test_lro)')
    parser.add_argument('--n_sets', type=int, default=5,
                        help='Number of datasets to plot')
    parser.add_argument('--variant', type=str, default='random',
                        help="Selection method: 'first' or 'random'")
    parser.add_argument('--output_name', type=str, default='data_grid.pdf',
                        help='Output filename')
    args = parser.parse_args()
    
    plot_dataset(run_dir=args.run_dir, n_sets=args.n_sets, 
                 variant=args.variant, output_name=args.output_name)
