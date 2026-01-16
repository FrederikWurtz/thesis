import json
import os
import sys

import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from master.models.losses import calculate_total_loss
from master.train.train_utils import normalize_inputs
from master.train.trainer_core import FluidDEMDataset, DEMDataset
import time
import argparse

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import destroy_process_group
from master.train.trainer_new import Trainer, ddp_setup, load_train_objs, prepare_dataloader, is_main
from master.train.checkpoints import save_file_as_ini, read_file_from_ini

import matplotlib.pyplot as plt

from master.configs.config_utils import load_config_file
from master.models.unet import UNet
torch.multiprocessing.set_start_method('spawn', force=True)

def plot_output_target(output, target, figsize=(10,5), show_plot=True):
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    ax1, ax2 = axes
    im1 = ax1.imshow(output, cmap='terrain')
    ax1.set_title('Predicted Output')
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    im2 = ax2.imshow(target, cmap='terrain')
    ax2.set_title('Target')
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    if show_plot:
        plt.show()
    return fig 

def main(run_dir: str, config_override_file: str = None):

    ddp_setup()

    os.environ['OMP_NUM_THREADS'] = '2'  # Set number of OpenMP threads
    
    sup_dir = "./runs"
    run_path = os.path.join(sup_dir, run_dir)
    if not os.path.exists(run_path):
        raise RuntimeError(f"Run directory {run_path} does not exist. Please initialize the run first.")
    
    config_path = os.path.join(run_path, 'stats', 'config.ini')
    config = load_config_file(config_path) # load default config

    # Apply overrides if provided
    if config_override_file and os.path.exists(config_override_file):
        with open(config_override_file, 'r') as f:
            overrides = json.load(f)
        config.update(overrides)
        if is_main():
            print(f"Applied config overrides: {overrides}")

    snapshot_path = os.path.join(run_path, 'checkpoints', 'snapshot.pt')

    mean_std_path = os.path.join(run_path, 'stats', 'input_stats.ini')
    input_stats = read_file_from_ini(mean_std_path)
    train_mean = torch.tensor([float(input_stats['MEAN'][i]) for i in range(len(input_stats['MEAN']))])
    train_std = torch.tensor([float(input_stats['STD'][i]) for i in range(len(input_stats['STD']))])

    train_set, val_set, test_set, model, optimizer = load_train_objs(config, run_path)
    train_loader = prepare_dataloader(train_set, config["BATCH_SIZE"], 
                                      num_workers=config["NUM_WORKERS_DATALOADER"], 
                                      prefetch_factor=config["PREFETCH_FACTOR"])
    val_loader = prepare_dataloader(val_set, config["BATCH_SIZE"], 
                                    num_workers=config["NUM_WORKERS_DATALOADER"], 
                                    prefetch_factor=config["PREFETCH_FACTOR"])
    

    trainer = Trainer(model, train_loader, optimizer, config, snapshot_path, train_mean, train_std, val_loader)


    for batch_idx, (images, reflectance_maps, targets, metas) in enumerate(trainer.train_data):
        if batch_idx > 0:
            break  # Just one batch for inspection
        
        images = images.to(trainer.gpu_id)
        metas = metas.to(trainer.gpu_id)
        reflectance_maps = reflectance_maps.to(trainer.gpu_id)
        images = normalize_inputs(images, trainer.train_mean, trainer.train_std)

        targets = targets.to(trainer.gpu_id)

        # Calculate loss, which is returned as a mean over the batch
        with torch.autocast('cuda', enabled=trainer.use_amp, dtype=trainer.dtype):
            outputs = trainer.model(images, metas, target_size=targets.shape[-2:])
            # loss = calculate_total_loss(outputs, targets, reflectance_maps, metas, device=trainer.gpu_id,
            #                                 camera_params=trainer.config["CAMERA_PARAMS"], hapke_params=trainer.config["HAPKE_KWARGS"],
            #                                 w_grad=trainer.config["W_GRAD"], w_refl=trainer.config["W_REFL"], w_mse=trainer.config["W_MSE"])
        
        # Simple backward - no scaler!
    
    # Inspect model outputs
    # Create plot of outputs

        if is_main():

            output_dir = os.path.join(run_path, 'debug_outputs')
            os.makedirs(output_dir, exist_ok=True)

            # for the first, calculate statistics of output
            first_output = outputs[0].detach().cpu().numpy()[0]
            print(f"Output stats - min: {first_output.min():.4f}, max: {first_output.max():.4f}, mean: {first_output.mean():.4f}, std: {first_output.std():.4f}")
            # look for nans
            n_nans = np.isnan(first_output).sum()
            print(f"Number of NaNs in first output: {n_nans}")

            for i in range(5): # First 5 samples in batch
                output_np = outputs[i].detach().cpu().numpy()[0]
                target_np = targets[i].detach().cpu().numpy()[0]

                fig = plot_output_target(output_np, target_np, figsize=(10,5), show_plot=False)
                output_path = os.path.join(output_dir, f'batch{batch_idx}_sample{i}_dem_shadow.png')
                fig.savefig(output_path)
                plt.close(fig)
                print(f"Saved debug plot to {output_path}")

        destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Debug trained DEM model by inspecting outputs on a batch of data.')
    parser.add_argument('run_dir', type=str, nargs='?', help='Directory name of the run to debug (inside ./runs/).')
    parser.add_argument('--run_dir', type=str, help='Directory name of the run to debug (inside ./runs/).')
    parser.add_argument('--config_override', type=str, default=None, help='Path to JSON file with config overrides.')
    parser.add_argument('--new_run', action='store_true', help='Whether this is a new run (no checkpoints).')

    args = parser.parse_args()
    main(run_dir=args.run_dir, config_override_file=args.config_override)