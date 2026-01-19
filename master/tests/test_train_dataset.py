import os
import numpy as np
from master.train.trainer_core import FluidDEMDataset
from master.configs.config_utils import load_config_file
from torch.utils.data import DataLoader
import torch
from master.train.trainer_new import prepare_dataloader, ddp_setup
from torch.distributed import destroy_process_group

config = load_config_file()
config["FLUID_N_DEMS"] = 2  # For testing, use a small number

print("Testing FluidDEMDataset with configuration:")
print(config['FLUID_N_DEMS'])

if __name__ == "__main__":

    ddp_setup()
    dataset = FluidDEMDataset(config=config)
    train_loader = prepare_dataloader(dataset, config["BATCH_SIZE"], 
                                num_workers=config["NUM_WORKERS_DATALOADER"], 
                                prefetch_factor=config["PREFETCH_FACTOR"])
    print("Dataset length:", len(dataset))

    for epoch in range(100,105):
        dataset.set_epoch(epoch)  # Set epoch to 5 for reproducibility
        print(f"Dataset epoch set to: {dataset.epoch}")
        sample = dataset[10]  # Get the first sample
        print("Sample keys:", [type(item) for item in sample])
        images, reflectance_maps, dem, meta = sample

        print(f"Images shape: {images.shape}")  # Expected: ([5, H, W]) for 5 images in the sample
        print(f"Reflectance maps shape: {reflectance_maps.shape}")  # Expected: ([5, H, W])
        print(f"DEM shape: {dem.shape}")  # Expected: ([H, W])
        print(f"Metadata shape: {meta.shape}")  # Expected: ([5, 5])


        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(5, 3, figsize=(12, 20))

        for i in range(5):
            axes[i, 0].imshow(images[i], cmap='gray', origin='lower')
            axes[i, 0].set_title(f'Image {i}')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(reflectance_maps[i], cmap='gray', origin='lower')
            axes[i, 1].set_title(f'Reflectance Map {i}')
            axes[i, 1].axis('off')
            
            if i == 0:
                axes[i, 2].imshow(dem[0], cmap='terrain', origin='lower')
                axes[i, 2].set_title('DEM')
                axes[i, 2].axis('off')

            else:
                axes[i, 2].axis('off')
                metadata_text = f"SUN_AZ: {meta[i, 0]:.1f}°\nSUN_EL: {meta[i, 1]:.1f}°\nCAM_AZ: {meta[i, 2]:.1f}°\nCAM_EL: {meta[i, 3]:.1f}°\nCAM_DIST: {meta[i, 4]:.1f}"
                axes[i, 2].text(0.5, 0.5, metadata_text, ha='center', va='center', fontsize=10)
        
            fig.suptitle(f"FluidDEMDataset Sample (Epoch {dataset.epoch})", fontsize=16, y=1.04)

        plt.tight_layout()
        save_dir = "master/tests/test_outputs"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'fluid_dem_dataset_sample_epoch_{dataset.epoch}_10th_sample_2nd_run.png')
        plt.savefig(save_path)
        print(f"Sample visualization saved to {save_path}")


    destroy_process_group()
    
