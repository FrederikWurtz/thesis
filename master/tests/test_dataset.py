import numpy as np
from master.train import *
from master.configs.config_utils import load_config_file

config = load_config_file()
config["FLUID_N_DEMS"] = 2  # For testing, use a small number

print("Testing FluidDEMDataset with configuration:")
print(config['FLUID_N_DEMS'])

if __name__ == "__main__":
    dataset = FluidDEMDataset(config=config)
    print("Dataset length:", len(dataset))
    sample = dataset[0]  # Get the first sample
    print("Sample keys:", [type(item) for item in sample])
    images, reflectance_maps, dem, meta = sample

    print(f"Images shape: {images.shape}")  # Expected: ([5, H, W]) for 5 images in the sample
    print(f"Reflectance maps shape: {reflectance_maps.shape}")  # Expected: ([5, H, W])
    print(f"DEM shape: {dem.shape}")  # Expected: ([H, W])
    print(f"Metadata shape: {meta.shape}")  # Expected: ([5, 5])


    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(5, 3, figsize=(12, 20))

    for i in range(5):
        axes[i, 0].imshow(images[i], cmap='gray')
        axes[i, 0].set_title(f'Image {i}')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(reflectance_maps[i], cmap='gray')
        axes[i, 1].set_title(f'Reflectance Map {i}')
        axes[i, 1].axis('off')
        
        if i == 0:
            axes[i, 2].imshow(dem[0], cmap='viridis')
            axes[i, 2].set_title('DEM')
            axes[i, 2].axis('off')
        else:
            axes[i, 2].axis('off')
            metadata_text = f"SUN_AZ: {meta[i, 0]:.1f}°\nSUN_EL: {meta[i, 1]:.1f}°\nCAM_AZ: {meta[i, 2]:.1f}°\nCAM_EL: {meta[i, 3]:.1f}°\nCAM_DIST: {meta[i, 4]:.1f}"
            axes[i, 2].text(0.5, 0.5, metadata_text, ha='center', va='center', fontsize=10)

    plt.tight_layout()
    plt.show()
    
