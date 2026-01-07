import numpy as np
from master.train import *
from master.configs.config_utils import load_config_file
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    npz_file_dir = os.path.join("runs", "debug_epoch1.npz")
    data = np.load(npz_file_dir)
    print("Loaded .npz file keys:", data.files)
    sample_id = 15  # Change this to visualize different samples
    images = data['images'][sample_id]  # First sample
    reflectance_maps = data['reflectance_maps'][sample_id]  # First sample
    dem = data['targets'][sample_id]  # First sample
    meta = data['meta'][sample_id]  # First sample

    print(f"Images shape: {images.shape}")  # Expected: ([5, H, W]) for 5 images in the sample
    print(f"Reflectance maps shape: {reflectance_maps.shape}")  # Expected: ([5, H, W])
    print(f"DEM shape: {dem.shape}")  # Expected: ([H, W])
    print(f"Metadata shape: {meta.shape}")  # Expected: ([5, 5])

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
    
