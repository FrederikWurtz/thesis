import numpy as np
from master.data_sim.generator import generate_and_return_data, generate_and_save_data
from master.configs.config_utils import load_config_file

config = load_config_file()

if __name__ == "__main__":
    images, reflectance_maps, dem_np, metas = generate_and_return_data(config)
    print(f"Generated {len(images)} images and {len(reflectance_maps)} reflectance maps.")
    print(f"DEM shape: {dem_np.shape}, dtype: {dem_np.dtype}")
    print(f"Metadata length: {len(metas)}, length of first metadata entry: {len(metas[0])}")

    # Optionally save the generated data
    # generate_and_save_data(config=config, path="generated_data")
