import numpy as np
from master.train.trainer_core import run_fluid_training
from master.data.datasets.fluid_demdataset import FluidDEMDataset
import matplotlib.pyplot as plt
import os
from master.configs.config_utils import load_config_file

config = load_config_file()
config["FLUID_TRAIN_DEMS"] = 10  # For testing, use a small number

print("Testing FluidDEMDataset with configuration:")
print(config['FLUID_TRAIN_DEMS'])

if __name__ == "__main__":

    run_dir = os.path.join("runs", "test_run")
    val_dir = os.path.join("runs", "val_run")
    test_dir = os.path.join("runs", "test_run")

    returned_values = run_fluid_training(config=config, 
                                        run_dir=run_dir, 
                                        val_dir=val_dir, 
                                        test_dir=test_dir, 
                                        new_training=True)


