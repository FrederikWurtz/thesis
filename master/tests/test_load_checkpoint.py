import numpy as np
from master.train.checkpoints import load_checkpoint, read_file_from_ini
from pprint import pprint
import os

if __name__ == "__main__":
    print("Current working directory:", os.getcwd())
    checkpoint_path = os.path.join("runs", "run_3","stats", "timing_info.ini")
    checkpoint_data = read_file_from_ini(checkpoint_path)
    print("Loaded checkpoint data:")
    pprint(checkpoint_data)
