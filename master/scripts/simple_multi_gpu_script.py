import json
import os
# 🔥 Must be set BEFORE importing torch
os.environ['TORCH_LOGS'] = '-all'  # Suppress all torch logging warnings
import sys
import warnings

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from master.train.trainer_core import FluidDEMDataset, DEMDataset
import time
import argparse

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import destroy_process_group
from master.train.trainer_new import Trainer, ddp_setup, load_train_objs, prepare_dataloader, is_main
from master.train.checkpoints import save_file_as_ini, read_file_from_ini

# 🔥 Suppress torch.compile() warnings - MUST BE BEFORE torch.multiprocessing.set_start_method
warnings.filterwarnings('ignore', category=UserWarning, module='torch._dynamo')
warnings.filterwarnings('ignore', category=UserWarning, module='torch._logging')
warnings.filterwarnings('ignore', message='.*Profiler function.*will be ignored.*')

from master.configs.config_utils import load_config_file
from master.models.unet import UNet
torch.multiprocessing.set_start_method('spawn', force=True)

def main(run_dir: str, config_override_file: str = None, new_run: bool = False):
    t0_main = time.time()

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

    if new_run:
        if is_main():
            print("Starting a new run, resetting relevant parameters...")
            # Remove all files in checkpoint folder
            checkpoint_dir = os.path.join(run_path, 'checkpoints')
            if os.path.exists(checkpoint_dir):
                for filename in os.listdir(checkpoint_dir):
                    file_path = os.path.join(checkpoint_dir, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                if is_main():
                    print(f"Cleared checkpoint directory: {checkpoint_dir}")


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

    if is_main():
        number_of_gpus = torch.cuda.device_count()
        print(f"Number of GPUs detected: {number_of_gpus}")
        equipment_info_path = os.path.join(run_path, 'stats', 'equipment_info.ini')
        save_file_as_ini({'NUM_GPUS': [str(number_of_gpus)]}, equipment_info_path)
        print("Everything set up")
    trainer.train(config["EPOCHS"])

    # Skip testing if profiling
    if config.get("SKIP_TEST", False):
        if is_main():
            print("Skipping test phase (profiling mode)")
        destroy_process_group()
        return

    print("Training complete.")
    print("Testing on test dataset...")
    global_test_loss, global_ame = trainer.test()
    
    test_loss_dir = os.path.join(run_path, 'stats', 'test_results.ini')
    save_file_as_ini({'TEST_LOSS': [str(global_test_loss)], 'TEST_AME': [str(global_ame)]}, test_loss_dir)
    print(f"Test Loss: {global_test_loss:.6f}, Test AME: {global_ame:.6f}")
    print("Test results saved.")

    t1_main = time.time()
    total_time = t1_main - t0_main
    print(f"Total time taken (including training and testing): {total_time/60:.2f} minutes")
    total_time_path = os.path.join(run_path, 'stats', 'total_time.ini')
    save_file_as_ini({'TOTAL_TIME_MINUTES': [str(total_time/60)], 
                                       'TOTAL_TIME_SECONDS': [str(total_time)], 
                                       'TOTAL_TIME_HRS': [str(total_time/3600)]}, total_time_path)


    print("Cleaning up...")

    destroy_process_group()


if __name__ == "__main__":
    # Support both old and new usage
    parser = argparse.ArgumentParser(description="Multi-GPU training script")
    parser.add_argument("run_dir", nargs="?", help="Run directory name")
    parser.add_argument("--run_dir", dest="run_dir_flag", help="Run directory name (flag form)")
    parser.add_argument("config_override_file", nargs="?", help="Config override file path")
    parser.add_argument("--config_override_file", dest="config_override_file_flag", help="Config override file path (flag form)")
    parser.add_argument("--new_run", action="store_true", help="Indicates a new run")
    
    args = parser.parse_args()
    
    # Support both positional and flag-based arguments
    run_dir = args.run_dir_flag or args.run_dir
    config_override_file = args.config_override_file_flag or args.config_override_file
    
    if not run_dir:
        print("Usage: torchrun ... simple_multi_gpu_script.py <run_dir> [config_override_file]")
        print("   or: torchrun ... simple_multi_gpu_script.py --run_dir <run_dir> [--config_override_file <file>]")
        sys.exit(1)
    
    main(run_dir, config_override_file, new_run=args.new_run)