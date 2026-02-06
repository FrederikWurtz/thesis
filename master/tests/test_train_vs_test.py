import json
import multiprocessing
import os
import subprocess
import time
import argparse
import warnings

# 🔥 Must be set BEFORE importing torch
os.environ['TORCH_LOGS'] = '-all'  # Suppress all torch logging warnings
# 🔥 Suppress torch.compile() warnings - MUST BE BEFORE torch.multiprocessing.set_start_method
warnings.filterwarnings('ignore', category=UserWarning, module='torch._dynamo')
warnings.filterwarnings('ignore', category=UserWarning, module='torch._logging')
warnings.filterwarnings('ignore', message='.*Profiler function.*will be ignored.*')

import torch

from torch.distributed import destroy_process_group
from master.train.trainer_new import Trainer_multiGPU, Trainer_multiGPU_multi_band
from master.train.trainer_core import ddp_setup, load_train_objs, prepare_dataloader, is_main
from master.train.checkpoints import save_file_as_ini, read_file_from_ini
from master.configs.config_utils import load_config_file


torch.multiprocessing.set_start_method('spawn', force=True)


def main(run_dir: str, config_override_file: str = None, new_run: bool = False):
    t0_main = time.time()
    
    ddp_setup()

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
    train_mean = torch.tensor(input_stats['MEAN'])
    train_std = torch.tensor(input_stats['STD'])

    # value for each process to share current epoch - otherwise the deterministic randomness will not be set correctly!
    EPOCH_SHARED = multiprocessing.Value('i', 0)  # 'i' means integer

    train_set, val_set, test_set, model, optimizer = load_train_objs(config, run_path, epoch_shared=EPOCH_SHARED) 
    
    train_loader = prepare_dataloader(train_set, config["BATCH_SIZE"], 
                                      num_workers=config["NUM_WORKERS_DATALOADER"], 
                                      prefetch_factor=config["PREFETCH_FACTOR"],
                                      use_shuffle=False
                                      )
                                      
    
    val_loader = prepare_dataloader(val_set, config["BATCH_SIZE"], 
                                    num_workers=config["NUM_WORKERS_DATALOADER"], 
                                    prefetch_factor=config["PREFETCH_FACTOR"],
                                    use_shuffle=False)
    
    test_loader = prepare_dataloader(test_set, config["BATCH_SIZE"], 
                                     num_workers=config["NUM_WORKERS_DATALOADER"], 
                                     prefetch_factor=config["PREFETCH_FACTOR"],
                                     use_shuffle=False)

    if config["USE_MULTI_BAND"]:
        if is_main():
            print("Using multi-band model for training...")
        trainer = Trainer_multiGPU_multi_band(model, train_loader, optimizer, config, snapshot_path, train_mean, train_std, val_data=val_loader, test_data=test_loader)
    else:
        trainer = Trainer_multiGPU(model, train_loader, optimizer, config, snapshot_path, train_mean, train_std, val_data=val_loader, test_data=test_loader)

    if is_main():
        number_of_gpus = torch.cuda.device_count()
        print(f"Number of GPUs detected: {number_of_gpus}")
        equipment_info_path = os.path.join(run_path, 'stats', 'equipment_info.ini')
        save_file_as_ini({'NUM_GPUS': [str(number_of_gpus)]}, equipment_info_path)
        print("Everything set up")

    epoch = 10 # random epoch to make everything equal
    trainset_via_train_loss = trainer._run_epoch(epoch, return_val=True)  # run one epoch to initialize everything
    
    trainer.val_data = train_loader  # set validation data to training data for testing purposes
    trainset_via_val_loss = trainer._validate(epoch)  # run validation on training data
    
    trainer.test_data = train_loader  # set test data to training data for testing purposes
    trainset_via_test_loss, ame = trainer.test()  # run test on training data

    if is_main():
        print("\n=== Summary of Losses on Training Data ===")
        print(f"Training Loss: {trainset_via_train_loss:.10f}")
        print(f"Validation Loss: {trainset_via_val_loss:.10f}")
        print(f"Test Loss: {trainset_via_test_loss:.10f}")
        
        
    trainer.train_data = val_loader  # set training data to validation data for testing purposes
    valset_via_train_loss = trainer._run_epoch(epoch, return_val=True)  # run one epoch to initialize everything
    
    trainer.val_data = val_loader  # set validation data to validation data for testing purposes
    valset_via_val_loss = trainer._validate(epoch)  # run validation on validation
    
    trainer.test_data = val_loader  # set test data to validation data for testing purposes
    valset_via_test_loss, ame = trainer.test()  # run test on validation data
    
    if is_main():
        print("\n=== Summary of Losses on Validation Data ===")
        print(f"Training Loss: {valset_via_train_loss:.10f}")
        print(f"Validation Loss: {valset_via_val_loss:.10f}")
        print(f"Test Loss: {valset_via_test_loss:.10f}")

    print("Cleaning up...")

    destroy_process_group()


if __name__ == "__main__":
    # Support both old and new usage
    parser = argparse.ArgumentParser(description="Multi-GPU training script")
    parser.add_argument("run_dir", nargs="?", help="Run directory name")
    parser.add_argument("--run_dir", dest="run_dir_flag", help="Run directory name (flag form)", required=False)
    parser.add_argument("config_override_file", nargs="?", help="Config override file path")
    parser.add_argument("--config_override_file", dest="config_override_file_flag", help="Config override file path (flag form)")
    parser.add_argument("--new_run", action="store_true", help="Indicates a new run")
    
    args = parser.parse_args()
    
    # Support both positional and flag-based arguments
    run_dir = args.run_dir_flag or args.run_dir
    config_override_file = args.config_override_file_flag or args.config_override_file
    
    main(run_dir, config_override_file, new_run=args.new_run)