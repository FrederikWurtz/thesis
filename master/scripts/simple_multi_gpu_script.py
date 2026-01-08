
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from master.train.trainer_core import FluidDEMDataset
import time
import argparse

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from master.train.trainer_new import Trainer, ddp_setup, load_train_objs, prepare_dataloader

import os
from master.configs.config_utils import load_config_file
from master.models.losses import calculate_total_loss
from master.models.unet import UNet




def main(run_dir: str):

    # Check we are in distributed mode, else raise error
    if not torch.distributed.is_initialized():
        raise RuntimeError("Distributed not initialized. Please run this script using torch.distributed.launch or torchrun for multi-GPU training.")
    
    ddp_setup()
    sup_dir = "./runs"
    run_path = os.path.join(sup_dir, run_dir)
    if not os.path.exists(run_path):
        raise RuntimeError(f"Run directory {run_path} does not exist. Please initialize the run first.")
    
    config_path = os.path.join(run_path, 'stats', 'config.ini')
    config = load_config_file(config_path) # load default config
    snapshot_path = os.path.join(run_path, 'checkpoints', 'snapshot.pt')


    dataset, model, optimizer = load_train_objs(config, fluid = True)
    train_data = prepare_dataloader(dataset, config["BATCH_SIZE"])
    trainer = Trainer(model, train_data, optimizer, config, snapshot_path)
    print("Everything set up")
    trainer.train(config["EPOCHS"])
    
    destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', type=str, required=True)
    args = parser.parse_args()

    main(args.run_dir)