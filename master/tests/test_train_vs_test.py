import os
os.environ["OMP_NUM_THREADS"] = "2"

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


from master.configs.config_utils import load_config_file
from master.models.unet import UNet
torch.multiprocessing.set_start_method('spawn', force=True)

def main(run_dir: str):

    ddp_setup()
    
    sup_dir = "./runs"
    run_path = os.path.join(sup_dir, run_dir)
    if not os.path.exists(run_path):
        raise RuntimeError(f"Run directory {run_path} does not exist. Please initialize the run first.")
    
    config_path = os.path.join(run_path, 'stats', 'config.ini')
    config = load_config_file(config_path) # load default config
    snapshot_path = os.path.join(run_path, 'checkpoints', 'snapshot.pt')

    mean_std_path = os.path.join(run_path, 'stats', 'input_stats.ini')
    input_stats = read_file_from_ini(mean_std_path)
    train_mean = torch.tensor([float(input_stats['MEAN'][i]) for i in range(len(input_stats['MEAN']))])
    train_std = torch.tensor([float(input_stats['STD'][i]) for i in range(len(input_stats['STD']))])

    train_set, val_set, test_set, model, optimizer = load_train_objs(config, run_path)

    # Prepare data loaders
    # Create a global epoch counter, to help with reproducibility across dataloader workers - issue arrises when using num_workers>0
    GLOBAL_EPOCH = 0

    train_loader = prepare_dataloader(train_set, config["BATCH_SIZE"], 
                                      num_workers=config["NUM_WORKERS_DATALOADER"], 
                                      prefetch_factor=config["PREFETCH_FACTOR"])
    val_loader = prepare_dataloader(val_set, config["BATCH_SIZE"], 
                                    num_workers=config["NUM_WORKERS_DATALOADER"], 
                                    prefetch_factor=config["PREFETCH_FACTOR"])
    test_loader = prepare_dataloader(test_set, config["BATCH_SIZE"], 
                                     num_workers=config["NUM_WORKERS_DATALOADER"], 
                                     prefetch_factor=config["PREFETCH_FACTOR"])
    
    trainer = Trainer(model, train_loader, optimizer, config, snapshot_path, train_mean, train_std, val_loader, test_loader)

    # if is_main():
    #     print("Starting testing...")
    # # Actal test dataset is passed here
    # global_test_loss, global_ame = trainer.test()
    # # New addition: also evaluate on training data, using same test function
    # global_train_loss, global_train_ame = trainer.test(data_loader=train_loader)
    # Also evaluate on train data using train_function

    # To access the current epoch value, use the variable 'epoch' directly:
    # print(f"trainer.train_data.epoch: {trainer.train_data.dataset.epoch}")
    epoch = 100  # arbitrary epoch number for logging
    train_loss_via_train = trainer._run_epoch(epoch=epoch, return_val=True)
    print(f"trainer.train_data.epoch: {trainer.train_data.dataset.epoch}")
    

    # # print values nicely
    # if is_main():
    #     print(f"Train Loss via test(): {global_train_loss:.2e}, Train AME via test(): {global_train_ame:.6f}")
    #     print(f"Train Loss via train(): {train_loss_via_train:.2e}, at epoch {epoch}")
    #     print(f"Test Loss: {global_test_loss:.2e}, Test AME: {global_ame:.6f}")





    destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', type=str, required=True)
    args = parser.parse_args()

    main(args.run_dir)