import os
import warnings
import argparse


# 🔥 Must be set BEFORE importing torch
os.environ["OMP_NUM_THREADS"] = "2"
os.environ['TORCH_LOGS'] = '-all'  # Suppress all torch logging warnings
import torch

# 🔥 Suppress torch.compile() warnings - MUST BE BEFORE torch.multiprocessing.set_start_method
warnings.filterwarnings('ignore', category=UserWarning, module='torch._dynamo')
warnings.filterwarnings('ignore', category=UserWarning, module='torch._logging')
warnings.filterwarnings('ignore', message='.*Profiler function.*will be ignored.*')

from torch.distributed import destroy_process_group
from master.train.trainer_new import Trainer, ddp_setup, load_train_objs, prepare_dataloader, is_main
from master.train.checkpoints import save_file_as_ini, read_file_from_ini
from master.configs.config_utils import load_config_file

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

    if is_main():
        print("Starting testing...")
    global_test_loss, global_ame = trainer.test()
    
    if is_main():
        test_loss_dir = os.path.join(run_path, 'stats', 'test_results.ini')
        save_file_as_ini({'TEST_LOSS': float(global_test_loss), 'TEST_AME': float(global_ame)}, test_loss_dir)
        print(f"Test Loss: {global_test_loss:.2e}, Test AME: {global_ame:.6f}")
        print("Test results saved.")
        print("Cleaning up...")

    destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir', nargs="?", type=str,
                        help='Directory of the trained model run containing stats and checkpoints.')
    parser.add_argument('--run_dir', type=str, dest='run_dir_flag', required=False,
                        help='Directory of the trained model run containing stats and checkpoints.')
    args = parser.parse_args()

    # Support both positional and flag-based arguments
    run_dir = args.run_dir_flag or args.run_dir

    main(run_dir)