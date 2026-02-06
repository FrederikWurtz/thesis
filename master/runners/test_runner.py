import os
import warnings
import argparse
import numpy as np

from master.data_sim.dataset_io import list_pt_files
from master.data_sim.generator import generate_and_save_data_pooled_multi_gpu
from master.train.trainer_core import DEMDataset


# 🔥 Must be set BEFORE importing torch
os.environ['TORCH_LOGS'] = '-all'  # Suppress all torch logging warnings
import torch

# 🔥 Suppress torch.compile() warnings - MUST BE BEFORE torch.multiprocessing.set_start_method
warnings.filterwarnings('ignore', category=UserWarning, module='torch._dynamo')
warnings.filterwarnings('ignore', category=UserWarning, module='torch._logging')
warnings.filterwarnings('ignore', message='.*Profiler function.*will be ignored.*')

from torch.distributed import destroy_process_group
from master.train.trainer_new import Trainer_multiGPU, Trainer_multiGPU_multi_band, Trainer_singleGPU
from master.train.trainer_core import ddp_setup, load_train_objs, prepare_dataloader, is_main
from master.train.checkpoints import save_file_as_ini, read_file_from_ini
from master.configs.config_utils import load_config_file

torch.multiprocessing.set_start_method('spawn', force=True)

def main(run_dir: str, test_on_separate_data: bool):

    if torch.cuda.is_available():
        if is_main():
            n_GPUs = torch.cuda.device_count()
            print(f"CUDA hardware detected - setting up testing on {n_GPUs} GPU(s)...")
        use_multiGPU = True
        ddp_setup()
    else:
        use_multiGPU = False

    sup_dir = "./runs"
    run_path = os.path.join(sup_dir, run_dir)
    if not os.path.exists(run_path):
        raise RuntimeError(f"Run directory {run_path} does not exist. Please initialize the run first.")
    
    config_path = os.path.join(run_path, 'stats', 'config.ini')
    config = load_config_file(config_path) # load default config
    snapshot_path = os.path.join(run_path, 'checkpoints', 'snapshot.pt')
    

    mean_std_path = os.path.join(run_path, 'stats', 'input_stats.ini')
    input_stats = read_file_from_ini(mean_std_path)
    train_mean = torch.tensor(float(input_stats["MEAN"]))
    train_std = torch.tensor([float(input_stats["STD"])])   

    train_set, val_set, test_set, model, optimizer = load_train_objs(config, run_path)
    train_loader = prepare_dataloader(train_set, config["BATCH_SIZE"], 
                                    num_workers=config["NUM_WORKERS_DATALOADER"], 
                                    prefetch_factor=config["PREFETCH_FACTOR"],
                                    multi_gpu=use_multiGPU)
    val_loader = prepare_dataloader(val_set, config["BATCH_SIZE"], 
                                    num_workers=config["NUM_WORKERS_DATALOADER"], 
                                    prefetch_factor=config["PREFETCH_FACTOR"],
                                    multi_gpu=use_multiGPU)
    if not test_on_separate_data: # normal testing
        test_loader = prepare_dataloader(test_set, config["BATCH_SIZE"], 
                                        num_workers=config["NUM_WORKERS_DATALOADER"], 
                                        prefetch_factor=config["PREFETCH_FACTOR"],
                                        multi_gpu=use_multiGPU)
    else: # testing on separate data
        alt_test_dir = os.path.join(run_path, 'train')
        print(f"Testing on separate data in directory: {alt_test_dir}")
        # check for existence, else generate data
        if not os.path.exists(alt_test_dir):
            os.makedirs(alt_test_dir, exist_ok=True)
            print("\n=== Generating New Dataset ===")
            # Set seed for reproducibility
            base_seed = config["BASE_SEED"] if "BASE_SEED" in config else 42
            torch.manual_seed(base_seed - 7) # -7 to differ from training seed
            np.random.seed((base_seed - 7) % (2**32 - 1)) # -7 to differ from training seed
            # create alternative test files
            config["USE_SEPARATE_VALTEST_PARS"] = True
            generate_and_save_data_pooled_multi_gpu(config, images_dir=alt_test_dir, n_dems=config["FLUID_VAL_DEMS"])
        
        test_set = list_pt_files(alt_test_dir)  # load test dataset
        print(f"Number of test samples in separate test set: {len(test_set)}")
        test_set = DEMDataset(test_set)
        test_loader = prepare_dataloader(test_set, config["BATCH_SIZE"], 
                                        num_workers=config["NUM_WORKERS_DATALOADER"], 
                                        prefetch_factor=config["PREFETCH_FACTOR"],
                                        multi_gpu=use_multiGPU)
            
    
    if use_multiGPU:
        if config["USE_MULTI_BAND"]:
            trainer = Trainer_multiGPU_multi_band(model, train_loader, optimizer, config, snapshot_path, train_mean, train_std, val_loader, test_loader)
        else:
            trainer = Trainer_multiGPU(model, train_loader, optimizer, config, snapshot_path, train_mean, train_std, val_loader, test_loader)
            
        if is_main():
            print("Starting testing...")
    else:
        trainer = Trainer_singleGPU(model, train_loader, optimizer, config, snapshot_path, train_mean, train_std, val_loader, test_loader)
        print("Starting testing...")

    global_test_loss, global_ame = trainer.test()
    
    if config["USE_MULTI_BAND"]:
        dem_ame, w_ame, theta_ame = global_ame
        if is_main():
            test_loss_dir = os.path.join(run_path, 'stats', 'test_results.ini')
            if test_on_separate_data:
                test_loss_dir = os.path.join(run_path, 'stats', 'alt_test_results.ini')
            save_file_as_ini({'TEST_LOSS': float(global_test_loss), 
                              'DEM_AME': float(dem_ame), 
                              'W_AME': float(w_ame), 
                              'THETA_AME': float(theta_ame)}, test_loss_dir)
            print("Test results saved.")
            print("Cleaning up...")
    else:
        if is_main():
            test_loss_dir = os.path.join(run_path, 'stats', 'test_results.ini')
            if test_on_separate_data:
                test_loss_dir = os.path.join(run_path, 'stats', 'alt_test_results.ini')
            save_file_as_ini({'TEST_LOSS': float(global_test_loss), 'TEST_AME': float(global_ame)}, test_loss_dir)
            print("Test results saved.")
            print("Cleaning up...")

    if use_multiGPU:
        destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir', nargs="?", type=str,
                        help='Directory of the trained model run containing stats and checkpoints.')
    parser.add_argument('--run_dir', type=str, dest='run_dir_flag', required=False,
                        help='Directory of the trained model run containing stats and checkpoints.')
    parser.add_argument('--test_on_separate_data', action='store_true',
                        help="Indicates to test on separate data, generated with separate validation/testing parameters.")
    args = parser.parse_args()

    # Support both positional and flag-based arguments
    run_dir = args.run_dir_flag or args.run_dir

    main(run_dir, test_on_separate_data=args.test_on_separate_data)