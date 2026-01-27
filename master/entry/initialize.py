from master.configs.config_utils import load_config_file, create_folder_structure
from master.data_sim.generator import generate_and_save_data_pooled_multi_gpu
from master.train.trainer_core import DEMDataset
from master.data_sim.dataset_io import list_pt_files
from master.train.train_utils import compute_input_stats, round_list
from master.train.checkpoints import save_file_as_ini

from torch.utils.data import DataLoader
import os
import shutil
import argparse
import numpy as np
import torch
import subprocess



def _parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('run_dir', nargs="?", type=str,
                   help='Name of the run directory to use or create.')
    p.add_argument('--run_dir', type=str, dest='run_dir_flag', required=False,
                   help='(Optional) Name of the run directory to use or create (overrides positional argument).')
    p.add_argument('--new_run', action='store_true', default=False,
                   help='If set, creates a new run directory (removes existing if present).')
    p.add_argument('--run_training', action='store_true', default=False,
                   help='If set, starts training after initialization.')
    p.add_argument('--skip_data_gen', action='store_true', default=False,
                   help='If set, skips data generation (useful for continuing runs).')
    p.add_argument('--use_LRO_dems', type=bool, default=None,
                   help='Whether to use LRO DEMs for data generation (overrides config).')
    p.add_argument('--epochs', type=int, default=None,
                   help='Number of training epochs (overrides config).')
    p.add_argument('--num_workers', type=int, default=None,
                   help='Number of worker processes for data loading (overrides config).')
    p.add_argument('--lr_patience', type=int, default=None,
                   help='Number of epochs with no improvement after which learning rate will be reduced (overrides config).')
    p.add_argument('--lr', type=float, default=None,
                   help='Initial learning rate (overrides config).')
    p.add_argument('--batch_size', type=int, default=None,
                   help='Batch size for training and validation (overrides config).')
    p.add_argument('--w_mse', type=float, default=None,
                   help='Weight for MSE loss component (overrides config).')
    p.add_argument('--w_grad', type=float, default=None,
                   help='Weight for gradient loss component (overrides config).')
    p.add_argument('--w_refl', type=float, default=None,
                   help='Weight for reflectance loss component (overrides config).')
    p.add_argument('--fluid_train_dems', type=int, default=None,
                   help='Number of DEMs to use for training set (overrides config).')
    p.add_argument('--fluid_val_dems', type=int, default=None,
                   help='Number of DEMs to use for validation set (overrides config).')
    p.add_argument('--fluid_test_dems', type=int, default=None,
                   help='Number of DEMs to use for test set (overrides config).')
    p.add_argument('--images_per_dem', type=int, default=None,
                   help='Number of images to generate per DEM (overrides config).')
    
    # Support both positional and flag-based arguments
    return p.parse_args(argv)

def main(argv=None):
    args = _parse_args(argv)
    run_dir = args.run_dir_flag or args.run_dir
    config = load_config_file() # load default config
    config["RUN_DIR"] = run_dir
    # Determine run directory, and parse new/skip_data_gen flags
    run_path = os.path.join(config["SUP_DIR"], run_dir)

    if os.path.exists(run_path) and not os.path.exists(os.path.join(run_path, 'stats')):
        # Existing run directory found, but no stats subdir
        shutil.rmtree(run_path)  # remove incomplete run directory
        print(f"Removed incomplete run directory at {run_path}.")

    if os.path.exists(run_path):
        print("Existing run directory detected.")
        # Existing run directory found
        if args.new_run is True:
            # User requested new run
            if not args.skip_data_gen:
                # Remove existing run directory entirely
                print(60*"=")
                print("New run flag set. Removing existing run directory...")
                try:
                    if os.path.isdir(run_path):
                        shutil.rmtree(run_path)
                        print(f"Removed existing run directory: {run_path}")
                except Exception as e:
                    print(f"Failed to remove run_path {run_path}: {e}")
                print(60*"=")
                run_path, val_path, test_path = create_folder_structure(config)
            elif args.skip_data_gen:
                # User requested to skip data generation, so keep existing val/test files but remove old training data
                print(60*"=")
                print("New run flag set but skipping data generation. Continuing with existing run directory, but removing old training data...")
                print(f"Using run directory: {run_path}")
                print(60*"=")
                val_path = os.path.join(run_path, 'val')
                test_path = os.path.join(run_path, 'test')
                # Remove old training data if present
                checkpoint_path = os.path.join(run_path, 'checkpoints')
                stats_path = os.path.join(run_path, 'stats')
                if os.path.exists(stats_path):
                    try:
                        shutil.rmtree(stats_path)
                    except Exception as e:
                        print(f"Could not remove stats directory {stats_path}: {e}")
                if os.path.exists(checkpoint_path):
                    try:
                        shutil.rmtree(checkpoint_path)
                    except Exception as e:
                        print(f"Could not remove checkpoint directory {checkpoint_path}: {e}")

                # also 

        elif args.new_run is False:
            # Continuing existing run
            print(60*"=")
            print("Existing directory found. Continuing training for existing run...")
            print(f"Using run directory: {run_path}")
            print(60*"=")

            stats_path = os.path.join(run_path, 'stats')
            os.makedirs(stats_path, exist_ok=True)
            # reload config from existing run directory
            config_path = os.path.join(stats_path, 'config.ini')
            config = load_config_file(config_path) # reload config from existing run

            val_path = os.path.join(run_path, 'val')
            test_path = os.path.join(run_path, 'test')
    else:
        # No existing run directory found; create new run
        print(60*"=")
        print(f"No existing run directory found.")
        print(f"Creating new run at: {run_path}")
        print(60*"=")
        run_path, val_path, test_path = create_folder_structure(config)
        args.new_run = True


    # apply CLI overrides to config by uppercasing arg names (only if key exists in config)
    allowed_cli_only = {"NEW_RUN", "SKIP_DATA_GEN", "USE_LRO_DEMS", "RUN_TRAINING"}
    for arg_name, val in vars(args).items():
        if val is None:
            continue
        cfg_key = arg_name.upper()
        if cfg_key not in config and cfg_key not in allowed_cli_only:
            print(f"⚠ Warning: Unknown config key '{cfg_key}' from CLI arguments; ignoring.")
            continue
        if isinstance(val, bool):
            # print(f"Overriding config key '{cfg_key}' from CLI with new bool: {val}")
            config[cfg_key] = val
        else:
            # print(f"Overriding config key '{cfg_key}' from CLI with value: {val}")
            config[cfg_key] = val
    
    print("LRO DEM usage set to:", end=" ")
    print(config["USE_LRO_DEMS"])

    # If run is found to be new, generate data
    if args.new_run:
        if not args.skip_data_gen: # unless user requests to skip data generation
            print("\n=== Generating New Dataset ===")
            # Set seed for reproducibility
            base_seed = config["BASE_SEED"] if "BASE_SEED" in config else 42
            torch.manual_seed(base_seed - 1) # -1 to differ from training seed
            np.random.seed((base_seed - 1) % (2**32 - 1)) # -1 to differ from training seed
            # create validation files
            config["USE_SEPARATE_VALTEST_PARS"] = True  # ensure separate val/test params
            print(f"Using alternative val/test parameters: {config['USE_SEPARATE_VALTEST_PARS']}")
            generate_and_save_data_pooled_multi_gpu(config, images_dir=val_path, n_dems=config["FLUID_VAL_DEMS"])
            # also calculate and save mean reflectance map over validation set
            val_files = list_pt_files(val_path)
            val_ds = DEMDataset(val_files, config=config)
            pin_memory = True if torch.cuda.is_available() else False
            val_loader = DataLoader(val_ds, batch_size=config["BATCH_SIZE"], shuffle=False, num_workers=4, pin_memory=pin_memory)
            print("\n=== Computing Input Statistics ===")
            train_mean, train_std = compute_input_stats(val_loader, images_per_dem=config["IMAGES_PER_DEM"])
            print(f"Mean: {round_list(train_mean.tolist(), 10)}")
            print(f"Std: {round_list(train_std.tolist(), 10)}")

            # create test files
            # also set different seed for test set generation
            torch.manual_seed(base_seed - 10) # -10 to differ from training seed
            np.random.seed((base_seed - 10) % (2**32 - 1)) # -10 to differ from training seed
            generate_and_save_data_pooled_multi_gpu(config, images_dir=test_path, n_dems=config["FLUID_TEST_DEMS"])
            print("Dataset generation complete.\n")

            mean_std_path = os.path.join(run_path, 'stats', 'input_stats.ini')
            save_file_as_ini({'MEAN': train_mean.tolist(), 'STD': train_std.tolist()}, mean_std_path)

            # plot LRO sampling distributions for val and test sets
            config["USE_SEPARATE_VALTEST_PARS"] = False  # reset to default for training data generation
            print(f"Using alternative val/test parameters: {config['USE_SEPARATE_VALTEST_PARS']}")
            # reset seeds for training data generation - only used to plot LRO sampling distributions
            torch.manual_seed(base_seed) 
            np.random.seed(base_seed % (2**32 - 1))
            train_path = os.path.join(run_path, 'train')
            os.makedirs(train_path, exist_ok=True)
            generate_and_save_data_pooled_multi_gpu(config, images_dir=train_path, n_dems=config["FLUID_TRAIN_DEMS"])

            import sys
            env = os.environ.copy()
            cmd = [
                sys.executable,
                "master/validate/plot_lro_sampling.py",
                run_dir,
                "val",
                "train"
            ]
            subprocess.run(cmd, env=env)

        elif args.skip_data_gen:
            pass  # skip data generation as per user request

    config["SAVE_LRO_METAS"] = False  
    print("SAVE_LRO_METAS set to:", end=" ")
    print(config["SAVE_LRO_METAS"])
    config_path = os.path.join(run_path, 'stats', 'config.ini')
    save_file_as_ini(config, config_path) # save final config to run directory

    # also create dirs for stats, figures and checkpoints if not existing
    stats_path = os.path.join(run_path, 'stats')
    os.makedirs(stats_path, exist_ok=True)
    figs_path = os.path.join(run_path, 'figures')
    os.makedirs(figs_path, exist_ok=True)
    checkpoint_path = os.path.join(run_path, 'checkpoints')
    os.makedirs(checkpoint_path, exist_ok=True)


    # Run validation plotting commands
    subprocess.run(["python", "master/entry/plot.py", "--run_dir", args.run_dir])
    subprocess.run(["python", "master/entry/plot.py", "--run_dir", args.run_dir, "--use_train_set"])

    if not args.run_training:
        print(f"Initialization complete, and data visualization done. You can now run the training script using run directory \"{args.run_dir}\".")
    
    if args.run_training:
        # Start training
        print("Starting training...")
        from master.entry.train import run_train
        run_train(run_dir=run_dir, new_run=args.new_run)


if __name__ == '__main__':

    main(argv=None)

