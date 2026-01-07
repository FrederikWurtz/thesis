import argparse
from master.data_sim.generator import generate_and_save_data_pooled
from master.configs.config_utils import load_config_file, create_folder_structure
from master.train.runner import run_fluid_training
from master.train.trainer_core import evaluate_on_test_files
from master.train.checkpoints import read_file_from_ini, save_checkpoint, load_checkpoint, save_file_as_ini
from master.utils.interactivity_utils import install_quiet_signal_handlers
from master.utils.load_est_utils import total_cpu_seconds, compute_timing_info, print_timing_info
import os
import shutil
import psutil 
import time
import multiprocessing as mp

def _parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--new_run', action='store_true', default=False)
    p.add_argument('--skip_data_gen', action='store_true', default=False)
    p.add_argument('--run_dir', type=str, default=None)
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--num_workers', type=int, default=None)
    p.add_argument('--lr_patience', type=int, default=None)
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--batch_size', type=int, default=None)
    p.add_argument('--w_mse', type=float, default=None)
    p.add_argument('--w_grad', type=float, default=None)
    p.add_argument('--w_refl', type=float, default=None)
    p.add_argument('--fluid_train_dems', type=int, default=None)
    p.add_argument('--fluid_val_dems', type=int, default=None)
    p.add_argument('--fluid_test_dems', type=int, default=None)
    p.add_argument('--images_per_dem', type=int, default=None)
    return p.parse_args(argv)

def main(argv=None):
    args = _parse_args(argv)
    config = load_config_file() # load default config

    # Initialize CPU and GPU timing measurement
    _proc = psutil.Process(psutil.Process().pid)
    try:
        start_cpu_seconds = total_cpu_seconds(_proc)
    except Exception:
        _proc = None
        start_cpu_seconds = None
    t0 = time.time()

    # handle Ctrl-C gracefully
    # might also supress other tracebacks in threads, so set quiet_exceptions=False if full tracebacks are desired
    # install_quiet_signal_handlers(quiet_exceptions=False)
    
    # Determine run directory, and parse new/skip_data_gen flags
    run_dir = os.path.join(config["SUP_DIR"], config["RUN_DIR"])
    if os.path.exists(run_dir):
        # Existing run directory found
        if args.new_run is True:
            # User requested new run
            if not args.skip_data_gen:
                # Remove existing run directory entirely
                print(60*"=")
                print("New run flag set. Removing existing run directory...")
                try:
                    if os.path.isdir(run_dir):
                        shutil.rmtree(run_dir)
                        print(f"Removed existing run directory: {run_dir}")
                except Exception as e:
                    print(f"Failed to remove run_dir {run_dir}: {e}")
                print(60*"=")
                run_dir, val_dir, test_dir = create_folder_structure(config)
            elif args.skip_data_gen:
                # User requested to skip data generation, so keep existing val/test files but remove old training data
                print(60*"=")
                print("New run flag set but skipping data generation. Continuing with existing run directory, but removing old training data...")
                print(f"Using run directory: {run_dir}")
                print(60*"=")
                val_dir = os.path.join(run_dir, 'val')
                test_dir = os.path.join(run_dir, 'test')
                # Remove old training data if present
                checkpoint_dir = os.path.join(run_dir, 'checkpoints')
                stats_dir = os.path.join(run_dir, 'stats')
                if os.path.exists(stats_dir):
                    try:
                        shutil.rmtree(stats_dir)
                    except Exception as e:
                        print(f"Could not remove stats directory {stats_dir}: {e}")
                if os.path.exists(checkpoint_dir):
                    try:
                        shutil.rmtree(checkpoint_dir)
                    except Exception as e:
                        print(f"Could not remove checkpoint directory {checkpoint_dir}: {e}")

                # also 

        elif args.new_run is False:
            # Continuing existing run
            print(60*"=")
            print("Existing directory found. Continuing training for existing run...")
            print(f"Using run directory: {run_dir}")
            print(60*"=")

            stats_dir = os.path.join(run_dir, 'stats')
            os.makedirs(stats_dir, exist_ok=True)
            # reload config from existing run directory
            config_path = os.path.join(stats_dir, 'config.ini')
            config = load_config_file(config_path) # reload config from existing run

            val_dir = os.path.join(run_dir, 'val')
            test_dir = os.path.join(run_dir, 'test')
    else:
        # No existing run directory found; create new run
        print(60*"=")
        print(f"No existing run directory found.")
        print(f"Creating new run at: {run_dir}")
        print(60*"=")
        run_dir, val_dir, test_dir = create_folder_structure(config)
        args.new_run = True


    # apply CLI overrides to config by uppercasing arg names (only if key exists in config)
    allowed_cli_only = {"NEW_RUN", "SKIP_DATA_GEN"}
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


    mp.set_start_method("spawn", force=True)  # for dataloader workers and data generation - ensures compatibility across platforms
    
    # If run is found to be new, generate data
    if args.new_run:
        if not args.skip_data_gen: # unless user requests to skip data generation
            print("\n=== Generating New Dataset ===")
            # create validation files
            generate_and_save_data_pooled(config, images_dir=val_dir, n_dems=config["FLUID_VAL_DEMS"])
            # create test files
            generate_and_save_data_pooled(config, images_dir=test_dir, n_dems=config["FLUID_TEST_DEMS"])
            print("Dataset generation complete.\n")
        elif args.skip_data_gen:
            pass  # skip data generation as per user request
    
    config_path = os.path.join(run_dir, 'stats', 'config.ini')
    save_file_as_ini(config, config_path) # save final config to run directory

    returned_values = run_fluid_training(config=config, 
                                         run_dir=run_dir, 
                                         val_dir=val_dir, 
                                         test_dir=test_dir, 
                                         new_training=args.new_run)
    
    best_epoch, start_epoch, end_epoch, model, test_loader, device, train_mean, train_std, camera_params, hapke_params, use_amp, non_blocking = returned_values
    print(f"Training from {start_epoch} to {end_epoch} completed. Best validation at epoch {best_epoch}.")


    print("Performing evaluation on the test files using best checkpoint.")
    # Evaluate on test set
    test_loss, test_ame = evaluate_on_test_files(model = model, 
                                                test_loader = test_loader, 
                                                device = device,
                                                train_mean = train_mean,
                                                train_std = train_std,
                                                camera_params = camera_params,
                                                hapke_params = hapke_params,
                                                use_amp = use_amp,
                                                w_mse = config["W_MSE"],
                                                w_grad = config["W_GRAD"],
                                                w_refl = config["W_REFL"],
                                                non_blocking = non_blocking)
    print(f"Test loss: {test_loss:.4f}, Test AME: {test_ame:.4f}")
    run_stats_dict = {
        'best_epoch': best_epoch,
        'test_loss': test_loss,
        'test_ame': test_ame
    }
    stats_dir = os.path.join(run_dir, 'stats')
    os.makedirs(stats_dir, exist_ok=True)
    run_stats_path = os.path.join(stats_dir, 'run_stats.ini')
    save_file_as_ini(run_stats_dict, run_stats_path)
    print(f"Saved run statistics to {run_stats_path}")


    # Final timing info
    total_time = time.time() - t0
    timing_info, cpu_delta = compute_timing_info(start_cpu_seconds, total_time, device=device, proc=_proc, epochs = end_epoch - start_epoch + 1)
    timing_path = os.path.join(run_dir, 'stats', 'timing_info.ini')
    print_timing_info(timing_info)

    #if this is not a new training, get earlier timing info and add to current
    if not args.new_run:
        if os.path.exists(timing_path):
            previous_timing_info = read_file_from_ini(timing_path)
            # accumulate times
            for key in ['wall_seconds', 'cpu_seconds', 'gpu_seconds']:
                if key in previous_timing_info and key in timing_info:
                    timing_info[key] += previous_timing_info[key]
        else:
            print(f"Warning: timing info file {timing_path} not found for continuation run.")
                    
    save_file_as_ini(timing_info, timing_path)
    print(f"Saved timing information to {timing_path}")
    print_timing_info(timing_info)


if __name__ == '__main__':
    main()

