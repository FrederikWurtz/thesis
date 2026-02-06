import subprocess
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

import torch
import argparse
import sys
import signal
import numpy as np
import time
from master.train.checkpoints import save_file_as_ini, read_file_from_ini
from master.data_sim.generator import generate_and_save_data_pooled_multi_gpu
from master.configs.config_utils import load_config_file

import gc


def handle_sigint(signum, frame):
    print("Interrupt received. Destroying process group...")
    raise KeyboardInterrupt

def kill_orphaned_processes():
    subprocess.run(["pkill", "-f", "train_runner_multiGPU.py"])
    subprocess.run(["pkill", "-f", "torchrun"])
        # Clean up semaphores
    subprocess.run(["bash", "-c", "rm -f /dev/shm/sem.torch_*"], stderr=subprocess.DEVNULL)
    
def cleanup():
    print("Cleaning up resources...")
    gc.collect()
    torch.cuda.empty_cache()  # Clear GPU cache
    # Close any open multiprocessing pools
    if hasattr(torch.multiprocessing, 'get_context'):
        torch.multiprocessing.set_sharing_strategy('file_system')
    kill_orphaned_processes()
    print("Cleanup done.")

    

def run_train(run_dir, new_run=False):
    if torch.cuda.is_available():

        n_proc_per_node = torch.cuda.device_count()
        # Limit to 3 GPUs to avoid race condition issues with 4th GPU
        n_proc_per_node = min(n_proc_per_node, 3)
        print(f"Starting training with {n_proc_per_node} GPUs...")
        cmd = [
            "torchrun",
            f"--nproc_per_node={n_proc_per_node}",
            "--standalone",
            "master/runners/train_runner_multiGPU.py",
            run_dir
        ]
        if new_run:
            cmd.append("--new_run")
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "2"
        env["GDAL_NUM_THREADS"] = "2"
        env["CUDA_VISIBLE_DEVICES"] = "0,1,2"

        signal.signal(signal.SIGINT, handle_sigint)
        signal.signal(signal.SIGTERM, handle_sigint)

        try:
            subprocess.run(cmd, env=env, check=True)
        except KeyboardInterrupt:
            print("Training interrupted by user")
            cleanup()
            sys.exit(0)
        except Exception as e:
            print(f"Error during training: {e}")

    else:
        print("No CUDA available - starting training with python call...")
        env = os.environ.copy()
        cmd = [
            sys.executable,
            "master/runners/train_runner_singleGPU.py",
            run_dir
        ]
        if new_run:
            cmd.append("--new_run")
        # Only wrap the training command in caffeinate if on Darwin and not already caffeinated
        if sys.platform == 'darwin' and 'CAFFEINATED' not in env:
            print("=" * 60)
            print("Starting caffeinate to prevent system sleep during training")
            print("This ensures full performance even if the screen turns off")
            print("=" * 60)
            env['CAFFEINATED'] = '1'  # Mark that we're now caffeinated
            cmd = ['caffeinate', '-dims'] + cmd
        subprocess.run(cmd, env=env, check=True)

def run_train_semifluid(config, run_dir, new_run=False, initial_data_gen=False):    
    TOTAL_EPOCHS = config["TOTAL_EPOCHS"]
    NEW_FLUID_DATA_EVERY = config["NEW_FLUID_DATA_EVERY"]
    
    # generate list of epochs where new fluid data should be generated
    fluid_data_epochs = list(range(0, TOTAL_EPOCHS+NEW_FLUID_DATA_EVERY, NEW_FLUID_DATA_EVERY))[1:] # exclude epoch 0
    # ensure last epoch is larger than TOTAL_EPOCHS, otherwise loop will never end
    
    training_times_per_epoch = []
    generation_times = []
    total_time_elapsed = 0.0
    epochs_completed = 0
    
    # check if a run exists already
    if os.path.exists(os.path.join('runs' , run_dir, 'stats', 'timing_stats.ini')) and not new_run:
        print("Existing run detected and new_run flag is False. Resuming semi-fluid training...")
        val_losses_path = os.path.join('runs' , run_dir, 'checkpoints', 'val_losses.csv')
        val_data = np.genfromtxt(val_losses_path, delimiter=',', skip_header=1)
        epochs_completed = val_data[:,0].max().astype(int)
        print(f"Resuming from epoch {epochs_completed}...")
        
        # remove epochs already completed from fluid_data_epochs
        for i, epoch in enumerate(fluid_data_epochs):
            if epoch <= epochs_completed:
                continue
            else:
                fluid_data_epochs = fluid_data_epochs[i:]
                break
        print("Remaining fluid data epochs after resuming:", fluid_data_epochs)
        new_run = False
        timing_stats = read_file_from_ini(os.path.join('runs' , run_dir, 'stats', 'timing_stats.ini'))
        total_time_elapsed = timing_stats["total_time_elapsed"]
        training_times_per_epoch = timing_stats.get("training_times_per_epoch", [])
        generation_times = timing_stats.get("generation_times", [])
        
    elif os.path.exists(os.path.join('runs' , run_dir, 'stats', 'timing_stats.ini')) and new_run:
        print("New run flag detected and existing run found. Removing existing run data...")
        import shutil
        shutil.rmtree(os.path.join('runs' , run_dir))
        print("Starting new semi-fluid training run...")
        new_run = True
    else:
        print("No existing run found. Starting new semi-fluid training run...")
    
    if initial_data_gen:
        print("Initial data generation for semi-fluid training...")
        seed = config["BASE_SEED"] + epochs_completed  # use base seed for initial data generation
        torch.manual_seed(seed) 
        np.random.seed(seed % (2**32 - 1))
        train_path = os.path.join('runs', run_dir, 'train_temp') 
        os.makedirs(train_path, exist_ok=True)
        generate_and_save_data_pooled_multi_gpu(config, images_dir=train_path, n_dems=config["FLUID_TRAIN_DEMS"], n_gpus = min(torch.cuda.device_count(), 3))
        print("Initial fluid training data generated.")
    
    #override epoch parameter for semi-fluid training
    for epochs in fluid_data_epochs:
        
        config["EPOCHS"] = epochs + 1 # add 1 to ensure we train for the full number of epochs in this phase, since the loop will break when epochs is reached, not exceeded
        
        # save updated config to run directory
        config_path = os.path.join('runs' , run_dir, 'stats', 'config.ini')
        save_file_as_ini(config, config_path) 
        print(f"Starting training from {epochs_completed} to {epochs} epochs with SemiFluid dataset...")
        time_start_train = time.time()
        
        run_train(run_dir, new_run=new_run)
        epochs_completed = epochs  # update completed epochs
        
        time_end_train = time.time()
        elapsed = time_end_train - time_start_train
        training_times_per_epoch.append(elapsed/NEW_FLUID_DATA_EVERY)
        
        new_run = False  # only the first call should be a new run
        print(f"Completed training from {epochs_completed} to {epochs} epochs."
              f" Regenerating fluid data for next phase...")
        
        seed = config["BASE_SEED"] + epochs  # change seed for new data generation
        torch.manual_seed(seed) 
        np.random.seed(seed % (2**32 - 1))
        train_path = os.path.join('runs', run_dir, 'train_temp') 
        # remove entire train_temp directory before regenerating data
        if os.path.exists(train_path):
            import shutil
            shutil.rmtree(train_path)
            print(f"Cleared existing training data at {train_path}")
        # make directory again
        os.makedirs(train_path, exist_ok=True)
        # generate new fluid training data
        time_start_generation = time.time()
        generate_and_save_data_pooled_multi_gpu(config, images_dir=train_path, n_dems=config["FLUID_TRAIN_DEMS"], n_gpus = min(torch.cuda.device_count(), 3))
        time_end_generation = time.time()
        elapsed_generation = time_end_generation - time_start_generation
        generation_times.append(elapsed_generation)
        print(f"Regenerated fluid training data for next phase.")
        
        total_time_elapsed += elapsed + elapsed_generation
        
        # Save timing statistics
        timing_stats = {
            'total_time_elapsed': total_time_elapsed,
            'generation_times': generation_times,
            'training_times_per_epoch': training_times_per_epoch,
            'average_training_time_per_epoch': np.mean(training_times_per_epoch) if training_times_per_epoch else 0,
            'average_generation_time': np.mean(generation_times) if generation_times else 0,
            'len_fluid_data': config["FLUID_TRAIN_DEMS"],
            'num_gpus': min(torch.cuda.device_count(), 3),
            'epochs_completed': epochs
        }
        timing_path = os.path.join('runs', run_dir, 'stats', 'timing_stats.ini')
        save_file_as_ini(timing_stats, timing_path)
        
        # light cleanup after each phase to free up memory before next training phase
        gc.collect()
        torch.cuda.empty_cache()

        

if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Train a UNet model.")
    args.add_argument('run_dir', nargs="?" , type=str,
                      help='Directory of the trained model run containing stats and checkpoints.')
    args.add_argument('--run_dir', type=str, dest='run_dir_flag', required=False,
                      help='Directory of the trained model run containing stats and checkpoints.')
    args.add_argument('--new_run', action='store_true',
                      help="Indicates a new run.")
    args.add_argument('--initial_data_gen', action='store_true',
                      help="Indicates whether to start generating new data.")
    args = args.parse_args()

    # Support both positional and flag-based arguments
    run_dir = args.run_dir_flag or args.run_dir
    
    config = load_config_file(os.path.join('runs' , run_dir, 'stats', 'config.ini'))
    
    if config["USE_SEMIFLUID"]:
        for trial in range(3):  # prøv op til 3 gange
            try:
                run_train_semifluid(config, run_dir, new_run=args.new_run, initial_data_gen=args.initial_data_gen)
                break  # hvis det lykkes, bryd ud af loopen
            
            except KeyboardInterrupt:
                print("Training interrupted by user")
                cleanup()
                raise

            except Exception as e:
                print(f"Error in trial {trial + 1}: {e}")
                if trial == 2:  # sidste forsøg fejlede også
                    print("All attempts failed, stopping.")
                    cleanup()
                    raise  # løft fejlen videre, hvis du vil stoppe helt
                else:
                    print("Trying again...")

    else:
        run_train(run_dir, new_run=args.new_run)
        print("Training completed successfully.")
        cleanup()
