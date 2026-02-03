import subprocess
import os
import torch
import argparse
import sys
import signal
import torch.distributed

def handle_sigint(signum, frame):
    print("Interrupt received. Destroying process group...")
    raise KeyboardInterrupt

def kill_orphaned_processes():
    subprocess.run(["pkill", "-f", "train_runner_multiGPU.py"])
    subprocess.run(["pkill", "-f", "torchrun"])

def run_train(run_dir, new_run=False):
    if torch.cuda.is_available():

        n_proc_per_node = torch.cuda.device_count()
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

        signal.signal(signal.SIGINT, handle_sigint)
        signal.signal(signal.SIGTERM, handle_sigint)

        try:
            subprocess.run(cmd, env=env, check=True)
        except KeyboardInterrupt:
            print("Training interrupted by user.")
            kill_orphaned_processes()
            sys.exit(0)
            print("Cleaned up after interruption.")
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


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Train a UNet model.")
    args.add_argument('run_dir', nargs="?" , type=str,
                      help='Directory of the trained model run containing stats and checkpoints.')
    args.add_argument('--run_dir', type=str, dest='run_dir_flag', required=False,
                      help='Directory of the trained model run containing stats and checkpoints.')
    args.add_argument('--new_run', action='store_true',
                      help="Indicates a new run.")
    args = args.parse_args()

    # Support both positional and flag-based arguments
    run_dir = args.run_dir_flag or args.run_dir

    run_train(run_dir, new_run=args.new_run)
