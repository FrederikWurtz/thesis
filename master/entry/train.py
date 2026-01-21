import subprocess
import os
import torch
import argparse

def run_train(run_dir, new_run=False):
    n_proc_per_node = torch.cuda.device_count()
    print(f"Starting training with {n_proc_per_node} GPUs...")
    cmd = [
        "torchrun",
        f"--nproc_per_node={n_proc_per_node}",
        "--standalone",
        "master/train/train_main.py",
        run_dir
    ]
    if new_run:
        cmd.append("--new_run")
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "2"
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
