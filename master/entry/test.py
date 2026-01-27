import subprocess
import os
import torch
import argparse

def run_test(run_dir, test_on_separate_data=False):
    if torch.cuda.is_available():
        n_proc_per_node = torch.cuda.device_count()
        print(f"Starting training with {n_proc_per_node} GPUs...")
        cmd = [
            "torchrun",
            f"--nproc_per_node={n_proc_per_node}",
            "--standalone",
            "master/runners/test_runner.py",
            run_dir
        ]
        if test_on_separate_data:
            cmd.append("--test_on_separate_data")
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "2"
        subprocess.run(cmd, env=env, check=True)
    else:
        print("No CUDA available - starting training with python call...")
        import sys
        env = os.environ.copy()
        cmd = [
            sys.executable,
            "master/runners/test_runner.py",
            run_dir
        ]
        if test_on_separate_data:
            cmd.append("--test_on_separate_data")
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
    args = argparse.ArgumentParser(description="Test a UNet model.")
    args.add_argument('run_dir', nargs="?" , type=str,
                      help='Directory of the trained model run containing stats and checkpoints.')
    args.add_argument('--run_dir', type=str, dest='run_dir_flag', required=False,
                      help='Directory of the trained model run containing stats and checkpoints.')
    args.add_argument('--test_on_separate_data', action='store_true',
                      help="Indicates testing on separate data.")
    args = args.parse_args()

    # Support both positional and flag-based arguments
    run_dir = args.run_dir_flag or args.run_dir

    run_test(run_dir, test_on_separate_data=args.test_on_separate_data)