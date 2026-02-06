import os
import argparse
import subprocess
import torch

from master.train.checkpoints import read_file_from_ini
from master.validate.plotting_new import plot_comprehensive_pt, plot_data_pt, plot_data_multi_band
from master.entry.test import run_test
from master.configs.config_utils import load_config_file
from master.entry.train import run_train

def run_plot(run_dir, test_on_separate_data=False, variant=None, use_train_set=False, diff=False):
        cmd = [
            "python",
            "master/validate/plotting_new.py",
            run_dir]
        if test_on_separate_data:
            cmd.append("--test_on_separate_data")
        if variant:
            cmd.extend(["--variant", variant])
        if use_train_set:
            cmd.append("--use_train_set")
        if diff:
            cmd.append("--diff")
        env = os.environ.copy()
        subprocess.run(cmd, env=env, check=True)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Plot comprehensive test set predictions for a trained UNet model.")
    args.add_argument('run_dir', nargs="?" , type=str,
                      help='Directory of the trained model run containing stats and checkpoints.')
    args.add_argument('--run_dir', type=str, dest='run_dir_flag', required=False,
                      help='Directory of the trained model run containing stats and checkpoints.')
    args.add_argument('--variant', type=str, default=False,
                      help="Variant for selecting test sets: 'first' or 'random'.")
    args.add_argument('--use_train_set', action='store_true', default=False,
                      help="Use the training set for plotting instead of the test set.")
    args.add_argument('--test_on_separate_data', action='store_true', default=False,
                      help="Indicates testing on separate data.")
    args.add_argument('--diff', action='store_true', default=False,
                      help="Include differences in the plots.")
    args = args.parse_args()
    
    # Support both positional and flag-based arguments
    run_dir = args.run_dir_flag or args.run_dir

    run_plot(run_dir, test_on_separate_data=args.test_on_separate_data, variant=args.variant, use_train_set=args.use_train_set, diff=args.diff)
    
    



