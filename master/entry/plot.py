import os
import argparse
import subprocess
import sys
import torch

from master.train.checkpoints import read_file_from_ini
from master.validate.plotting_new import plot_comprehensive_pt, plot_data_pt, plot_data_multi_band
from master.entry.test import run_test
from master.configs.config_utils import load_config_file
from master.entry.train import run_train

def run_plot(run_dir, test_on_separate_data=False, variant=None, use_train_set=False, diff=False, plot_dem_vis=False, plot_data_vis=False):
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
        if plot_dem_vis:
            
            env = os.environ.copy()
            cmd = [
                sys.executable,
                "master/validate/plot_lro_sampling.py",
                run_dir,
                "val",
                "train",
                "test"
            ]
            subprocess.run(cmd, env=env)
        elif plot_data_vis:
            # Run validation plotting commands
            print("Plotting data visualizations...")
            env = os.environ.copy()
            subprocess.run(["python", "master/validate/plotting_new.py", args.run_dir, "--plot_data_vis"], env=env, check=True)
            subprocess.run(["python", "master/validate/plotting_new.py", args.run_dir, "--use_train_set", "--plot_data_vis"], env=env, check=True)
        else:
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
    args.add_argument('--plot_dem_vis', action='store_true', default=False,
                      help="Plot DEM visualizations.")
    args.add_argument('--plot_data_vis', action='store_true', default=False,
                      help="Plot data visualizations.")
    args = args.parse_args()
    
    # Support both positional and flag-based arguments
    run_dir = args.run_dir_flag or args.run_dir

    run_plot(run_dir, test_on_separate_data=args.test_on_separate_data, variant=args.variant, use_train_set=args.use_train_set, diff=args.diff, plot_dem_vis=args.plot_dem_vis, plot_data_vis=args.plot_data_vis)
    
    



