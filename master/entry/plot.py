import os
import argparse
import subprocess
import torch

from master.train.checkpoints import read_file_from_ini
from master.validate.plotting_new import plot_comprehensive_pt, plot_data_pt
from master.entry.test import run_test


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Plot comprehensive test set predictions for a trained UNet model.")
    args.add_argument('run_dir', nargs="?" , type=str,
                      help='Directory of the trained model run containing stats and checkpoints.')
    args.add_argument('--run_dir', type=str, dest='run_dir_flag', required=False,
                      help='Directory of the trained model run containing stats and checkpoints.')
    args.add_argument('--variant', type=str, default='first',
                      help="Variant for selecting test sets: 'first' or 'random'.")
    args.add_argument('--use_train_set', action='store_true',
                      help="Use the training set for plotting instead of the test set.")
    args.add_argument('--test_on_separate_data', action='store_true',
                      help="Indicates testing on separate data.")
    args = args.parse_args()

    # Support both positional and flag-based arguments
    run_dir = args.run_dir_flag or args.run_dir

    print("Plotting test set predictions...")

    if not os.path.exists(os.path.join("runs", run_dir, 'checkpoints', 'snapshot.pt')):
        #no training exists, only plot data
        print("No training snapshot found, plotting data only...")
        plot_data_pt(run_dir=os.path.join("runs", run_dir),
                        n_sets=5,
                        variant=args.variant,  # 'first' or 'random'
                        same_scale=False,  # False, 'row', or 'all'
                        save_fig=True,
                        return_fig=False,
                        use_train_set = args.use_train_set)
    else:
        # Ensure test results exist and are valid
        if not args.test_on_separate_data:
            test_results_path = os.path.join("runs", run_dir, 'stats', 'test_results.ini')
        else:
            test_results_path = os.path.join("runs", run_dir, 'stats', 'alt_test_results.ini')

        if not os.path.exists(test_results_path):
            print("No test results found, running testing...")
            run_test(run_dir, test_on_separate_data=args.test_on_separate_data)

        filename = "comprehensive"
        if not args.use_train_set:
            filename += "_testset"
        if args.use_train_set:
            filename += "_trainset"
        if args.test_on_separate_data:
            filename += "_alt"
        if args.variant == 'first':
            filename += "_first"
        if args.variant == 'random':
            filename += "_random"
            
        filename += ".pdf"

        print("Training snapshot and test results found, plotting predictions...")
        plot_comprehensive_pt(run_dir=run_dir,
                            n_test_sets=5,
                            variant=args.variant,  # 'first' or 'random'
                            same_scale=False,  # False, 'row', or 'all'
                            figsize=(15, 10),
                            save_fig=True,
                            return_fig=False,
                            use_train_set = args.use_train_set,
                            filename=filename,
                            test_on_separate_data = args.test_on_separate_data)