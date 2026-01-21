import os
import argparse
import subprocess
import torch

from master.train.checkpoints import read_file_from_ini
from master.validate.plotting_new import plot_comprehensive_pt, plot_data_pt

def run_test(run_dir):
    n_proc_per_node = torch.cuda.device_count()
    print(f"No test results found - running test with {n_proc_per_node} GPUs before plotting...")
    cmd = [
        "torchrun",
        f"--nproc_per_node={n_proc_per_node}",
        "--standalone",
        "master/entry/test.py",
        "--run_dir", run_dir
    ]
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "2"
    subprocess.run(cmd, env=env, check=True)



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
        test_results_path = os.path.join("runs", run_dir, 'stats', 'test_results.ini')
        if not os.path.exists(test_results_path):
            run_test(run_dir)
        else:
            test_results = read_file_from_ini(test_results_path)
            if test_results.get("TEST_LOSS") is None:
                    run_test(run_dir)

        print("Training snapshot and test results found, plotting predictions...")
        plot_comprehensive_pt(run_dir=run_dir,
                            n_test_sets=5,
                            variant=args.variant,  # 'first' or 'random'
                            same_scale=False,  # False, 'row', or 'all'
                            figsize=(15, 10),
                            save_fig=True,
                            return_fig=False,
                            use_train_set = args.use_train_set
                            )