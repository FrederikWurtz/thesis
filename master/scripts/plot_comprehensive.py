import os
import matplotlib.pyplot as plt
import argparse

from master.validate.plotting_new import plot_comprehensive

def _parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--run_dir', type=str, default=None)
    return p.parse_args(argv)

if __name__ == '__main__':
    args = _parse_args()
    if args.run_dir is None:
        raise ValueError("Please provide --run_dir argument pointing to the run directory.")
    
    run_dir = os.path.join('runs', args.run_dir)  # Assuming runs are stored in 'runs/' directory    
    # Quick demo of plotting function
    plot_comprehensive(
        run_dir=run_dir,
        return_fig=False,
    )