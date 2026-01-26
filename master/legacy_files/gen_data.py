import argparse
from master.data_sim.generator import generate_and_save_data_pooled
from master.configs.config_utils import load_config_file, create_folder_structure
import os
import sys
import subprocess

def _parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--run_dir', type=str, default=None)
    p.add_argument('--n_dems', action='store_true', default=False)
    p.add_argument('--workers', action='store_true', default=False)
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    config = load_config_file()

    if args.run_dir is not None:
        config["RUN_DIR"] = args.run_dir
    if args.n_dems is not False:
        config["IMAGES_PER_DEM"] = args.n_dems
    if args.workers is not False:
        config["N_WORKERS"] = args.workers

    run_dir, val_dir, test_dir = create_folder_structure(config)

    # create validation files
    generate_and_save_data_pooled(config, images_dir=val_dir, n_dems=config["FLUID_VAL_DEMS"], )
    # create test files
    generate_and_save_data_pooled(config, images_dir=test_dir, n_dems=config["FLUID_TEST_DEMS"], )



if __name__ == '__main__':
        # Check if we're running on macOS
    if sys.platform == 'darwin':
        # Check if already running under caffeinate
        if 'CAFFEINATED' not in os.environ:
            print("=" * 60)
            print("Starting caffeinate to prevent system sleep during training")
            print("This ensures full performance even if the screen turns off")
            print("=" * 60)
            
            # Re-run this script under caffeinate
            # -d: Prevent display from sleeping
            # -i: Prevent system from idle sleeping
            # -m: Prevent disk from idle sleeping
            env = os.environ.copy()
            env['CAFFEINATED'] = '1'  # Mark that we're now caffeinated
            
            try:
                result = subprocess.run(
                    ['caffeinate', '-dims', sys.executable] + sys.argv,
                    env=env
                )
                sys.exit(result.returncode)
            except KeyboardInterrupt:
                print("\n\nTraining interrupted by user")
                sys.exit(130)
            except Exception as e:
                print(f"\n\nError starting caffeinate: {e}")
                print("Continuing without caffeinate...")
    main()
