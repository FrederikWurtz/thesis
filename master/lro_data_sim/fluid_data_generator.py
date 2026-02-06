
import os
from master.lro_data_sim.fluid_data_generator import generate_and_save_data_pooled_multi_gpu
from master.configs.config_utils import load_config_file


def main(run_dir: str, epoch: int = 0):
    config = load_config_file(run_dir)
    train_path = os.path.join(run_dir, 'train_temp')
    print(f"Generating fluid data for run '{config['RUN_NAME']}' at epoch {epoch}...")
    generate_and_save_data_pooled_multi_gpu(config, images_dir=train_path, n_dems=config["FLUID_TRAIN_DEMS"], use_cpu_fallback=True)
    
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate fluid data for a given run directory.")
    parser.add_argument('run_directory', type=str, help='Path to the run directory')
    parser.add_argument('epoch', type=int, default=0, help='Epoch number for data generation')
    args = parser.parse_args()
    
    main(args.run_directory, args.epoch)
    
