"""
Simple orchestration script to (optionally) generate synthetic training data and then
start training using the project's scripts.

Usage examples from the NN_v6 folder:
  # Generate data then train for 10 epochs
  python main.py --generate --n_dems 2 --n_sets_per_dem 5 --workers 4 --train --n_epochs 10

  # Only train using existing data_dir
  python main.py --no-generate --data_dir 2_dems --train --n_epochs 20

This script imports the project's `create_training_data` and `train_unet_mps` modules
and invokes their `main()` entrypoints. It constructs appropriate argv lists so the
internal argument parsers of those modules receive the expected CLI arguments.
"""

import sys
import os
import argparse


def run_create_data(args):
	# Build argv for create_training_data
	create_argv = ["create_training_data.py"]
	create_argv += ["--n_dems", str(args.n_dems)]
	create_argv += ["--n_sets_per_dem", str(args.n_sets_per_dem)]
	create_argv += ["--n_images_per_set", str(args.n_images_per_set)]
	create_argv += ["--dem_size", str(args.dem_size)]
	create_argv += ["--image_width", str(args.image_width), "--image_height", str(args.image_height)]
	create_argv += ["--focal_length", str(args.focal_length)]
	create_argv += ["--workers", str(args.workers)]
	create_argv += ["--batch_size", str(args.batch_size)]
	create_argv += ["--n_hills", str(args.n_hills)]
	create_argv += ["--n_ridges", str(args.n_ridges)]
	create_argv += ["--n_craters", str(args.n_craters)]

	if args.flat_bottom:
		create_argv.append("--flat_bottom")
	if args.do_not_clear_folder:
		create_argv.append("--do_not_clear_folder")
	if args.output_dir is not None:
		create_argv += ["--output_dir", args.output_dir]

	# Call the package generator directly so we can remove the legacy top-level script.
	from master.data_sim.generator import generate_dem_and_set

	n_samples = int(args.n_dems) * int(args.n_sets_per_dem)
	H_img = int(args.image_height)
	W_img = int(args.image_width)
	H_dem = int(args.dem_size)
	W_dem = int(args.dem_size)

	print("\n>>> Running data generation with parameters:", f"n_samples={n_samples}, out={args.output_dir}")
	generate_dem_and_set(args.output_dir or f"{args.n_dems}_dems", n_samples, H_img, W_img, H_dem, W_dem, seeds=None, n_workers=args.workers)


def run_training(args):
	# Use the package training runner instead of the legacy top-level script.
	from master.train.runner import run_training

	data_dir = args.output_dir if args.output_dir is not None else args.data_dir
	out_dir = args.output_dir or data_dir

	print(f"\n>>> Running training: data_dir={data_dir}, out_dir={out_dir}, epochs={args.n_epochs}")
	run_training(data_dir=data_dir, out_dir=out_dir, epochs=args.n_epochs, batch_size=args.train_batch_size, lr=(args.lr if args.lr is not None else 1e-3))


def parse_args():
	parser = argparse.ArgumentParser(description='Orchestrate data generation and training for NN_v6')
	parser.add_argument('--generate', dest='generate', action='store_true', default=True,
						help='Generate synthetic data before training (default: True)')
	parser.add_argument('--no-generate', dest='generate', action='store_false', help='Skip data generation')

	# Data generation options (subset)
	parser.add_argument('--n_dems', type=int, default=1, help='Number of DEMs to generate')
	parser.add_argument('--n_sets_per_dem', type=int, default=1, help='Number of sets per DEM')
	parser.add_argument('--n_images_per_set', type=int, default=5, help='Images per set')
	parser.add_argument('--dem_size', type=int, default=512, help='DEM size (pixels)')
	parser.add_argument('--image_width', type=int, default=128, help='Rendered image width')
	parser.add_argument('--image_height', type=int, default=128, help='Rendered image height')
	parser.add_argument('--focal_length', type=float, default=800, help='Camera focal length')
	parser.add_argument('--n_craters', type=int, default=0, help='Number of craters per DEM (default: 0)')
	parser.add_argument('--n_ridges', type=int, default=6, help='Number of ridges per DEM (default: 6)')
	parser.add_argument('--n_hills', type=int, default=8, help='Number of Gaussian hills per DEM (default: 8)')
	parser.add_argument('--workers', type=int, default=4, help='Worker processes for data generation')
	parser.add_argument('--batch_size', type=int, default=10, help='Batch size for data generation')
	parser.add_argument('--flat_bottom', action='store_true', help='Create flat DEMs')
	parser.add_argument('--do_not_clear_folder', action='store_true', help='Append to existing images folder')
	parser.add_argument('--output_dir', type=str, default=None, help='Output directory for generated data')

	# Training options (subset)
	parser.add_argument('--train', dest='train', action='store_true', default=True, help='Run training after generation')
	parser.add_argument('--no-train', dest='train', action='store_false', help='Do not run training')
	parser.add_argument('--n_epochs', type=int, default=10, help='Number of epochs to train')
	parser.add_argument('--train_batch_size', type=int, default=32, help='Training batch size')
	parser.add_argument('--train_num_workers', type=int, default=8, help='DataLoader workers for training')
	parser.add_argument('--new_training', action='store_true', help='Start new training (delete/ignore previous checkpoints)')
	parser.add_argument('--lr', type=float, default=None, help='Learning rate override for trainer')
	# fallback data_dir if user only wants to train
	parser.add_argument('--data_dir', type=str, default='1_dems', help='Data directory for training if not generating')

	return parser.parse_args()


def main():
	args = parse_args()

	# If output_dir not specified, choose a sensible default matching previous scripts
	if args.output_dir is None:
		args.output_dir = f"{args.n_dems}_dems"

	# Generate data if requested
	if args.generate:
		run_create_data(args)
	else:
		print("\n>>> Skipping data generation as requested")

	# Run training if requested
	if args.train:
		run_training(args)
	else:
		print("\n>>> Skipping training as requested")


if __name__ == '__main__':
	main()

