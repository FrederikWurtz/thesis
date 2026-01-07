#!/bin/bash
#SBATCH --job-name=fluid_train          # Job name
#SBATCH --output=fluid_train.out        # Standard output file
#SBATCH --error=fluid_train.err         # Standard error file
#SBATCH --time=04:00:00                 # Max runtime (hh:mm:ss)
#SBATCH --cpus-per-task=16              # Number of CPU cores
#SBATCH --mem=64G                       # Memory requested
#SBATCH --gres=gpu:2                    # Request 2 GPUs
#SBATCH --partition=gpu                 # GPU partition/queue (depends on your cluster)

# Load modules if your cluster uses them (example: CUDA, Python)
# module load python/3.12
# module load cuda/12.1

# Activate your virtual environment
source /work/FrederikWürtzSørensen#7865/venv/bin/activate

# Move into your project directory
cd /work/FrederikWürtzSørensen#7865

# Run your training script
python -m master.scripts.fluid_train