#!/bin/bash

# Help reduce thread overload?
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1


export CUDA_VISIBLE_DEVICES=0,1

# Good defaults while debugging
#export NCCL_DEBUG=WARN              # reduce log noise once stable (use INFO while debugging)
export NCCL_DEBUG=INFO
export NCCL_BLOCKING_WAI

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Encourage fast intra-node P2P and NVSwitch where available
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export NCCL_NVLS_ENABLE=1           # enables NVLink Switch (ignored if not present)

# Optional: pin the interface (already eth0)
export NCCL_SOCKET_IFNAME=eth0


# Change to right directory - be sure to add this when starting application
# cd FrederikWürtzSørensen#7865

# Activate your virtual environment
source venv/bin/activate

# Run your training script
torchrun --nproc_per_node=2 -m master.scripts.multi_gpu_entry --new_run --skip_data_gen
# python -m master.scripts.fluid_train --new_run