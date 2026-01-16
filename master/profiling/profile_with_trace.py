"""
Standalone profiler that ACTUALLY saves trace files
Run with: torchrun --nproc_per_node=1 --standalone master/scripts/profile_with_trace.py
"""
import torch
import torch.profiler
import os
import sys
import argparse
from pathlib import Path

# CRITICAL: Set spawn before any CUDA usage
torch.multiprocessing.set_start_method('spawn', force=True)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from master.train.trainer_new import load_train_objs, prepare_dataloader, ddp_setup
from master.train.train_utils import normalize_inputs
from master.configs.config_utils import load_config_file
from master.train.checkpoints import read_file_from_ini
from torch.distributed import destroy_process_group

def profile_one_epoch(run_dir, batch_size=64, workers=16, prefetch=16, omp_threads=4):
    """Profile exactly one epoch with full trace"""
    
    # Initialize distributed training (required even for single GPU)
    ddp_setup()
    
    # Set environment
    os.environ["OMP_NUM_THREADS"] = str(omp_threads)
    
    # Load config
    run_path = os.path.join("./runs", run_dir)
    config_path = os.path.join(run_path, 'stats', 'config.ini')
    config = load_config_file(config_path)
    
    config.update({
        'BATCH_SIZE': batch_size,
        'NUM_WORKERS_DATALOADER': workers,
        'PREFETCH_FACTOR': prefetch,
        'EPOCHS': 1,
    })
    
    print(f"\nProfiling configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Workers: {workers}")
    print(f"  Prefetch: {prefetch}")
    print(f"  OMP threads: {omp_threads}")
    
    # Load mean/std
    mean_std_path = os.path.join(run_path, 'stats', 'input_stats.ini')
    input_stats = read_file_from_ini(mean_std_path)
    train_mean = torch.tensor([float(input_stats['MEAN'][i]) for i in range(len(input_stats['MEAN']))])
    train_std = torch.tensor([float(input_stats['STD'][i]) for i in range(len(input_stats['STD']))])
    
    # Setup
    device = torch.device(f'cuda:{os.environ["LOCAL_RANK"]}')
    
    train_set, val_set, test_set, model, optimizer = load_train_objs(config, run_path)
    model = model.to(device)
    
    # Use prepare_dataloader (with DistributedSampler)
    train_loader = prepare_dataloader(
        train_set,
        config['BATCH_SIZE'],
        config['NUM_WORKERS_DATALOADER'],
        config['PREFETCH_FACTOR']
    )
    
    # Create profiler output directory
    profile_dir = f"./runs/{run_dir}/profiler_traces"
    os.makedirs(profile_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"PROFILING ONE EPOCH WITH TORCH PROFILER")
    print(f"{'='*70}")
    print(f"Output directory: {profile_dir}\n")
    
    # Run with profiler
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=0,
            warmup=1,
            active=6,
            repeat=1
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        
        # Train one epoch
        model.train()
        
        for batch_idx, (images, reflectance_maps, targets, metas) in enumerate(train_loader):
            if batch_idx >= 7:  # warmup(1) + active(6)
                break
                
            # Move to device
            images = images.to(device)
            reflectance_maps = reflectance_maps.to(device)
            targets = targets.to(device)
            metas = metas.to(device)
            
            # Normalize
            images = normalize_inputs(images, train_mean, train_std)
            
            # Forward pass
            outputs = model(images, metas, target_size=targets.shape[-2:])
            
            # Loss
            from master.models.losses import calculate_total_loss
            loss = calculate_total_loss(
                outputs, targets, reflectance_maps, metas,
                device=device,
                camera_params=config["CAMERA_PARAMS"],
                hapke_params=config["HAPKE_KWARGS"],
                w_grad=config["W_GRAD"],
                w_refl=config["W_REFL"],
                w_mse=config["W_MSE"]
            )
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.get("GRAD_CLIP", 1.0))
            optimizer.step()
            
            # Step profiler
            prof.step()
            
            print(f"  Batch {batch_idx}/7 profiled | Loss: {loss.item():.4f}")
    
    print(f"\n{'='*70}")
    print(f"✓ PROFILING COMPLETE")
    print(f"{'='*70}")
    print(f"\nTrace files saved to: {profile_dir}")
    
    # List saved files
    import glob
    trace_files = glob.glob(f"{profile_dir}/*.pt.trace.json*")
    if trace_files:
        print(f"\nGenerated {len(trace_files)} trace file(s):")
        for f in trace_files:
            print(f"  - {os.path.basename(f)}")
    
    print(f"\nTo view:")
    print(f"  tensorboard --logdir={profile_dir}")
    print(f"  Then open: http://localhost:6006/#pytorch_profiler")
    print(f"\n{'='*70}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', default='lro_test')
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--prefetch', type=int, default=16)
    parser.add_argument('--omp', type=int, default=4)
    
    args = parser.parse_args()
    
    profile_one_epoch(args.run_dir, args.batch, args.workers, args.prefetch, args.omp)