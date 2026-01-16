"""
Profiling script to analyze where training spends its time
Tracks CPU/GPU usage and identifies bottlenecks
"""
import json
import os
import sys
import time
import argparse
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.profiler import profile, record_function, ProfilerActivity

from master.train.trainer_core import FluidDEMDataset, DEMDataset
from master.train.trainer_new import Trainer, ddp_setup, load_train_objs, prepare_dataloader, is_main
from master.train.checkpoints import save_file_as_ini, read_file_from_ini
from master.configs.config_utils import load_config_file
from master.models.unet import UNet

torch.multiprocessing.set_start_method('spawn', force=True)


@contextmanager
def timer(name):
    """Context manager to time code blocks"""
    start = time.time()
    yield
    elapsed = time.time() - start
    print(f"[TIMER] {name}: {elapsed:.3f}s")


def profile_single_epoch(trainer, num_batches=10):
    """Profile a single training epoch with detailed timing, using actual Trainer logic"""
    trainer.model.train()
    
    # Timing accumulators
    times = {
        'data_loading': 0,
        'data_transfer': 0,
        'normalization': 0,
        'forward_pass': 0,
        'loss_computation': 0,
        'backward_pass': 0,
        'optimizer_step': 0,
        'total_iteration': 0
    }
    
    epoch_start = time.time()
    batch_count = 0
    
    print(f"\nProfiling {num_batches} batches...")
    
    for batch_idx, (images, reflectance_maps, targets, metas) in enumerate(trainer.train_data):
        iter_start = time.time()
        
        # Time data transfer to GPU
        transfer_start = time.time()
        images = images.to(trainer.gpu_id)
        metas = metas.to(trainer.gpu_id)
        reflectance_maps = reflectance_maps.to(trainer.gpu_id)
        targets = targets.to(trainer.gpu_id)
        times['data_transfer'] += time.time() - transfer_start
        
        # Normalize inputs (as in Trainer)
        norm_start = time.time()
        from master.train.train_utils import normalize_inputs
        images = normalize_inputs(images, trainer.train_mean, trainer.train_std)
        source = images, metas, reflectance_maps
        times['normalization'] += time.time() - norm_start
        
        # Forward pass
        from torch.amp import autocast
        forward_start = time.time()
        with autocast('cuda', enabled=trainer.use_amp, dtype=trainer.dtype):
            outputs = trainer.model(images, metas, target_size=targets.shape[-2:])
        times['forward_pass'] += time.time() - forward_start
        
        # Loss computation
        from master.models.losses import calculate_total_loss
        loss_start = time.time()
        with autocast('cuda', enabled=trainer.use_amp, dtype=trainer.dtype):
            loss = calculate_total_loss(
                outputs, targets, reflectance_maps, metas, 
                device=trainer.gpu_id,
                camera_params=trainer.config["CAMERA_PARAMS"], 
                hapke_params=trainer.config["HAPKE_KWARGS"],
                w_grad=trainer.config["W_GRAD"], 
                w_refl=trainer.config["W_REFL"], 
                w_mse=trainer.config["W_MSE"]
            )
        times['loss_computation'] += time.time() - loss_start
        
        # Backward pass
        backward_start = time.time()
        trainer.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), trainer.config["GRAD_CLIP"])
        times['backward_pass'] += time.time() - backward_start
        
        # Optimizer step
        opt_start = time.time()
        trainer.optimizer.step()
        times['optimizer_step'] += time.time() - opt_start
        
        times['total_iteration'] += time.time() - iter_start
        batch_count += 1
        
        # Only profile a few batches
        if batch_idx >= num_batches - 1:
            break
    
    epoch_time = time.time() - epoch_start
    
    # Calculate percentages
    print(f"\n{'='*60}")
    print(f"DETAILED TIMING BREAKDOWN ({batch_count} batches)")
    print(f"{'='*60}")
    print(f"Total time: {times['total_iteration']:.3f}s")
    print(f"\nTime breakdown:")
    for key, value in times.items():
        if key != 'total_iteration':
            pct = (value / times['total_iteration']) * 100 if times['total_iteration'] > 0 else 0
            print(f"  {key:20s}: {value:6.3f}s ({pct:5.1f}%)")
    
    # Calculate time per batch
    avg_batch_time = times['total_iteration'] / batch_count if batch_count > 0 else 0
    print(f"\nAverage time per batch: {avg_batch_time:.3f}s")
    
    # Estimate time for full epoch
    num_batches_total = len(trainer.train_data)
    estimated_epoch = avg_batch_time * num_batches_total
    print(f"Estimated full epoch time: {estimated_epoch:.1f}s ({estimated_epoch/60:.1f} min)")
    
    return times


def profile_dataloader(train_loader, num_batches=20):
    """Profile just the data loading"""
    print(f"\n{'='*60}")
    print("PROFILING DATA LOADER")
    print(f"{'='*60}")
    
    times = []
    start = time.time()
    
    for batch_idx, (images, reflectance_maps, targets, metas) in enumerate(train_loader):
        batch_time = time.time() - start
        times.append(batch_time)
        
        if batch_idx >= num_batches - 1:
            break
        
        start = time.time()
    
    avg_time = sum(times) / len(times)
    print(f"Average data loading time per batch: {avg_time:.3f}s")
    print(f"Min: {min(times):.3f}s, Max: {max(times):.3f}s")
    
    return avg_time


def main(run_dir: str, num_profile_batches: int = 10):
    """Run profiling on the training script"""
    
    # Mock DDP setup for single GPU
    os.environ["LOCAL_RANK"] = "0"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    
    # Find a free port
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    os.environ["MASTER_PORT"] = str(port)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        from torch.distributed import init_process_group
        init_process_group(backend="nccl")
    
    print(f"Using device: {device}")
    
    sup_dir = "./runs"
    run_path = os.path.join(sup_dir, run_dir)
    if not os.path.exists(run_path):
        raise RuntimeError(f"Run directory {run_path} does not exist.")
    
    config_path = os.path.join(run_path, 'stats', 'config.ini')
    config = load_config_file(config_path)
    
    # Override for profiling
    original_batch = config.get("BATCH_SIZE", 64)
    config["BATCH_SIZE"] = min(original_batch, 32)
    config["EPOCHS"] = 1
    
    print(f"Configuration: Batch size={config['BATCH_SIZE']}, "
          f"Workers={config['NUM_WORKERS_DATALOADER']}, "
          f"Prefetch={config['PREFETCH_FACTOR']}")
    
    # Load stats
    mean_std_path = os.path.join(run_path, 'stats', 'input_stats.ini')
    input_stats = read_file_from_ini(mean_std_path)
    train_mean = torch.tensor([float(input_stats['MEAN'][i]) for i in range(len(input_stats['MEAN']))])
    train_std = torch.tensor([float(input_stats['STD'][i]) for i in range(len(input_stats['STD']))])
    
    # Load training objects using the existing function
    print("\nLoading training objects...")
    with timer("Load training objects"):
        train_set, val_set, test_set, model, optimizer = load_train_objs(config, run_path)
    
    # Create data loaders
    train_loader = prepare_dataloader(
        train_set, 
        config["BATCH_SIZE"],
        num_workers=config["NUM_WORKERS_DATALOADER"],
        prefetch_factor=config["PREFETCH_FACTOR"]
    )
    
    val_loader = prepare_dataloader(
        val_set,
        config["BATCH_SIZE"],
        num_workers=config["NUM_WORKERS_DATALOADER"],
        prefetch_factor=config["PREFETCH_FACTOR"]
    )
    
    # Create a mock snapshot path (won't actually save)
    snapshot_path = os.path.join(run_path, 'checkpoints', 'profiling_snapshot.pt')
    
    # Create Trainer instance
    print("\nCreating Trainer instance...")
    with timer("Create Trainer"):
        trainer = Trainer(
            model=model,
            train_data=train_loader,
            optimizer=optimizer,
            config=config,
            snapshot_path=snapshot_path,
            train_mean=train_mean,
            train_std=train_std,
            val_data=val_loader
        )
    
    # Profile data loader
    print(f"\n{'='*60}")
    print("STEP 1: Profiling Data Loader Only")
    print(f"{'='*60}")
    
    dataloader_time = profile_dataloader(train_loader, num_batches=20)
    
    # Profile single batch generation
    print(f"\n{'='*60}")
    print("STEP 2: Profiling Single Batch Generation")
    print(f"{'='*60}")
    
    with timer("Generate single batch from dataset"):
        sample_data = train_set[0]
        print(f"Generated data with {len(sample_data)} components")
    
    # Profile training iteration using Trainer
    print(f"\n{'='*60}")
    print("STEP 3: Profiling Training Iterations (Using Trainer)")
    print(f"{'='*60}")
    
    times = profile_single_epoch(trainer, num_batches=num_profile_batches)
    
    # Use PyTorch profiler for detailed GPU/CPU analysis
    print(f"\n{'='*60}")
    print("STEP 4: PyTorch Profiler (Detailed GPU/CPU)")
    print(f"{'='*60}")
    
    # Create a simple dataloader with no workers for profiling to avoid crashes
    print("Creating simple dataloader (no workers) for profiling...")
    profile_loader = DataLoader(
        train_set,
        batch_size=config["BATCH_SIZE"],
        shuffle=False,
        num_workers=0,  # No workers to avoid multiprocessing issues with profiler
        pin_memory=False
    )
    
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA] if torch.cuda.is_available() else [ProfilerActivity.CPU],
        record_shapes=False,  # Disable shapes to reduce overhead
        with_stack=False  # Disable stack for cleaner output
    ) as prof:
        with record_function("training_iteration"):
            for batch_idx, (images, reflectance_maps, targets, metas) in enumerate(profile_loader):
                images = images.to(trainer.gpu_id)
                metas = metas.to(trainer.gpu_id)
                reflectance_maps = reflectance_maps.to(trainer.gpu_id)
                targets = targets.to(trainer.gpu_id)
                
                from master.train.train_utils import normalize_inputs
                images = normalize_inputs(images, trainer.train_mean, trainer.train_std)
                
                from torch.amp import autocast
                with autocast('cuda', enabled=trainer.use_amp, dtype=trainer.dtype):
                    outputs = trainer.model(images, metas, target_size=targets.shape[-2:])
                    from master.models.losses import calculate_total_loss
                    loss = calculate_total_loss(
                        outputs, targets, reflectance_maps, metas,
                        device=trainer.gpu_id,
                        camera_params=trainer.config["CAMERA_PARAMS"],
                        hapke_params=trainer.config["HAPKE_KWARGS"],
                        w_grad=trainer.config["W_GRAD"],
                        w_refl=trainer.config["W_REFL"],
                        w_mse=trainer.config["W_MSE"]
                    )
                
                trainer.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), trainer.config["GRAD_CLIP"])
                trainer.optimizer.step()
                
                if batch_idx >= 2:
                    break
    
    # Print profiler results
    print("\nTop 15 operations by CPU time:")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))
    
    if torch.cuda.is_available():
        print("\nTop 15 operations by CUDA time:")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
    
    # Clean up profile loader before saving
    del profile_loader
    
    # Save profiling results
    print("\nSaving results...")
    results_path = os.path.join(run_path, 'profiling_detailed_results.txt')
    try:
        with open(results_path, 'w') as f:
            f.write(f"Profiling Results for {run_dir}\n")
            f.write(f"{'='*60}\n\n")
            f.write(f"Data loader average time: {dataloader_time:.3f}s per batch\n\n")
            f.write(f"Training iteration breakdown:\n")
            for key, value in times.items():
                pct = (value / times['total_iteration']) * 100 if times['total_iteration'] > 0 else 0
                f.write(f"  {key:20s}: {value:6.3f}s ({pct:5.1f}%)\n")
            f.write(f"\n\nPyTorch Profiler Results:\n")
            f.write(f"{'='*60}\n\n")
            f.write("CPU time:\n")
            cpu_table = prof.key_averages().table(sort_by="cpu_time_total", row_limit=20)
            f.write(str(cpu_table))
            if torch.cuda.is_available():
                f.write("\n\nCUDA time:\n")
                cuda_table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=20)
                f.write(str(cuda_table))
        print(f"Results saved to: {results_path}")
    except Exception as e:
        print(f"Warning: Could not save detailed results: {e}")
        print(f"Skipping file save, continuing...")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Data loading time: {dataloader_time:.3f}s per batch")
    print(f"Compute time (forward+backward): {((times['forward_pass'] + times['backward_pass']) / times['total_iteration'] * 100):.1f}%")
    print(f"\nBottleneck: {'DATA GENERATION' if dataloader_time > 0.5 else 'COMPUTATION'}")
    print(f"\nRecommendation:")
    if dataloader_time > 0.5:
        print("  - Pre-generate and cache your dataset")
        print("  - Optimize your data generator (use vectorized ops)")
        print("  - Consider generating data in parallel offline")
    else:
        print("  - Model computation is the bottleneck")
        print("  - Consider model optimization or using multiple GPUs")
    
    # Cleanup - delete loaders first to avoid deadlock
    print("\nCleaning up...")
    del train_loader
    del val_loader
    
    # Now cleanup DDP
    from torch.distributed import destroy_process_group
    if torch.cuda.is_available():
        try:
            destroy_process_group()
            print("DDP cleanup complete")
        except Exception as e:
            print(f"Warning during DDP cleanup: {e}")
    
    print("\nProfiling complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile training script")
    parser.add_argument("run_dir", help="Run directory name")
    parser.add_argument("--batches", type=int, default=10, help="Number of batches to profile")
    
    args = parser.parse_args()
    main(args.run_dir, args.batches)
