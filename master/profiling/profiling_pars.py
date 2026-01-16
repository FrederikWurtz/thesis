"""
Quick profiling script to test different configurations
"""
import os
import sys
import time
import subprocess
import tempfile
import json

import torch

n_GPUs = torch.cuda.device_count()

# Test configurations
configs = [
    # (OMP_NUM_THREADS, BATCH_SIZE, NUM_WORKERS, PREFETCH_FACTOR)
    (2, 64, 16, 16),   # baseline optimized
    (4, 64, 16, 16),   # higher OMP
    (8, 64, 16, 16),   # even higher OMP
    (2, 128, 16, 16),  # larger batch
    (4, 128, 16, 16),  # larger batch + higher OMP
    (2, 128, 32, 16),  # larger batch + more workers
    (4, 128, 32, 16),  # larger batch + more workers + higher OMP
    (2, 256, 32, 16),  # even larger batch
    (4, 256, 32, 16),  # even larger batch + higher OMP
]

def test_config(omp_threads, batch_size, num_workers, prefetch):
    print(f"\n{'='*60}")
    print(f"Testing: OMP={omp_threads}, BATCH={batch_size}, WORKERS={num_workers}, PREFETCH={prefetch}")
    print(f"For {n_GPUs} GPUs")
    print(f"{'='*60}")
    
    # Create temporary config override file
    config_overrides = {
        "BATCH_SIZE": batch_size,
        "NUM_WORKERS_DATALOADER": num_workers,
        "PREFETCH_FACTOR": prefetch,
        "EPOCHS": 2,  # Just test 2 epochs
        "USE_PROFILER": False,
        "SKIP_TEST": True,  # Skip testing phase for profiling
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config_overrides, f)
        config_file = f.name
    
    try:
        # Set up environment
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(omp_threads)
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(n_GPUs))
        # Run torchrun
        cmd = [
            "torchrun",
            f"--nproc_per_node={n_GPUs}",
            "--standalone",
            "master/scripts/simple_multi_gpu_script.py",
            "lro_test",
            config_file,
            "--new_run"
        ]
        
        print(f"Running: {' '.join(cmd)}")
        start = time.time()
        
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            cwd="/work/FrederikWürtzSørensen#7865"
        )
        
        elapsed = time.time() - start
        
        if result.returncode != 0:
            print(f"STDERR: {result.stderr[-2000:]}")  # Print last 2000 chars
            raise RuntimeError(f"Training failed with return code {result.returncode}")
        
        # Print relevant output
        for line in result.stdout.split('\n'):
            if 'Epoch' in line or 'Time' in line or 'Loss' in line:
                print(line)
        
        print(f"Total time: {elapsed:.2f}s")
        print(f"Time per epoch: {elapsed/2:.2f}s")
        
        return elapsed / 2
        
    finally:
        # Clean up temp file
        if os.path.exists(config_file):
            os.remove(config_file)

if __name__ == "__main__":
    results = []
    for config in configs:
        try:
            time_per_epoch = test_config(*config)
            results.append((config, time_per_epoch))
        except Exception as e:
            print(f"Failed with error: {e}")
            results.append((config, None))
    
    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY FOR {n_GPUs} GPUS")
    print(f"{'='*60}")
    print(f"{'OMP':>4} | {'BATCH':>5} | {'WORKERS':>7} | {'PREFETCH':>8} | {'Time/Epoch':>12}")
    print("-" * 60)
    
    # Prepare results text
    results_text = f"\n{'='*60}\n"
    results_text += f"RESULTS SUMMARY FOR {n_GPUs} GPUS\n"
    results_text += f"{'='*60}\n"
    results_text += f"{'OMP':>4} | {'BATCH':>5} | {'WORKERS':>7} | {'PREFETCH':>8} | {'Time/Epoch':>12}\n"
    results_text += "-" * 60 + "\n"
    
    for config, time_val in results:
        if time_val:
            line = f"{config[0]:>4} | {config[1]:>5} | {config[2]:>7} | {config[3]:>8} | {time_val:>10.2f}s"
            print(line)
            results_text += line + "\n"
        else:
            line = f"{config[0]:>4} | {config[1]:>5} | {config[2]:>7} | {config[3]:>8} | {'FAILED':>12}"
            print(line)
            results_text += line + "\n"
    
    # Save to file
    results_file = "/work/FrederikWürtzSørensen#7865/runs/lro_test/profiling_results.txt"
    with open(results_file, 'w') as f:
        f.write(results_text)
    print(f"\nResults saved to: {results_file}")

