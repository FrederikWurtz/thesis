"""
Advanced epoch profiler with detailed CPU-GPU synchronization analysis
Shows where data transfers, kernel launches, and waits occur
"""
import os
import sys
import json
import time
import tempfile
import subprocess

import torch
import torch.cuda

def create_analysis_script():
    """Create a script that runs inside the trainer to get detailed metrics"""
    return """
import torch
import torch.cuda as cuda

class DetailedProfiler:
    def __init__(self):
        self.events = []
        self.cuda_events = {}
        
    def record(self, name, phase='start'):
        '''Record CPU timestamp'''
        self.events.append({
            'name': name,
            'phase': phase,
            'cpu_time': time.time(),
            'cuda_sync': cuda.is_available() and cuda.current_device() >= 0
        })
    
    def cuda_event(self, name):
        '''Record CUDA event for synchronization timing'''
        if cuda.is_available():
            self.cuda_events[name] = cuda.Event(enable_timing=True)
            self.cuda_events[name].record()
    
    def report(self):
        '''Print detailed timing report'''
        print("\\n" + "="*70)
        print("DETAILED PROFILING REPORT")
        print("="*70)
        
        if not self.events:
            print("No events recorded")
            return
        
        start_time = self.events[0]['cpu_time']
        last_time = start_time
        
        for event in self.events:
            delta = (event['cpu_time'] - last_time) * 1000  # ms
            elapsed = (event['cpu_time'] - start_time) * 1000  # ms
            print(f"[{elapsed:8.2f}ms] ({delta:6.2f}ms) {event['name']}")
            last_time = event['cpu_time']
        
        print("="*70 + "\\n")
"""

def run_detailed_profile(run_dir="lro_test", batch_size=64, num_workers=16, prefetch=16, omp_threads=4, num_batches=3):
    """Run detailed profiling with synchronization analysis"""
    
    print(f"\n{'='*70}")
    print(f"ADVANCED EPOCH PROFILER - CPU-GPU SYNCHRONIZATION ANALYSIS")
    print(f"{'='*70}")
    print(f"\nConfiguration:")
    print(f"  Run directory: {run_dir}")
    print(f"  Batch size: {batch_size}")
    print(f"  Workers: {num_workers}")
    print(f"  Prefetch: {prefetch}")
    print(f"  OMP threads: {omp_threads}")
    print(f"  Batches to profile: {num_batches}")
    print(f"\nWhat to look for:")
    print(f"  1. Data loading spikes (CPU time without GPU work)")
    print(f"  2. CPU-GPU transfer delays (cuda.synchronize() calls)")
    print(f"  3. Kernel launch overhead (gap between CPU record and GPU start)")
    print(f"  4. Idle GPU time (CPU waiting for GPU)")
    print(f"{'='*70}\n")
    
    # Create config
    config_overrides = {
        "BATCH_SIZE": batch_size,
        "NUM_WORKERS_DATALOADER": num_workers,
        "PREFETCH_FACTOR": prefetch,
        "EPOCHS": 1,
        "SAVE_EVERY": 10,  # Don't save yet
        "USE_PROFILER": True,
        "SKIP_TEST": True,
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config_overrides, f)
        config_file = f.name
    
    try:
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(omp_threads)
        env["CUDA_LAUNCH_BLOCKING"] = "1"
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        
        # Run with simplified profiling
        cmd = [
            "torchrun",
            "--nproc_per_node=1",
            "--standalone",
            "master/scripts/simple_multi_gpu_script.py",
            run_dir,
            config_file,
            "--new_run"
        ]
        
        print(f"Launching profiler (this will take 1-2 minutes)...\n")
        
        start = time.time()
        result = subprocess.run(
            cmd,
            env=env,
            cwd="/work/FrederikWürtzSørensen#7865"
        )
        elapsed = time.time() - start
        
        if result.returncode == 0:
            print(f"\n{'='*70}")
            print(f"✓ PROFILING COMPLETE")
            print(f"{'='*70}")
            print(f"Elapsed time: {elapsed:.2f}s")
            
            # Print next steps
            print(f"\nNext steps:")
            print(f"1. Check profiler output in: ./runs/{run_dir}/profiler/")
            print(f"2. Open in Chrome tracing: chrome://tracing/")
            print(f"3. Look for:")
            print(f"   - Long yellow bars = CPU work (data loading)")
            print(f"   - Long purple/pink bars = GPU kernels")
            print(f"   - Gaps = synchronization/transfer overhead")
            print(f"\nCommon bottlenecks:")
            print(f"  [CPU stalls] → Need more workers or prefetch")
            print(f"  [GPU idle]   → Model is too small or data loader too slow")
            print(f"  [Transfers]  → Host↔Device bandwidth limited")
            
            return True
        else:
            print(f"\n❌ Profiling failed with code {result.returncode}")
            return False
            
    finally:
        if os.path.exists(config_file):
            os.remove(config_file)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Advanced profiler showing CPU-GPU synchronization"
    )
    parser.add_argument("--run_dir", default="lro_test", help="Run directory")
    parser.add_argument("--batch", type=int, default=64, help="Batch size")
    parser.add_argument("--workers", type=int, default=16, help="Workers")
    parser.add_argument("--prefetch", type=int, default=16, help="Prefetch factor")
    parser.add_argument("--omp", type=int, default=4, help="OMP threads")
    
    args = parser.parse_args()
    
    success = run_detailed_profile(
        run_dir=args.run_dir,
        batch_size=args.batch,
        num_workers=args.workers,
        prefetch=args.prefetch,
        omp_threads=args.omp
    )
    
    sys.exit(0 if success else 1)
