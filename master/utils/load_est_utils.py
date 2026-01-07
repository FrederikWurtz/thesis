import os
import psutil
from typing import Optional, Tuple, Dict, Any
import torch


def total_cpu_seconds(proc: psutil.Process) -> Optional[float]:
    """
    Return cumulative CPU seconds (user+system) for proc and its children.
    Best-effort - returns None on error.
    """
    try:
        t = 0.0
        ct = proc.cpu_times()
        t += float(ct.user + ct.system)
        for c in proc.children(recursive=True):
            try:
                ct = c.cpu_times()
                t += float(ct.user + ct.system)
            except Exception:
                # ignore child errors
                pass
        return t
    except Exception:
        return None


def compute_cpu_seconds_delta(start_cpu_seconds: Optional[float],
                              proc: Optional[psutil.Process] = None) -> Optional[float]:
    """
    Compute delta between current cumulative CPU seconds and start_cpu_seconds.
    If proc is None will attempt to create a psutil.Process for current pid.
    Returns None if measurement unavailable.
    """
    try:
        if start_cpu_seconds is None:
            return None
        if proc is not None:
            end = total_cpu_seconds(proc)
        else:
            p = psutil.Process(os.getpid())
            ct = p.cpu_times()
            end = float(ct.user + ct.system)
        if end is None:
            return None
        return max(0.0, float(end - start_cpu_seconds))
    except Exception:
        return None


def compute_timing_info(start_cpu_seconds: Optional[float],
                        total_time_seconds: float,
                        device: Optional[Any] = None,
                        proc: Optional[psutil.Process] = None,
                        epochs: Optional[int] = None) -> Tuple[Dict[str, Any], Optional[float]]:
    """
    Consolidate wall/cpu/gpu timing info into a dictionary.

    Args:
        start_cpu_seconds: value recorded at training start (or None)
        total_time_seconds: wall-clock seconds elapsed (float)
        device: torch.device or string-like (may be None)
        proc: optional psutil.Process instance used to measure CPU seconds (best to pass the same proc used at start)

    Returns:
        (timing_info_dict, cpu_seconds_delta_or_None)
    """
    cpu_seconds_delta = compute_cpu_seconds_delta(start_cpu_seconds, proc=proc)

    gpu_hours_estimate = 0.0
    gpu_device = 'N/A'
    num_gpus = 0
    try:
        if device is not None:
            # device may be a torch.device or string
            dev_type = getattr(device, "type", str(device))
            gpu_device = str(dev_type)

            # Basic heuristic: if device is a GPU type, estimate GPU-hours as
            # wall time * number of GPUs. This handles single-node multi-GPU
            # cases and attempts to account for distributed runs via WORLD_SIZE.
            if str(dev_type) in ('cuda', 'mps'):
                num_gpus = 1
                try:
                    import torch
                    if hasattr(torch, "cuda") and torch.cuda.is_available():
                        num_gpus = max(1, torch.cuda.device_count())
                except Exception:
                    # torch not available or failed - fall back to env
                    pass

                try:
                    world_size = int(os.environ.get("WORLD_SIZE", "0"))
                    if world_size > 0:
                        # WORLD_SIZE often equals total processes across nodes;
                        # use the larger of device count and world size as a heuristic.
                        num_gpus = max(num_gpus, world_size)
                except Exception:
                    pass

                gpu_hours_estimate = (float(total_time_seconds) / 3600.0) * float(num_gpus)
                # include count in device string for clarity
                gpu_device = gpu_device
                num_gpus = num_gpus
    except Exception:
        pass

    timing_info = {
        # wall-clock (seconds and hours)
        'wall_seconds': float(total_time_seconds),
        'wall_hours': float(total_time_seconds) / 3600.0,
        'epochs': epochs,

        # cpu (seconds and hours)
        'cpu_seconds': float(cpu_seconds_delta) if cpu_seconds_delta is not None else None,
        'cpu_hours': (float(cpu_seconds_delta) / 3600.0) if cpu_seconds_delta is not None else None,
        'cpu_seconds_per_epoch': (float(cpu_seconds_delta) / float(epochs)) if (cpu_seconds_delta is not None and epochs is not None and epochs > 0) else None,


        # gpu info (device, hours estimate and seconds variant)
        'gpu_device': gpu_device,
        'num_gpus': num_gpus,
        'gpu_hours': float(gpu_hours_estimate),
        'gpu_seconds': float(gpu_hours_estimate * 3600.0),  # seconds equivalent (0.0 if no GPU)
        'gpu_seconds_per_epoch': (float(gpu_hours_estimate * 3600.0) / float(epochs)) if (epochs is not None and epochs > 0) else None,
        
        'cpu_measurement_available': cpu_seconds_delta is not None
    }
    return timing_info, cpu_seconds_delta

def print_timing_info(timing_info: Dict[str, Any]) -> None:
    print("="*60)
    print("Timing Summary:")
    print(f"  Epochs:         {timing_info['epochs']}")
    print(f"  Wall-clock time: {timing_info['wall_seconds']:.0f} seconds ({timing_info['wall_hours']:.2f} hours)")
    print(f"  CPU time:        {timing_info['cpu_seconds']:.0f} seconds ({timing_info['cpu_hours']:.2f} hours)")
    print(f"  CPU time per epoch: {timing_info['cpu_seconds_per_epoch']:.2f} seconds")
    print(f"  GPU time:        {timing_info['gpu_seconds']:.0f} seconds ({timing_info['gpu_hours']:.2f} hours)")
    print(f"  GPU time per epoch: {timing_info['gpu_seconds_per_epoch']:.2f} seconds")
    print("GPU Device:", timing_info['gpu_device'])
    print("Number of GPUs:", timing_info['num_gpus'])
    print("="*60)


def log_memory(tag="", verbose = True, device=None, proc: Optional[psutil.Process] = None) -> None:
    try:
        rss = proc.memory_info().rss / (1024**2)
        vms = proc.memory_info().vms / (1024**2)
        if device.type == 'cuda' and torch.cuda.is_available():
            cuda_alloc = torch.cuda.memory_allocated(device) / (1024**2)
            cuda_reserved = torch.cuda.memory_reserved(device) / (1024**2)
            if verbose:
                print(f"[MEM] {tag} RSS={rss:.1f}MB VMS={vms:.1f}MB CUDA_alloc={cuda_alloc:.1f}MB CUDA_reserved={cuda_reserved:.1f}MB")
        if device.type == 'mps' and hasattr(torch.mps, 'current_allocated_memory'):
            mps_alloc = torch.mps.current_allocated_memory() / (1024**2)
            if verbose:
                print(f"[MEM] {tag} RSS={rss:.1f}MB VMS={vms:.1f}MB MPS_alloc={mps_alloc:.1f}MB")
        else:
            if verbose:
                print(f"[MEM] {tag} RSS={rss:.1f}MB VMS={vms:.1f}MB")
    except Exception:
        pass


__all__ = ['total_cpu_seconds',
           'compute_cpu_seconds_delta',
           'compute_timing_info',
           'print_timing_info',
           'log_memory']