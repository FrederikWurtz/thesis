"""
Profiling script specifically for the loss computation
Breaks down which loss component is the bottleneck
"""
import os
import sys
import time
import argparse
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity

from master.models.losses import calculate_total_loss, compute_reflectance_map_from_dem
from master.train.trainer_core import FluidDEMDataset
from master.train.checkpoints import read_file_from_ini
from master.configs.config_utils import load_config_file


@contextmanager
def timer(name):
    """Context manager to time code blocks"""
    start = time.time()
    yield
    elapsed = time.time() - start
    print(f"[TIMER] {name}: {elapsed:.3f}s")


def profile_loss_components(outputs, targets, reflectance_maps, metas, device, camera_params, hapke_params, num_runs=10):
    """Profile each component of the loss function separately"""
    
    print(f"\n{'='*60}")
    print(f"Profiling Loss Components ({num_runs} runs)")
    print(f"{'='*60}")
    
    # 1. MSE Loss
    mse_times = []
    for _ in range(num_runs):
        start = time.time()
        loss_mse = F.mse_loss(outputs, targets, reduction='mean')
        torch.cuda.synchronize()  # Wait for GPU to finish
        mse_times.append(time.time() - start)
    
    avg_mse_time = sum(mse_times) / len(mse_times)
    print(f"\nMSE Loss: {avg_mse_time*1000:.3f}ms (avg over {num_runs} runs)")
    print(f"  Value: {loss_mse.item():.6f}")
    
    # 2. Gradient Loss
    def compute_gradients(tensor):
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=tensor.dtype, device=tensor.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                               dtype=tensor.dtype, device=tensor.device).view(1, 1, 3, 3)
        
        grad_x = F.conv2d(tensor, sobel_x, padding=1)
        grad_y = F.conv2d(tensor, sobel_y, padding=1)
        grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
        return grad_magnitude
    
    grad_times = []
    for _ in range(num_runs):
        start = time.time()
        out_grad_mag = compute_gradients(outputs)
        tgt_grad_mag = compute_gradients(targets)
        loss_grad = F.mse_loss(out_grad_mag, tgt_grad_mag, reduction='mean')
        torch.cuda.synchronize()
        grad_times.append(time.time() - start)
    
    avg_grad_time = sum(grad_times) / len(grad_times)
    print(f"\nGradient Loss: {avg_grad_time*1000:.3f}ms (avg over {num_runs} runs)")
    print(f"  Value: {loss_grad.item():.6f}")
    
    # 3. Reflectance Loss (THE BOTTLENECK)
    print(f"\nReflectance Loss Computation:")
    print(f"  Computing reflectance maps from predicted DEM...")
    
    refl_times = []
    for i in range(num_runs):
        start = time.time()
        predicted_reflectance_maps = compute_reflectance_map_from_dem(
            outputs, metas, device, camera_params, hapke_params
        )
        torch.cuda.synchronize()
        elapsed = time.time() - start
        refl_times.append(elapsed)
        if i == 0:
            print(f"  First run: {elapsed:.3f}s")
    
    avg_refl_time = sum(refl_times) / len(refl_times)
    print(f"  Average: {avg_refl_time:.3f}s (avg over {num_runs} runs)")
    
    loss_refl = F.mse_loss(predicted_reflectance_maps, reflectance_maps, reduction='mean')
    print(f"  Value: {loss_refl.item():.6f}")
    
    # Summary
    total_avg_time = avg_mse_time + avg_grad_time + avg_refl_time
    print(f"\n{'='*60}")
    print("BREAKDOWN SUMMARY")
    print(f"{'='*60}")
    print(f"MSE Loss:         {avg_mse_time*1000:8.3f}ms ({avg_mse_time/total_avg_time*100:5.1f}%)")
    print(f"Gradient Loss:    {avg_grad_time*1000:8.3f}ms ({avg_grad_time/total_avg_time*100:5.1f}%)")
    print(f"Reflectance Loss: {avg_refl_time*1000:8.3f}ms ({avg_refl_time/total_avg_time*100:5.1f}%)")
    print(f"{'='*60}")
    print(f"Total:            {total_avg_time*1000:8.3f}ms")
    
    return {
        'mse_time': avg_mse_time,
        'grad_time': avg_grad_time,
        'refl_time': avg_refl_time,
        'total_time': total_avg_time
    }


def profile_reflectance_computation_detailed(outputs, metas, device, camera_params, hapke_params):
    """Profile the reflectance computation in more detail"""
    
    print(f"\n{'='*60}")
    print("DETAILED REFLECTANCE COMPUTATION PROFILING")
    print(f"{'='*60}")
    
    B, _, H, W = outputs.shape
    print(f"Batch size: {B}, Image shape: ({H}, {W})")
    print(f"Processing {B} batches × 5 images = {B*5} total renders")
    
    # Import here to avoid circular imports
    from master.render.dem_utils import DEM
    from master.render.camera import Camera
    from master.render.renderer import Renderer
    from master.render.hapke_model import HapkeModel
    
    # Profile per-batch processing
    batch_times = []
    render_times = []
    setup_times = []
    
    total_start = time.time()
    
    for b in range(B):
        batch_start = time.time()
        
        # Time DEM creation and setup
        setup_start = time.time()
        dem_np = outputs[b, 0].detach().cpu().numpy()
        dem_obj = DEM(dem_np, cellsize=1, x0=0, y0=0)
        camera = Camera(
            image_width=camera_params['image_width'],
            image_height=camera_params['image_height'],
            focal_length=camera_params['focal_length'],
            device=device
        )
        hapke_model = HapkeModel(
            w=hapke_params['w'], 
            B0=hapke_params['B0'],
            h=hapke_params['h'], 
            phase_fun=hapke_params['phase_fun'],
            xi=hapke_params['xi']
        )
        renderer = Renderer(dem_obj, hapke_model, camera)
        setup_time = time.time() - setup_start
        setup_times.append(setup_time)
        
        # Time rendering for 5 images
        batch_render_times = []
        for img_idx in range(5):
            sun_az = metas[b, img_idx, 0].item()
            sun_el = metas[b, img_idx, 1].item()
            cam_az = metas[b, img_idx, 2].item()
            cam_el = metas[b, img_idx, 3].item()
            cam_dist = metas[b, img_idx, 4].item()
            
            render_start = time.time()
            renderer.render_shading(
                sun_az_deg=sun_az,
                sun_el_deg=sun_el,
                camera_az_deg=cam_az,
                camera_el_deg=cam_el,
                camera_distance_from_center=cam_dist,
                model="hapke"
            )
            refl_map = renderer.reflectance_map
            batch_render_times.append(time.time() - render_start)
        
        avg_render_time = sum(batch_render_times) / len(batch_render_times)
        render_times.append(avg_render_time)
        
        batch_time = time.time() - batch_start
        batch_times.append(batch_time)
        
        if b == 0:
            print(f"\nFirst batch timing:")
            print(f"  Setup (DEM, Camera, Renderer): {setup_time*1000:.3f}ms")
            print(f"  Average render per image:       {avg_render_time*1000:.3f}ms")
            print(f"  Total for 5 images:             {sum(batch_render_times)*1000:.3f}ms")
            print(f"  Total batch time:               {batch_time*1000:.3f}ms")
    
    total_time = time.time() - total_start
    
    avg_batch_time = sum(batch_times) / len(batch_times)
    avg_setup_time = sum(setup_times) / len(setup_times)
    avg_render_time = sum(render_times) / len(render_times)
    
    print(f"\nAverages across {B} batches:")
    print(f"  Setup per batch:        {avg_setup_time*1000:.3f}ms")
    print(f"  Render per image:       {avg_render_time*1000:.3f}ms")
    print(f"  Total per batch:        {avg_batch_time*1000:.3f}ms")
    print(f"  Total for all batches:  {total_time:.3f}s")
    
    print(f"\n{'='*60}")
    print("BOTTLENECK ANALYSIS")
    print(f"{'='*60}")
    print(f"Setup overhead:   {avg_setup_time/avg_batch_time*100:.1f}%")
    print(f"Rendering:        {(avg_render_time*5)/avg_batch_time*100:.1f}%")
    print(f"\nThe rendering loop processes {B*5} images sequentially on CPU.")
    print(f"Each render_shading call: ~{avg_render_time*1000:.1f}ms")
    print(f"Total time in rendering: ~{avg_render_time*5*B:.3f}s")


def main(run_dir: str, batch_size: int = 8, num_runs: int = 10):
    """Profile the loss computation"""
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    sup_dir = "./runs"
    run_path = os.path.join(sup_dir, run_dir)
    if not os.path.exists(run_path):
        raise RuntimeError(f"Run directory {run_path} does not exist.")
    
    config_path = os.path.join(run_path, 'stats', 'config.ini')
    config = load_config_file(config_path)
    
    print(f"\n{'='*60}")
    print("SETUP")
    print(f"{'='*60}")
    print(f"Batch size: {batch_size}")
    print(f"Number of profiling runs: {num_runs}")
    
    # Load dataset to get sample data
    print("\nLoading dataset...")
    with timer("Load dataset"):
        dataset = FluidDEMDataset(config)
    
    # Generate a batch of data
    print(f"Generating {batch_size} samples...")
    samples = []
    with timer(f"Generate {batch_size} samples"):
        for i in range(batch_size):
            samples.append(dataset[i])
    
    # Collate into batch
    images = torch.stack([s[0] for s in samples]).to(device)
    reflectance_maps = torch.stack([s[1] for s in samples]).to(device)
    targets = torch.stack([s[2] for s in samples]).to(device)
    metas = torch.stack([s[3] for s in samples]).to(device)
    
    print(f"\nData shapes:")
    print(f"  Images: {images.shape}")
    print(f"  Reflectance maps: {reflectance_maps.shape}")
    print(f"  Targets: {targets.shape}")
    print(f"  Metas: {metas.shape}")
    
    # Create fake outputs (use targets as placeholder)
    outputs = targets.clone() + torch.randn_like(targets) * 0.1
    
    # Get camera and hapke params from config
    camera_params = config["CAMERA_PARAMS"]
    hapke_params = config["HAPKE_KWARGS"]
    
    # Profile total loss computation
    print(f"\n{'='*60}")
    print("PROFILING TOTAL LOSS COMPUTATION")
    print(f"{'='*60}")
    
    total_times = []
    for i in range(num_runs):
        start = time.time()
        loss = calculate_total_loss(
            outputs, targets, reflectance_maps, metas,
            device=device,
            camera_params=camera_params,
            hapke_params=hapke_params,
            w_grad=config["W_GRAD"],
            w_refl=config["W_REFL"],
            w_mse=config["W_MSE"]
        )
        torch.cuda.synchronize()
        elapsed = time.time() - start
        total_times.append(elapsed)
        
        if i == 0:
            print(f"First run: {elapsed:.3f}s")
            if hasattr(loss, 'loss_components'):
                print(f"  MSE:  {loss.loss_components['mse']:.6f}")
                print(f"  Grad: {loss.loss_components['grad']:.6f}")
                print(f"  Refl: {loss.loss_components['refl']:.6f}")
    
    avg_total_time = sum(total_times) / len(total_times)
    print(f"\nAverage total loss computation: {avg_total_time:.3f}s")
    print(f"Time per batch item: {avg_total_time/batch_size:.3f}s")
    
    # Profile individual components
    component_times = profile_loss_components(
        outputs, targets, reflectance_maps, metas,
        device, camera_params, hapke_params, num_runs=num_runs
    )
    
    # Detailed reflectance profiling (single run)
    profile_reflectance_computation_detailed(
        outputs, metas, device, camera_params, hapke_params
    )
    
    # PyTorch profiler for one iteration
    print(f"\n{'='*60}")
    print("PYTORCH PROFILER (Detailed)")
    print(f"{'='*60}")
    
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA] if torch.cuda.is_available() else [ProfilerActivity.CPU],
        record_shapes=False,
        with_stack=False
    ) as prof:
        with record_function("calculate_total_loss"):
            loss = calculate_total_loss(
                outputs, targets, reflectance_maps, metas,
                device=device,
                camera_params=camera_params,
                hapke_params=hapke_params,
                w_grad=config["W_GRAD"],
                w_refl=config["W_REFL"],
                w_mse=config["W_MSE"]
            )
    
    print("\nTop 20 operations by CPU time:")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
    
    if torch.cuda.is_available():
        print("\nTop 20 operations by CUDA time:")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    
    # Save results
    results_path = os.path.join(run_path, 'loss_profiling_results.txt')
    with open(results_path, 'w') as f:
        f.write(f"Loss Computation Profiling Results\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Number of runs: {num_runs}\n\n")
        f.write(f"Average total loss time: {avg_total_time:.3f}s\n")
        f.write(f"Time per batch item: {avg_total_time/batch_size:.3f}s\n\n")
        f.write(f"Component breakdown:\n")
        f.write(f"  MSE Loss:         {component_times['mse_time']*1000:8.3f}ms ({component_times['mse_time']/component_times['total_time']*100:5.1f}%)\n")
        f.write(f"  Gradient Loss:    {component_times['grad_time']*1000:8.3f}ms ({component_times['grad_time']/component_times['total_time']*100:5.1f}%)\n")
        f.write(f"  Reflectance Loss: {component_times['refl_time']*1000:8.3f}ms ({component_times['refl_time']/component_times['total_time']*100:5.1f}%)\n\n")
        f.write(f"PyTorch Profiler Results:\n")
        f.write(f"{'='*60}\n\n")
        f.write("CPU time:\n")
        f.write(prof.key_averages().table(sort_by="cpu_time_total", row_limit=30))
    
    print(f"\nResults saved to: {results_path}")
    
    # Final summary
    print(f"\n{'='*60}")
    print("OPTIMIZATION RECOMMENDATIONS")
    print(f"{'='*60}")
    print(f"1. The reflectance computation takes {component_times['refl_time']/component_times['total_time']*100:.1f}% of total loss time")
    print(f"2. Each batch processes {batch_size * 5} images sequentially on CPU")
    print(f"3. Object creation (DEM, Camera, Renderer) happens in a loop")
    print(f"\nSuggestions:")
    print(f"  - Batch the rendering operations")
    print(f"  - Move rendering to GPU if possible")
    print(f"  - Reuse Camera and HapkeModel objects across batches")
    print(f"  - Consider caching/pre-computing reflectance maps")
    print(f"  - Parallelize the 5-image loop with multiprocessing")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile loss computation")
    parser.add_argument("run_dir", help="Run directory name")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size to profile")
    parser.add_argument("--num_runs", type=int, default=10, help="Number of profiling runs")
    
    args = parser.parse_args()
    main(args.run_dir, args.batch_size, args.num_runs)
