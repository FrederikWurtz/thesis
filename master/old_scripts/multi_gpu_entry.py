import torch
from master.train import cli
import os
import torch.distributed as dist
import time

from master.train.checkpoints import read_file_from_ini, save_file_as_ini
from master.train.cli_multi_gpu import multi_gpu_cli_no_training
from master.train.train_utils import init_distributed, get_device
from master.train.runner_multi_gpu import launch_multi_gpu_training
from master.train.trainer_core import evaluate_on_test_files
from master.utils.load_est_utils import compute_timing_info, print_timing_info


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

def main(argv=None):
    # Distributed Data Parallel setup (if applicable)
    t0 = time.time()

    # Use CUDA for multi-GPU, check if available
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Multi-GPU training requires CUDA.")
    
    
    backend = 'nccl' 
    rank, world_size = init_distributed(backend=backend)
    if world_size == 1:
        raise RuntimeError(f"World size is 1; "
            "This script is intended for multi-GPU training only.")
    if is_main_process():
        print(f"Distributed training initialized. Rank: {rank}, World Size: {world_size}")
    

    # Parse arguments and set up distributed training
    if is_main_process():
        config, run_dir, val_dir, test_dir, args = multi_gpu_cli_no_training(argv)
    else:
        config, run_dir, val_dir, test_dir, args = None, None, None, None, None

    # Broadcast til alle ranks
    object_list = [config, run_dir, val_dir, test_dir, args]
    dist.broadcast_object_list(object_list, src=0)
    config, run_dir, val_dir, test_dir, args = object_list

    # Launch multiple processes for multi-GPU training
    returned_values = launch_multi_gpu_training(config=config, run_dir=run_dir, val_dir=val_dir, test_dir=test_dir, 
                                                new_training=args.new_training, world_size=world_size, rank=rank)
    (
        best_epoch,
        start_epoch,
        end_epoch,
        model,
        test_loader,
        train_mean,
        train_std,
        camera_params,
        hapke_params,
        use_amp,
        non_blocking
    ) = returned_values

    dist.barrier() # ensure all processes have finished training

    if is_main_process():
        print(f"Multi-GPU training setup complete. Best Epoch: {best_epoch}, Start Epoch: {start_epoch}, End Epoch: {end_epoch}")
        print("Performing evaluation on the test files using best checkpoint.")

    # Evaluate on test set - only on main process
    if is_main_process():
        print("Evaluating on test set...")
        test_loss, test_ame = evaluate_on_test_files(model = model, 
                                                    test_loader = test_loader, 
                                                    device = get_device(),
                                                    train_mean = train_mean,
                                                    train_std = train_std,
                                                    camera_params = camera_params,
                                                    hapke_params = hapke_params,
                                                    use_amp = use_amp,
                                                    w_mse = config["W_MSE"],
                                                    w_grad = config["W_GRAD"],
                                                    w_refl = config["W_REFL"],
                                                    non_blocking = non_blocking)
    
        print(f"Test loss: {test_loss:.4f}, Test AME: {test_ame:.4f}")
        run_stats_dict = {
            'best_epoch': best_epoch,
            'test_loss': test_loss,
            'test_ame': test_ame
        }
        stats_dir = os.path.join(run_dir, 'stats')
        os.makedirs(stats_dir, exist_ok=True)
        run_stats_path = os.path.join(stats_dir, 'run_stats.ini')
        save_file_as_ini(run_stats_dict, run_stats_path)
        print(f"Saved run statistics to {run_stats_path}")


    if is_main_process():
        # Final timing info
        total_time = time.time() - t0
        timing_info = {'wall_seconds': total_time}
        timing_path = os.path.join(run_dir, 'stats', 'timing_info.ini')

        #if this is not a new training, get earlier timing info and add to current
        if not args.new_run:
            if os.path.exists(timing_path):
                previous_timing_info = read_file_from_ini(timing_path)
                # accumulate times
                for key in ['wall_seconds']:
                    if key in previous_timing_info and key in timing_info:
                        timing_info[key] += previous_timing_info[key]
            else:
                print(f"Warning: timing info file {timing_path} not found for continuation run.")
                        
        save_file_as_ini(timing_info, timing_path)
        print(f"Saved timing information to {timing_path}")
        print_timing_info(timing_info)

if __name__ == '__main__':
    main(argv=None)












