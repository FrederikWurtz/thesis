"""Training runner: orchestrates dataset loading, training loop and checkpointing.

This module provides a `run_training` function and a CLI `main()` wrapper.
It relies on the core trainer components in `trainer_core` and the
`checkpoints` utilities.
"""

import os
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import psutil
import gc
import torch.multiprocessing as mp
import torch.distributed as dist

from master.train.trainer_core import DEMDataset, FluidDEMDataset, DEMDatasetHDF5, SemifluidDEMDataset
from master.models.unet import UNet
from master.train.trainer_core import train_epoch, validate_epoch, estimate_dynamic_batch_size
from master.train.train_utils import get_device, compute_input_stats, round_list, collate, init_distributed, convert_npz_to_hdf5, worker_init_fn
from master.train.checkpoints import read_file_from_ini, save_last_and_best, load_checkpoint, save_checkpoint, save_file_as_ini
from master.data_sim import dataset_io
from master.utils.interactivity_utils import input_listener_context, handle_input_commands
from master.utils.load_est_utils import log_memory

def run_fluid_training(config=None, run_dir=None, val_dir=None, test_dir=None, new_training=None):
    if config is None:
        raise ValueError("config must be provided")
    if run_dir is None:
        raise ValueError("run_dir must be provided")
    if val_dir is None:
        raise ValueError("val_dir must be provided")
    if test_dir is None:
        raise ValueError("test_dir must be provided")

    # estimate_dynamic_batch_size = False # set to True to enable dynamic batch size estimation

    # get device
    device = get_device()
    print(f"Using device: {device}")
        # Mixed precision setup if applicable
    use_amp = True if device.type == 'cuda' else False
    pin_memory = (device.type == 'cuda')
    non_blocking = (device.type == 'cuda')
    autocast_device_type = device.type if device.type in ['cuda', 'mps'] else 'cpu'
    scaler = torch.amp.GradScaler(enabled=use_amp)
    if device.type == 'mps':
        print("\nUsing Apple Silicon MPS backend")
        print('Using Autocast for MPS, but GradScaler is disabled (no FP16 support on MPS)')
        print("pin_memory set to False for MPS")
        print("non_blocking transfers disabled for MPS")
    elif device.type == 'cuda':
        print("\nUsing mixed precision training (FP16)")
        print("pin_memory set to True for CUDA")
        print("non_blocking transfers enabled for CUDA")
    else:
        print("\nUsing CPU backend")
        print("pin_memory set to False for CPU")
        print("non_blocking transfers disabled for CPU")


    # memory/process helper - change verbose to False to disable logging
    proc = psutil.Process()
    log_memory("Startup", device=device, proc=proc, verbose=False)

    # prepare datasets and dataloaders
    val_files = dataset_io.list_npz_files(val_dir)
    if len(val_files) != config["FLUID_VAL_DEMS"]:
        raise RuntimeError(f'Number of validation DEMs found ({len(val_files)}) does not match config setting FLUID_VAL_DEMS ({config["FLUID_VAL_DEMS"]})')
    # also check test if test files exist
    test_files = dataset_io.list_npz_files(test_dir)
    if len(test_files) != config["FLUID_TEST_DEMS"]:
        raise RuntimeError(f'Number of test DEMs found ({len(test_files)}) does not match config setting FLUID_TEST_DEMS ({config["FLUID_TEST_DEMS"]})')

    # initialize model
    model = UNet(in_channels=config["IMAGES_PER_DEM"], out_channels=1)
    model.to(device)

    # setup datasets and dataloaders
    if config["USE_SEMIFLUID"] is True: # use semi-fluid dataset for training dataset
        print(f"Using Semi-Fluid DEM Dataset with reuse limit of {config['REUSE_LIMIT']} for training dataset")
        temporary_dir = os.path.join(run_dir, 'semifluid_temp')
        os.makedirs(temporary_dir, exist_ok=True)
        train_ds = SemifluidDEMDataset(config=config, temporary_dir=temporary_dir, reuse_limit=config["REUSE_LIMIT"])
    else:
        print("Using Fluid DEM Dataset for training dataset")
        train_ds = FluidDEMDataset(config)

    if config["USE_HDF5"] is True:
        print("Using HDF5 datasets for validation and test datasets")
        # check if HDF5 files already exist; if not, convert from NPZ
        val_dataset_path = os.path.join(val_dir, 'validation_data.h5')
        test_dataset_path = os.path.join(test_dir, 'test_data.h5')
        if not os.path.exists(val_dataset_path) or not os.path.exists(test_dataset_path):
            print("Converting NPZ files to HDF5 format for faster loading...")
            convert_npz_to_hdf5(
                                npz_dir=val_dir,
                                output_path=os.path.join(val_dir, 'validation_data.h5'),
                            )           
            convert_npz_to_hdf5(
                                npz_dir=test_dir,
                                output_path=os.path.join(test_dir, 'test_data.h5'),
                            )
            
        val_ds = DEMDatasetHDF5(val_dataset_path)
        test_ds = DEMDatasetHDF5(test_dataset_path)

    else:   
        val_ds = DEMDataset(val_files)
        test_ds = DEMDataset(test_files)


    # quickly fetch learning rate from config
    if not new_training:
        checkpoint_dir = os.path.join(run_dir, 'checkpoints')
        last_checkpoint = os.path.join(checkpoint_dir, 'last.pth')
        checkpoint = load_checkpoint(last_checkpoint, map_location=device)
        current_lr = checkpoint['optimizer_state_dict']['param_groups'][0]['lr']
    else:
        current_lr = config["LR"]

    # initialize optimizer, scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=current_lr, weight_decay=config["WEIGHT_DECAY"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=config["LR_PATIENCE"])

    # Distributed Data Parallel setup (if applicable)
    backend = 'nccl' if device.type == 'cuda' else 'gloo'
    rank, world_size = init_distributed(backend=backend)
    sampler = None

    if world_size > 1:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        print(f"Wrapping model in DDP on rank {rank}/{world_size}...")
        model = DDP(model, device_ids=[local_rank] if device.type == 'cuda' else None)
        sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank)

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=config["BATCH_SIZE"], shuffle=True, num_workers=config["NUM_WORKERS"], 
                             collate_fn=collate, pin_memory=pin_memory, sampler=sampler, persistent_workers=True, prefetch_factor=4)
    
    if config["USE_HDF5"] is True:
        val_loader = DataLoader(val_ds, batch_size=config["BATCH_SIZE"], shuffle=True, num_workers=config["NUM_WORKERS"], 
                                collate_fn=collate, pin_memory=pin_memory, sampler=sampler, persistent_workers=True, prefetch_factor=4,
                                worker_init_fn=worker_init_fn)
        test_loader = DataLoader(test_ds, batch_size=config["BATCH_SIZE"], shuffle=True, num_workers=config["NUM_WORKERS"], 
                                collate_fn=collate, pin_memory=pin_memory, sampler=sampler, persistent_workers=True, prefetch_factor=4,
                                worker_init_fn=worker_init_fn)
        
    else:
        val_loader = DataLoader(val_ds, batch_size=config["BATCH_SIZE"], shuffle=True, num_workers=config["NUM_WORKERS"], 
                                collate_fn=collate, pin_memory=pin_memory, sampler=sampler, persistent_workers=True, prefetch_factor=4)
        test_loader = DataLoader(test_ds, batch_size=config["BATCH_SIZE"], shuffle=True, num_workers=config["NUM_WORKERS"], 
                                collate_fn=collate, pin_memory=pin_memory, sampler=sampler, persistent_workers=True, prefetch_factor=4)

    # branching logic: are we starting new training or resuming?
    if new_training:
        print("="*60)
        start_epoch = 1
        end_epoch = config["EPOCHS"]
        print(f"Preparing for new training run for {end_epoch} epochs...")
        checkpoint_dir = os.path.join(run_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        best_val = float('inf')
        current_lr = config["LR"]
        # track training and validation losses from start
        train_losses = []
        val_losses = []
        lr_changes = [(0, config['LR'])]  # list of (epoch, lr) tuples
        lr_changes_path = os.path.join(run_dir, 'stats', 'lr_changes.ini')
        # Compute input normalization stats from training data
        print("\n=== Computing Input Statistics ===")
        train_mean, train_std = compute_input_stats(val_loader, images_per_dem=config["IMAGES_PER_DEM"])
        print(f"Mean: {round_list(train_mean.tolist(), 10)}")
        print(f"Std: {round_list(train_std.tolist(), 10)}")
        print("="*60)
    else:
        # Resuming from checkpoint
        print("="*60)
        # Get last checkpoint and load lr and epoch info
        checkpoint_dir = os.path.join(run_dir, 'checkpoints')
        last_checkpoint = os.path.join(checkpoint_dir, 'last.pth')
        if not os.path.isfile(last_checkpoint):
            raise FileNotFoundError(f'No checkpoint found at {last_checkpoint}')
        checkpoint = load_checkpoint(last_checkpoint, map_location=device)
        current_lr = checkpoint['optimizer_state_dict']['param_groups'][0]['lr']
        best_val = checkpoint['val_loss']
        start_epoch = checkpoint['epoch'] + 1  # resume from epoch
        end_epoch = config["EPOCHS"] + start_epoch  # extend total epochs

        # Get losses and lr changes from files
        stats_dir = os.path.join(run_dir, 'stats')
        lr_changes_path = os.path.join(stats_dir, 'lr_changes.ini')
        train_losses_path = os.path.join(stats_dir, 'train_losses.ini')
        val_losses_path = os.path.join(stats_dir, 'val_losses.ini')
        if not os.path.isfile(train_losses_path) or not os.path.isfile(val_losses_path) or not os.path.isfile(lr_changes_path):
            raise FileNotFoundError(f'Training or validation losses files not found in {stats_dir}')
        train_losses = read_file_from_ini(train_losses_path, ftype=list)
        val_losses = read_file_from_ini(val_losses_path, ftype=list)
        lr_changes = read_file_from_ini(lr_changes_path, ftype=list)

        print(f"Resuming training. Will train for {config['EPOCHS']} more epochs,")
        print(f"i.e. from epoch {start_epoch} to {end_epoch}...")
        # Load input normalization stats from checkpoint
        train_mean = checkpoint['train_mean']
        train_std = checkpoint['train_std']
        print(f"Loaded training input stats:")
        print(f"Mean: {round_list(train_mean.tolist(),10)}")
        print(f"Std: {round_list(train_std.tolist(), 10)}")
        print(f"Resumed learning rate: {current_lr:.2e}")
        print("Best validation loss so far: {:.4f}".format(best_val))
        print("="*60)


    camera_params = {'image_width': config["IMAGE_W"], 'image_height': config["IMAGE_H"], 'focal_length': config["FOCAL_LENGTH"]}
    hapke_params = config["HAPKE_KWARGS"]

    # if estimate_dynamic_batch_size:
    #     dynamic_batch_size = estimate_dynamic_batch_size(model=model, config=config,
    #                                                     optimizer=optimizer, device=device, scaler=scaler,
    #                                                     start_bs=16, max_trials=10, max_batch_size=256,
    #                                                     use_amp=use_amp, camera_params=camera_params, hapke_params=hapke_params,
    #                                                     w_mse=config["W_MSE"], w_grad=config["W_GRAD"], w_refl=config["W_REFL"])
    #     # adjust train loader batch size if needed
    #     if dynamic_batch_size > config["BATCH_SIZE"]:
    #         print(f"Adjusting training DataLoader batch size from {config['BATCH_SIZE']} to {dynamic_batch_size}")
    #         train_loader = DataLoader(train_ds, batch_size=dynamic_batch_size, shuffle=True, num_workers=config["NUM_WORKERS"], 
    #                                 collate_fn=collate, pin_memory=pin_memory, sampler=None, persistent_workers=True, prefetch_factor=4)


    if new_training is False: # if we are continuing, load model and optimizer state
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Count and display model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("="*60)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Training on device: {device}")
    print(f"Weights: MSE: {config['W_MSE']}, Gradient: {config['W_GRAD']}, Reflectance: {config['W_REFL']:.1e}")
    print("="*60)


    # Prepare folders and paths for saving checkpoints and stats
    stats_dir = os.path.join(run_dir, 'stats')
    os.makedirs(stats_dir, exist_ok=True)
    train_losses_path = os.path.join(stats_dir, 'train_losses.ini')
    val_losses_path = os.path.join(stats_dir, 'val_losses.ini')
    
    # Start background input listener thread
    with input_listener_context(daemon=True) as (command_queue, stop_event, input_thread):
        # Main training loop
        for epoch in range(start_epoch, end_epoch + 1): # inclusive of end_epoch
            #ADD TIME MEASUREMENTS!
            # Check for input commands, and handle them
            updated_end_epoch, early_stop = handle_input_commands(command_queue, optimizer, end_epoch, epoch)
            if updated_end_epoch != end_epoch:
                end_epoch = updated_end_epoch
            if early_stop:
                end_epoch = epoch
                break
            
            # set epoch for distributed sampler, if applicable
            if sampler is not None:
                sampler.set_epoch(epoch)  

            # Training (per-epoch transient progress bar)
            train_loss_val = train_epoch(model, train_loader, optimizer, scaler, device, train_mean, train_std,
                                    current_epoch=epoch, total_epochs=end_epoch, non_blocking=non_blocking,
                                    w_mse=config["W_MSE"], w_grad=config["W_GRAD"], w_refl=config["W_REFL"],
                                    use_amp=use_amp, hapke_params=hapke_params, camera_params=camera_params,
                                    autocast_device_type=autocast_device_type, grad_clip=config["GRAD_CLIP"])

            # Validation (per-epoch transient progress bar)
            val_loss_val = validate_epoch(model, val_loader, device, train_mean, train_std,
                                    current_epoch=epoch, total_epochs=end_epoch, non_blocking=non_blocking,
                                    w_mse=config["W_MSE"], w_grad=config["W_GRAD"], w_refl=config["W_REFL"],
                                    use_amp=use_amp, hapke_params=hapke_params, camera_params=camera_params,
                                    autocast_device_type=autocast_device_type)
            

            # # DEBUG: save batch loss figures - remember to change trainer_core to return batch losses too, and update calls above to unpack them
            # batch_figures_dir = os.path.join(run_dir, 'batch_loss_figures')
            # # save batch loss numbers as ini files
            # train_batch_losses_path = os.path.join(batch_figures_dir, f'epoch_{epoch}_train_batch_losses.ini')
            # val_batch_losses_path = os.path.join(batch_figures_dir, f'epoch_{epoch}_val_batch_losses.ini')
            # save_file_as_ini(batch_losses_train, train_batch_losses_path)
            # save_file_as_ini(batch_losses_val, val_batch_losses_path)
            # os.makedirs(batch_figures_dir, exist_ok=True)
            # # save batch loss figures for this epoch
            # fig, ax = plt.subplots(1,1, figsize=(12,5))
            # ax.plot(batch_losses_train, label='Train Batch Losses')
            # ax.plot(batch_losses_val, label='Validation Batch Losses')
            # # ax.set_yscale('log')
            # ax.set_title(f'Epoch {epoch} Train Batch Losses')
            # ax.set_xlabel('Batch Index')
            # ax.set_ylabel('Loss')
            # ax.legend()
            # fig.savefig(os.path.join(batch_figures_dir, f'epoch_{epoch}_batch_losses.pdf'))


  

            # Single tidy epoch summary line
            tqdm.write(f"Epoch {epoch}/{end_epoch} - Training loss: {train_loss_val:.4f}, Validation loss: {val_loss_val:.4f}")

            train_losses.append(train_loss_val)
            val_losses.append(val_loss_val)
            scheduler.step(val_loss_val)

            current_lr = optimizer.param_groups[0]['lr']
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss_val,
                'val_loss': val_loss_val,
                'train_mean': train_mean,
                'train_std': train_std,
                'learning_rate': current_lr,  # Save current LR for resuming
            }

            is_best = val_loss_val < best_val
            if is_best:
                best_val = val_loss_val
            
            # If LR has changed, log it
            if epoch > 1:
                if current_lr != lr_changes[-1][1]:
                    lr_changes.append((epoch, current_lr))


            # Save checkpoint and losses after each epoch, to avoid data loss if interrupted
            save_last_and_best(checkpoint, checkpoint_dir, is_best=is_best)
            save_file_as_ini(train_losses, train_losses_path)
            save_file_as_ini(val_losses, val_losses_path)
            save_file_as_ini(lr_changes, lr_changes_path)



            # free large locals and accelerator cache
            try:
                del outputs, loss, 
                del images, reflectance_maps, targets, meta
            except NameError:
                pass

            # free python-level garbage and GPU cache to avoid long-term accumulation
            gc.collect()
            try:
                # CUDA: release cached GPU memory
                if device.type == 'cuda' and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # MPS: release any cached memory if API present (PyTorch versions vary)
                elif device.type == 'mps' and hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
                    try:
                        torch.mps.empty_cache()
                    except Exception:
                        # some builds may not implement empty_cache fully
                        pass
                # no-op for CPU or unsupported backends
            except Exception:
                pass

            log_memory(f"epoch_end_{epoch}", device=device, proc=proc, verbose=config['VERBOSE'])



    # After training, set the model to the best checkpoint before returning 
    best_checkpoint = os.path.join(checkpoint_dir, 'best.pth')
    checkpoint = load_checkpoint(best_checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    best_epoch = checkpoint['epoch']
    # return model and test loader for evaluation
    returned_values = (best_epoch, 
                       start_epoch, 
                       end_epoch, 
                       model, 
                       test_loader, 
                       device, 
                       train_mean, 
                       train_std, 
                       camera_params, 
                       hapke_params,
                       use_amp,
                       non_blocking)
    
    return returned_values
