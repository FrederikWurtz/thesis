import os
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import gc
import torch.distributed as dist

from master.train.trainer_core import DEMDataset, FluidDEMDataset, DEMDatasetHDF5, SemifluidDEMDataset
from master.models.unet import UNet
from master.train.trainer_core import train_epoch, validate_epoch
from master.train.train_utils import compute_input_stats, round_list, collate, convert_npz_to_hdf5, worker_init_fn
from master.train.checkpoints import read_file_from_ini, save_last_and_best, load_checkpoint, save_file_as_ini
from master.data_sim import dataset_io
from master.utils.interactivity_utils import input_listener_context, handle_input_commands


def is_main_process():
    return dist.get_rank() == 0

def launch_multi_gpu_training(config=None, run_dir=None, val_dir=None, test_dir=None, new_training=None, world_size=None, rank=None):
    '''Prepare and launch multi-GPU training by setting up datasets, model, and training parameters.'''

    if is_main_process():
        print("Preparing for multi-GPU training...")
        if config is None:
            raise ValueError("config must be provided")
        if run_dir is None:
            raise ValueError("run_dir must be provided")
        if val_dir is None:
            raise ValueError("val_dir must be provided")
        if test_dir is None:
            raise ValueError("test_dir must be provided")

    # Distributed Data Parallel setup
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    print(f"Wrapping model in DDP on rank {rank}/{world_size}, local rank {local_rank}...")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # setup for CUDA training - mixed precision, pin memory, non-blocking transfers
    use_amp = True 
    pin_memory = True
    non_blocking = True
    autocast_device_type = 'cuda'
    scaler = torch.amp.GradScaler(enabled=use_amp)
    if is_main_process():
        print("\nUsing mixed precision training (FP16)")
        print("pin_memory set to True for CUDA")
        print("non_blocking transfers enabled for CUDA")

    # prepare datasets and dataloaders
    val_files = dataset_io.list_npz_files(val_dir)
    test_files = dataset_io.list_npz_files(test_dir)

    if is_main_process():
        if len(val_files) != config["FLUID_VAL_DEMS"]:
            raise RuntimeError(f'Number of validation DEMs found ({len(val_files)}) does not match config setting FLUID_VAL_DEMS ({config["FLUID_VAL_DEMS"]})')
        # also check test if test files exist
        if len(test_files) != config["FLUID_TEST_DEMS"]:
            raise RuntimeError(f'Number of test DEMs found ({len(test_files)}) does not match config setting FLUID_TEST_DEMS ({config["FLUID_TEST_DEMS"]})')


    # setup datasets and dataloaders
    if config["USE_SEMIFLUID"] is True: # use semi-fluid dataset for training dataset
        temporary_dir = os.path.join(run_dir, 'semifluid_temp')
        if is_main_process():
            print(f"Using Semi-Fluid DEM Dataset with reuse limit of {config['REUSE_LIMIT']} for training dataset")
            os.makedirs(temporary_dir, exist_ok=True)
        dist.barrier()  # ensure all processes wait for temp dir creation
        train_ds = SemifluidDEMDataset(config=config, temporary_dir=temporary_dir, reuse_limit=config["REUSE_LIMIT"])
    else:
        if is_main_process():
            print("Using Fluid DEM Dataset for training dataset")
        train_ds = FluidDEMDataset(config)

    if config["USE_HDF5"] is True:
        if is_main_process():
            print("Using HDF5 datasets for validation and test datasets")
        # check if HDF5 files already exist; if not, convert from NPZ
        val_dataset_path = os.path.join(val_dir, 'validation_data.h5')
        test_dataset_path = os.path.join(test_dir, 'test_data.h5')
        if is_main_process():
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
        dist.barrier()  # ensure only main process converts files
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

    # initialize model
    model = UNet(in_channels=config["IMAGES_PER_DEM"], out_channels=1)
    model.to(device)

    # wrap model in DDP
    model = DDP(model, device_ids=[local_rank] if device.type == 'cuda' else None)
    sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank)

    # initialize optimizer, scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=current_lr, weight_decay=config["WEIGHT_DECAY"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=config["LR_PATIENCE"])

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=config["BATCH_SIZE"], shuffle=False, num_workers=config["NUM_WORKERS"], 
                            collate_fn=collate, pin_memory=pin_memory, sampler=sampler, persistent_workers=True, prefetch_factor=4)

    if config["USE_HDF5"] is True:
        val_loader = DataLoader(val_ds, batch_size=config["BATCH_SIZE"], shuffle=False, num_workers=config["NUM_WORKERS"], 
                                collate_fn=collate, pin_memory=pin_memory, sampler=None, persistent_workers=True, prefetch_factor=4,
                                worker_init_fn=worker_init_fn)
        test_loader = DataLoader(test_ds, batch_size=config["BATCH_SIZE"], shuffle=False, num_workers=config["NUM_WORKERS"], 
                                collate_fn=collate, pin_memory=pin_memory, sampler=None, persistent_workers=True, prefetch_factor=4,
                                worker_init_fn=worker_init_fn)
        
    else:
        val_loader = DataLoader(val_ds, batch_size=config["BATCH_SIZE"], shuffle=False, num_workers=config["NUM_WORKERS"], 
                                collate_fn=collate, pin_memory=pin_memory, sampler=None, persistent_workers=True, prefetch_factor=4)
        test_loader = DataLoader(test_ds, batch_size=config["BATCH_SIZE"], shuffle=False, num_workers=config["NUM_WORKERS"], 
                                collate_fn=collate, pin_memory=pin_memory, sampler=None, persistent_workers=True, prefetch_factor=4)

    # branching logic: are we starting new training or resuming?
    if new_training:
        if is_main_process():
            start_epoch = 1
            end_epoch = config["EPOCHS"]
            checkpoint_dir = os.path.join(run_dir, 'checkpoints')
            os.makedirs(checkpoint_dir, exist_ok=True)
            best_val = float('inf')
            current_lr = config["LR"]
            # track training and validation losses from start
            train_losses = []
            val_losses = []
            lr_changes = [(0, config['LR'])]  # list of (epoch, lr) tuples
            lr_changes_path = os.path.join(run_dir, 'stats', 'lr_changes.ini')
            # Compute input normalization stats from training data - only rank 0 does this
            train_mean, train_std = compute_input_stats(val_loader, images_per_dem=config["IMAGES_PER_DEM"])
            train_mean_tensor = torch.tensor(train_mean, device=device)
            train_std_tensor = torch.tensor(train_std, device=device)
            print("="*60)
            print(f"Preparing for new training run for {end_epoch} epochs...")
            print("\n=== Computing Input Statistics ===")
            print(f"Mean: {round_list(train_mean.tolist(), 10)}")
            print(f"Std: {round_list(train_std.tolist(), 10)}")
            print("="*60)
        else:
            # Initialize for non-main processes
            start_epoch = 1
            end_epoch = config["EPOCHS"]
            checkpoint_dir = os.path.join(run_dir, 'checkpoints')
            best_val = float('inf')
            train_losses = []
            val_losses = []
            lr_changes = [(0, config['LR'])]
            train_mean_tensor = torch.zeros((config["IMAGES_PER_DEM"],), device=device)
            train_std_tensor = torch.zeros((config["IMAGES_PER_DEM"],), device=device)

        dist.barrier()  # ensure all processes wait for stats computation

        # broadcast start and end epoch, train mean and std to all processes
        epoch_tensor = torch.tensor([start_epoch, end_epoch], dtype=torch.long, device=device)
        dist.broadcast(epoch_tensor, src=0)
        if not is_main_process():
            start_epoch = epoch_tensor[0].item()
            end_epoch = epoch_tensor[1].item()
        dist.broadcast(train_mean_tensor, src=0)
        dist.broadcast(train_std_tensor, src=0)
        train_mean = train_mean_tensor.cpu().numpy()
        train_std = train_std_tensor.cpu().numpy()

    else:
        # Resuming from checkpoint
        # Get last checkpoint and load lr and epoch info - only rank 0 does this
        if is_main_process():
            checkpoint_dir = os.path.join(run_dir, 'checkpoints')
            last_checkpoint = os.path.join(checkpoint_dir, 'last.pth')
            if not os.path.isfile(last_checkpoint):
                raise FileNotFoundError(f'No checkpoint found at {last_checkpoint}')
            checkpoint = load_checkpoint(last_checkpoint, map_location=device)
            current_lr = checkpoint['optimizer_state_dict']['param_groups'][0]['lr']
            best_val = checkpoint['val_loss']
            start_epoch = checkpoint['epoch'] + 1  # resume from epoch
            end_epoch = config["EPOCHS"] + start_epoch  # extend total epochs
            # Load input normalization stats from checkpoint
            train_mean = checkpoint['train_mean']
            train_std = checkpoint['train_std']
            train_mean_tensor = torch.tensor(train_mean, device=device)
            train_std_tensor = torch.tensor(train_std, device=device)
        else:
            # Resuming from checkpoint - non-main processes
            checkpoint_dir = os.path.join(run_dir, 'checkpoints')
            start_epoch = None  # Will be broadcast
            end_epoch = None  # Will be broadcast
            train_mean_tensor = torch.zeros((config["IMAGES_PER_DEM"],), device=device)
            train_std_tensor = torch.zeros((config["IMAGES_PER_DEM"],), device=device)
        dist.barrier()  # ensure all processes wait for checkpoint loading
        # broadcast start and end epoch, train mean and std to all processes
        epoch_tensor = torch.tensor([start_epoch, end_epoch], dtype=torch.long, device=device)
        dist.broadcast(epoch_tensor, src=0)
        dist.broadcast(train_mean_tensor, src=0)
        dist.broadcast(train_std_tensor, src=0)
        if not is_main_process():
            start_epoch = epoch_tensor[0].item()
            end_epoch = epoch_tensor[1].item()
            train_mean = train_mean_tensor.cpu().numpy()
            train_std = train_std_tensor.cpu().numpy()

        if is_main_process():
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

            print("="*60)
            print(f"Resuming training. Will train for {config['EPOCHS']} more epochs,")
            print(f"i.e. from epoch {start_epoch} to {end_epoch}...")
            print(f"Loaded training input stats:")
            print(f"Mean: {round_list(train_mean.tolist(),10)}")
            print(f"Std: {round_list(train_std.tolist(), 10)}")
            print(f"Resumed learning rate: {current_lr:.2e}")
            print("Best validation loss so far: {:.4f}".format(best_val))
            print("="*60)

    camera_params = {'image_width': config["IMAGE_W"], 'image_height': config["IMAGE_H"], 'focal_length': config["FOCAL_LENGTH"]}
    hapke_params = config["HAPKE_KWARGS"]

    if new_training is False: # if we are continuing, load model and optimizer state
        if is_main_process():
            # first get rank 0 to load states
            model_state = checkpoint['model_state_dict']
            optimizer_state = checkpoint['optimizer_state_dict']
        else:
            model_state = None
            optimizer_state = None
    
        # Broadcast state dicts
        object_list = [model_state, optimizer_state]
        dist.broadcast_object_list(object_list, src=0)
        
        # All ranks load state
        model.module.load_state_dict(object_list[0])
        optimizer.load_state_dict(object_list[1])

    if is_main_process():
        # Count and display model parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("="*60)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Training on device: {device}")
        print(f"Weights: MSE: {config['W_MSE']}, Gradient: {config['W_GRAD']}, Reflectance: {config['W_REFL']:.1e}")
        print("="*60)

    if is_main_process():
        # Prepare folders and paths for saving checkpoints and stats
        stats_dir = os.path.join(run_dir, 'stats')
        os.makedirs(stats_dir, exist_ok=True)
        train_losses_path = os.path.join(stats_dir, 'train_losses.ini')
        val_losses_path = os.path.join(stats_dir, 'val_losses.ini')

    
    # Start input listener kun på rank 0
    command_queue = None
    stop_event = None
    listener_ctx = None
    if is_main_process():
        print("Starting input listener thread on main process...")
        listener_ctx = input_listener_context(daemon=True)
        command_queue, stop_event, input_thread = listener_ctx.__enter__()
    else:
        command_queue = None
        stop_event = None

    # Synkroniser alle ranks før træning
    dist.barrier()

    early_stop = False

    for epoch in range(start_epoch, end_epoch + 1):  # inclusive of end_epoch
        # Rank 0 håndterer input og sender signal til andre ranks
        if is_main_process():
            updated_end_epoch, early_stop = handle_input_commands(command_queue, optimizer, end_epoch, epoch)
            if updated_end_epoch != end_epoch:
                end_epoch = updated_end_epoch
            if early_stop:
                end_epoch = epoch

        # Broadcast early_stop og end_epoch til alle ranks
        control_tensor = torch.tensor([int(early_stop), end_epoch], device=device)
        dist.broadcast(control_tensor, src=0)
        early_stop = bool(control_tensor[0].item())
        end_epoch = control_tensor[1].item()

        if early_stop:
            break

        # Set epoch for distributed sampler
        sampler.set_epoch(epoch)

        # Training step
        train_loss_val = train_epoch(
            model, train_loader, optimizer, scaler, device, train_mean, train_std,
            current_epoch=epoch, total_epochs=end_epoch, non_blocking=non_blocking,
            w_mse=config["W_MSE"], w_grad=config["W_GRAD"], w_refl=config["W_REFL"],
            use_amp=use_amp, hapke_params=hapke_params, camera_params=camera_params,
            autocast_device_type=autocast_device_type, grad_clip=config["GRAD_CLIP"]
        )

        # Validation kun på rank 0
        if is_main_process():
            val_loss_val = validate_epoch(
                model, val_loader, device, train_mean, train_std,
                current_epoch=epoch, total_epochs=end_epoch, non_blocking=non_blocking,
                w_mse=config["W_MSE"], w_grad=config["W_GRAD"], w_refl=config["W_REFL"],
                use_amp=use_amp, hapke_params=hapke_params, camera_params=camera_params,
                autocast_device_type=autocast_device_type
            )
            scheduler.step(val_loss_val)
        else:
            val_loss_val = 0.0  # placeholder

        # Broadcast val_loss og LR til alle ranks
        val_loss_tensor = torch.tensor(val_loss_val, device=device)
        dist.broadcast(val_loss_tensor, src=0)
        val_loss_val = val_loss_tensor.item()

        current_lr = torch.tensor(optimizer.param_groups[0]['lr'], device=device)
        dist.broadcast(current_lr, src=0)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr.item()

        # Logging og checkpoint kun på rank 0
        if is_main_process():
            tqdm.write(f"Epoch {epoch}/{end_epoch} - Training loss: {train_loss_val:.4f}, Validation loss: {val_loss_val:.4f}")
            train_losses.append(train_loss_val)
            val_losses.append(val_loss_val)

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss_val,
                'val_loss': val_loss_val,
                'train_mean': train_mean,
                'train_std': train_std,
                'learning_rate': current_lr.item(),
            }
            is_best = val_loss_val < best_val
            if is_best:
                best_val = val_loss_val

            if epoch > 1 and current_lr.item() != lr_changes[-1][1]:
                lr_changes.append((epoch, current_lr.item()))

            save_last_and_best(checkpoint, checkpoint_dir, is_best=is_best)
            save_file_as_ini(train_losses, train_losses_path)
            save_file_as_ini(val_losses, val_losses_path)
            save_file_as_ini(lr_changes, lr_changes_path)

        # Memory cleanup
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        elif device.type == 'mps' and hasattr(torch.mps, 'empty_cache'):
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

    # Luk input listener på rank 0
    if is_main_process() and listener_ctx:
        listener_ctx.__exit__(None, None, None)


    # Først: Rank 0 loader best checkpoint
    if is_main_process():
        best_checkpoint = os.path.join(checkpoint_dir, 'best.pth')
        checkpoint = load_checkpoint(best_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        best_epoch = checkpoint['epoch']
    else:
        best_epoch = 0

    # Broadcast best_epoch
    best_epoch_tensor = torch.tensor(best_epoch, dtype=torch.long, device=device)
    dist.broadcast(best_epoch_tensor, src=0)
    best_epoch = best_epoch_tensor.item()

    # Synkroniser alle ranks
    dist.barrier()

    # All ranks load from best checkpoint
    if is_main_process():
        model_state = model.module.state_dict()
    else:
        model_state = None

    object_list = [model_state]
    dist.broadcast_object_list(object_list, src=0)
    model.module.load_state_dict(object_list[0])

    # Nu har alle ranks den bedste model
    returned_values = (
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
    )

    return returned_values
