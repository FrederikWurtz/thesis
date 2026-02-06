"""Core trainer components split out from the migrated trainer module.

Provides:
- DEMDataset, FluidDEMDataset
- UNet (+ helper blocks: MetaEncoder, FiLMLayer, DoubleConv)
- compute_reflectance_map_from_dem, calculate_total_loss
- normalize_inputs, compute_input_stats, get_device
"""

import os
import numpy as np
import torch
import h5py
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


from master.data_sim.dataset_io import list_pt_files
from master.lro_data_sim.lro_generator import generate_and_return_lro_data
from master.data_sim.generator import generate_and_return_data_bacteria, generate_and_save_data_pooled_multi_gpu
from master.lro_data_sim.lro_generator_multi_band import generate_and_return_lro_data_multi_band
from master.models.losses import calculate_total_loss
from master.models.unet import UNet
from master.train.train_utils import normalize_inputs
from torch.distributed import init_process_group, destroy_process_group


def load_train_objs(config, run_path: str, epoch_shared=None):
    if config["USE_SEMIFLUID"]:
        if is_main():
            print(f"Semifluid training detected - using {os.path.join(run_path, 'train_temp')} directory for data...")
        train_files = list_pt_files(os.path.join(run_path, 'train_temp'))
        train_set = DEMDataset(train_files, config=config) # load your dataset
    else:
        print("Using standard FluidDEMDataset for training...")
        train_set = FluidDEMDataset(config, epoch_shared=epoch_shared) # load your dataset
    val_path = os.path.join(run_path, 'val') # load validation dataset
    val_files = list_pt_files(val_path)
    val_set = DEMDataset(val_files, config=config)
    test_files = list_pt_files(os.path.join(run_path, 'test'))  # load test dataset
    test_set = DEMDataset(test_files, config=config)
    if config["USE_MULTI_BAND"]:
        out_channels = 3 # DEM, w band and theta_bar band
        model = UNet(in_channels=config["IMAGES_PER_DEM"], 
                     out_channels=out_channels, 
                     features=config["UNET_FEATURES"],
                     meta_dim=config["META_DIM"],
                     meta_hidden=config["META_HIDDEN"],
                     meta_out=config["META_OUT"],
                     norm=config["UNET_NORM"],
                     num_groups=config["UNET_NUM_GROUPS"],
                     w_range=(config["W_MIN"], config["W_MAX"]), 
                     theta_range=(config["THETA_BAR_MIN"], config["THETA_BAR_MAX"]))  # load your model
    else:
        out_channels = 1
        model = UNet(in_channels=config["IMAGES_PER_DEM"], 
                    out_channels=out_channels, 
                    features=config["UNET_FEATURES"],
                    meta_dim=config["META_DIM"],
                    meta_hidden=config["META_HIDDEN"],
                    meta_out=config["META_OUT"],
                    norm=config["UNET_NORM"],
                    num_groups=config["UNET_NUM_GROUPS"])  # load your model
    optimizer = torch.optim.Adam(model.parameters(), lr=config["LR"], weight_decay=config["WEIGHT_DECAY"])
    return train_set, val_set, test_set, model, optimizer

def load_test_objs(config, test_path: str):
    test_set = list_pt_files(test_path)  # load test dataset
    test_set = DEMDataset(test_set, config=config)
    return test_set

def prepare_dataloader(dataset: Dataset, batch_size: int, num_workers: int = 2, prefetch_factor: int = 4, use_shuffle: bool = False, persistent_workers: bool = True, multi_gpu: bool = True) -> DataLoader:
    if multi_gpu:
        rank = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else 0
        return DataLoader(
            dataset,
            batch_size=batch_size,
            pin_memory=True,
            sampler=DistributedSampler(dataset, shuffle=use_shuffle),
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,  # 🔥 Keep workers alive between epochs
            pin_memory_device=f'cuda:{rank}',  # 🔥 Pin directly to target GPU
        )
    else:
        pin_memory = True if torch.cuda.is_available() else False
        return DataLoader(
            dataset,
            batch_size=batch_size,
            pin_memory=pin_memory,
            shuffle=use_shuffle,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,  # 🔥 Keep workers alive between epochs
        )

def set_global_epoch(epoch):
    global GLOBAL_EPOCH
    GLOBAL_EPOCH = epoch


def is_main():
    return int(os.environ["LOCAL_RANK"]) == 0 if "LOCAL_RANK" in os.environ else True

def ddp_setup():
    try:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))
        # 🔥 Enable cuDNN autotuner for convolution optimization
        torch.backends.cudnn.benchmark = True
        print(f"DDP setup complete on GPU {local_rank}")
        # Optional: Enable TF32 for Ampere GPUs (A100, RTX 3090, etc.)
        if torch.cuda.get_device_capability()[0] >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        # 🔥 Use TensorFloat32 for faster matmul on Ampere+ GPUs
        torch.set_float32_matmul_precision('high')  # or 'medium' for even more speed
        
    except KeyError:
        raise RuntimeError("LOCAL_RANK not found in environment variables. Please run this script using torch.distributed.launch or torchrun for multi-GPU training.")

class DEMDataset(Dataset):
    def __init__(self, files, config=None):
        self.files = files
        if config is not None:
            self.config = config

    def __len__(self):
        return len(self.files)
    
    def set_epoch(self, epoch: int):
        pass # placeholder for compatibility with SemiFluidDEMDataset

    def __getitem__(self, idx):
        # Load PyTorch tensors directly
        loaded = torch.load(self.files[idx], map_location='cpu')
        if self.config is not None and self.config["USE_MULTI_BAND"]:
            # Extract tensors using the correct keys from generator.py
            target_tensor = loaded['dem'].unsqueeze(0)  # Add channel dim
            images_tensor = loaded['data']
            reflectance_maps_tensor = loaded['reflectance_maps']
            meta_tensor = loaded['meta']
            w_tensor = loaded['w'].unsqueeze(0)
            theta_bar_tensor = loaded['theta_bar'].unsqueeze(0)
            lro_meta_tensor = loaded['lro_meta']
            return images_tensor, reflectance_maps_tensor, target_tensor, meta_tensor, w_tensor, theta_bar_tensor, lro_meta_tensor  
            
        else:
            # Extract tensors using the correct keys from generator.py
            target_tensor = loaded['dem'].unsqueeze(0)  # Add channel dim
            images_tensor = loaded['data']
            reflectance_maps_tensor = loaded['reflectance_maps']
            meta_tensor = loaded['meta']
            
            return images_tensor, reflectance_maps_tensor, target_tensor, meta_tensor

class FluidDEMDataset(Dataset):
    """Dataset that synthesizes DEMs and corresponding 5-image sets on the fly.

    - __len__ returns n_dems
    - __getitem__ returns (images_tensor, reflectance_maps_tensor, target_tensor, meta_tensor)
    """
    def __init__(self, config=None, epoch_shared=None):
        # NOTE: store simple kwargs/dicts only so the Dataset is picklable by DataLoader workers
        self.config = config
        self.epoch_shared = epoch_shared if epoch_shared is not None else 0
        self.base_seed = config["BASE_SEED"] if "BASE_SEED" in config else 42

    def set_epoch(self, epoch):
        """Set epoch for deterministic data generation."""

        # print(f"FluidDEMDataset: Old epoch: {self.epoch_shared.value}")
        self.epoch_shared.value = epoch
        # print(f"FluidDEMDataset: New epoch set to {self.epoch_shared.value} in shared memory.")
        # self.epoch = epoch
        # print(f"FluidDEMDataset: New epoch {self.epoch}")

    def __len__(self):
        return self.config["FLUID_TRAIN_DEMS"]

    def __getitem__(self, idx):
        # Set seed for reproducibility
        
        current_epoch = self.epoch_shared.value if hasattr(self.epoch_shared, 'value') else self.epoch_shared
        epoch_seed = self.base_seed + current_epoch * len(self) + idx

        # if idx == 0:
        #     print(f"FluidDEMDataset: Generating item idx {idx} for epoch {self.epoch_shared.value}, with seed {epoch_seed}")
        

        torch.manual_seed(epoch_seed)
        np.random.seed(epoch_seed % (2**32 - 1))

        if self.config["USE_LRO_DEMS"]:
            if self.config["USE_MULTI_BAND"]:
                images, reflectance_maps, dem_tensor, metas, w_tensor, theta_bar_tensor, lro_meta = generate_and_return_lro_data_multi_band(config=self.config, device='cpu')

                if is_main() and self.config["DEBUG"]:
                    # DEBUG: check for nans in all outputs
                    if any(torch.isnan(tensor).any() for tensor in images):
                        print("NaN detected in images tensor")
                    if any(torch.isnan(tensor).any() for tensor in reflectance_maps):
                        print("NaN detected in reflectance maps tensor")
                    if torch.isnan(dem_tensor).any():
                        print("NaN detected in dem tensor")
                    if torch.isnan(w_tensor).any():
                        print("NaN detected in w tensor")
                    if torch.isnan(theta_bar_tensor).any():
                        print("NaN detected in theta_bar tensor")

                if not torch.is_tensor(dem_tensor):
                    dem_tensor = torch.from_numpy(dem_tensor)

                return torch.stack(images), torch.stack(reflectance_maps), dem_tensor.unsqueeze(0), torch.tensor(metas, dtype=torch.float32), w_tensor.unsqueeze(0), theta_bar_tensor.unsqueeze(0), torch.tensor(lro_meta, dtype=torch.float32)
            else:
                images, reflectance_maps, dem_tensor, metas = generate_and_return_lro_data(config=self.config, device='cpu')

                if not torch.is_tensor(dem_tensor):
                    dem_tensor = torch.from_numpy(dem_tensor)
    
                return torch.stack(images), torch.stack(reflectance_maps), dem_tensor.unsqueeze(0), torch.tensor(metas, dtype=torch.float32)

        else:
            images, reflectance_maps, dem_np, metas = generate_and_return_data_bacteria(config=self.config)

            images_np = np.stack(images, axis=0)  # (5, H_img, W_img)
            refl_np = np.stack(reflectance_maps, axis=0)  # (5, H_dem, W_dem)
            meta_np = np.array(metas, dtype=np.float32)  # (5,5)

            target_tensor = torch.from_numpy(dem_np).unsqueeze(0)
            images_tensor = torch.from_numpy(images_np)
            reflectance_maps_tensor = torch.from_numpy(refl_np)
            meta_tensor = torch.from_numpy(meta_np)

            return images_tensor, reflectance_maps_tensor, target_tensor, meta_tensor


class SemiFluidDEMDataset(Dataset):
    """
    Dataset der læser pre-genererede samples fra disk.
    Data genereres epoch-vis via _generate_new_data(), som KUN rank 0 må kalde.
    """

    def __init__(self, config, epoch_shared=None):
        
        self.config = config
        self.new_data_every = config["NEW_FLUID_DATA_EVERY"]
        self.base_seed = config["BASE_SEED"]

        # shared epoch state (kan bare være en int til at starte med)
        self.epoch_shared = epoch_shared if epoch_shared is not None else 0

        # hvor vi gemmer midlertidig data
        self.temporary_dir = os.path.join("runs", config["RUN_DIR"], "train_temp")
        # first run of data has been done in initialize.py            
        self.files = []    
        self._refresh_file_list()

    def _refresh_file_list(self):
        """Opdatér self.files ud fra directory-indholdet."""
        self.files = sorted(
            os.path.join(self.temporary_dir, f)
            for f in os.listdir(self.temporary_dir)
            if os.path.isfile(os.path.join(self.temporary_dir, f))
        )
        if is_main():
            print(f"SemiFluidDEMDataset: Loaded {len(self.files)} files from {self.temporary_dir}")
        
        # opdatér fil-liste efter data er genereret

    def set_epoch(self, epoch: int):
        """
        Kaldes fra traineren hver epoch.
        Rank 0 genererer evt. nyt data, de andre venter.
        """
        # opdatér lokal state
        self.epoch_shared = epoch

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Load PyTorch tensors directly
        loaded = torch.load(self.files[idx], map_location='cpu')
        if self.config is not None and self.config["USE_MULTI_BAND"]:
            # Extract tensors using the correct keys from generator.py
            target_tensor = loaded['dem'].unsqueeze(0)  # Add channel dim
            images_tensor = loaded['data']
            reflectance_maps_tensor = loaded['reflectance_maps']
            meta_tensor = loaded['meta']
            w_tensor = loaded['w'].unsqueeze(0)
            theta_bar_tensor = loaded['theta_bar'].unsqueeze(0)
            lro_meta_tensor = loaded['lro_meta']
            return images_tensor, reflectance_maps_tensor, target_tensor, meta_tensor, w_tensor, theta_bar_tensor, lro_meta_tensor
        else:
            # Extract tensors using the correct keys from generator.py
            target_tensor = loaded['dem'].unsqueeze(0)  # Add channel dim
            images_tensor = loaded['data']
            reflectance_maps_tensor = loaded['reflectance_maps']
            meta_tensor = loaded['meta']
            
            return images_tensor, reflectance_maps_tensor, target_tensor, meta_tensor















# class DEMDatasetHDF5(Dataset):
#     def __init__(self, hdf5_path):
#         self.hdf5_path = hdf5_path
#         # Åbn kort for at hente længde
#         with h5py.File(self.hdf5_path, 'r') as f:
#             self.length = f['images'].shape[0]

#         # Filhåndtag åbnes senere pr. worker
#         self.file = None
#         self.images = None
#         self.reflectance_maps = None
#         self.dems = None
#         self.metas = None

#     def __del__(self):
#         if self.file is not None:
#             try:
#                 self.file.close()
#             except:
#                 pass

#     def __len__(self):
#         return self.length

#     def __getitem__(self, idx):
#         # Sørg for at filen er åbnet (sker i worker_init_fn)
#         if self.file is None:
#             raise RuntimeError("HDF5 file not opened. Did you set worker_init_fn?")
        
#         images_tensor = torch.from_numpy(self.images[idx]).float()
#         reflectance_maps_tensor = torch.from_numpy(self.reflectance_maps[idx]).float()
#         target_tensor = torch.from_numpy(self.dems[idx]).unsqueeze(0).float()
#         meta_tensor = torch.from_numpy(self.metas[idx]).float()
#         return images_tensor, reflectance_maps_tensor, target_tensor, meta_tensor


# class SemifluidDEMDataset(Dataset):
#     def __init__(self, config=None, temporary_dir=None, reuse_limit=10):
#         self.config = config
#         self.reuse_limit = reuse_limit
#         self.cache = {}  # idx -> reuse_count
#         self.temp_dir = temporary_dir

#     def __len__(self):
#         return int(self.config["FLUID_TRAIN_DEMS"])

#     def _generate_and_save(self, idx):
#         # Generer syntetisk data
#         images, reflectance_maps, dem_np, metas = generate_and_return_data(config=self.config)
#         images_np = np.stack(images, axis=0)
#         refl_np = np.stack(reflectance_maps, axis=0)
#         meta_np = np.array(metas, dtype=np.float32)

#         # Gem til disk
#         path = os.path.join(self.temp_dir, f"dataset_{idx}.npz")
#         np.savez(path, images=images_np, refl=refl_np, dem=dem_np, meta=meta_np)

#     def _load_from_disk(self, idx):
#         path = os.path.join(self.temp_dir, f"dataset_{idx}.npz")
#         arr = np.load(path)
#         # Konverter til tensors
#         images_tensor = torch.from_numpy(arr["images"])
#         reflectance_maps_tensor = torch.from_numpy(arr["refl"])
#         target_tensor = torch.from_numpy(arr["dem"]).unsqueeze(0)
#         meta_tensor = torch.from_numpy(arr["meta"])
#         return images_tensor, reflectance_maps_tensor, target_tensor, meta_tensor

#     def __getitem__(self, idx):
#         # Hvis vi har genereret før og reuse_count < limit → load fra disk
#         if idx in self.cache and self.cache[idx] < self.reuse_limit:
#             self.cache[idx] += 1
#             return self._load_from_disk(idx)

#         # Ellers generer nyt og nulstil tæller
#         self._generate_and_save(idx)
#         self.cache[idx] = 1
#         return self._load_from_disk(idx)




# def train_epoch(model, train_loader, optimizer, scaler, device, train_mean, train_std, current_epoch=None, total_epochs=None, non_blocking=None,
#                 w_mse=None, w_grad=None, w_refl=None, use_amp=None, hapke_params=None, camera_params=None, autocast_device_type=None, grad_clip=None):
#     """Train for one epoch with reflectance map loss"""
#     model.train()
#     running_loss = 0.0
#     total_images = 0

#     train_pbar = tqdm(train_loader, desc=f"Training on epoch {current_epoch}/{total_epochs}", leave=False, position=0, dynamic_ncols=True)

#     #DEBUG: track batch losses
#     # batch_losses_train = []

#     for batch_idx, (images, reflectance_maps, targets, meta) in enumerate(train_pbar):
#         # move everything to device - costly but necessary
#         images = images.to(device, non_blocking=non_blocking)
#         reflectance_maps = reflectance_maps.to(device, non_blocking=non_blocking)
#         targets = targets.to(device, non_blocking=non_blocking)
#         meta = meta.to(device, non_blocking=non_blocking)

#         # print(f"Pre-normalization: mean={images.mean().item():.4f}, std={images.std().item():.4f}")
#         # print(f"Images min value: {images.min().item():.4f}, max value: {images.max().item():.4f}")

#         # Normalize in-place (inputs are on device)
#         images = normalize_inputs(images, train_mean, train_std)

#         # #check that normalization worked
#         # print(f"Post-normalization: mean={images.mean().item():.4f}, std={images.std().item():.4f}")

#         optimizer.zero_grad()
        
#         with torch.amp.autocast(device_type=autocast_device_type, enabled=use_amp):
#             # forward pass
#             outputs = model(images, meta, target_size=targets.shape[-2:])
#             # compute loss (calculate_total_loss should accept device tensors)
#             loss = calculate_total_loss(outputs, targets, reflectance_maps, meta, device=device,
#                                         camera_params=camera_params, hapke_params=hapke_params,
#                                         w_grad=w_grad, w_refl=w_refl, w_mse=w_mse)
        
#         # backward pass and step optimizer via scaler
#         scaler.scale(loss).backward()

#         if grad_clip is not None and grad_clip > 0:
#             scaler.unscale_(optimizer)
#             torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

#         scaler.step(optimizer)
#         scaler.update()

#         # # DEBUG: Print loss components for first batch
#         # if batch_idx == 0:
#         #     if hasattr(loss, 'loss_components'):
#         #         comp = loss.loss_components
#         #         print(f"[TRAIN] MSE: {comp['mse']:.3f} (w={w_mse}), Grad: {comp['grad']:.3f} (w={w_grad}), Refl: {comp['refl']:.3f} (w={w_refl})")
#         #         total = w_mse * comp['mse'] + w_grad * comp['grad'] + w_refl * comp['refl']
#         #         print(f"[TRAIN] Weighted contributions: MSE={w_mse * comp['mse']:.3f}, Grad={w_grad * comp['grad']:.3f}, Refl={w_refl * comp['refl']:.3f}, Total={total:.3f}")

#         # DEBUG: track batch losses
#         # batch_losses_train.append(loss.item())

#         # accumulate running loss
#         bsz = images.size(0) # batch size
#         running_loss += loss.item() * bsz # accumulate loss times batch size
#         total_images += bsz
#         train_pbar.set_postfix(loss=f"{(running_loss / total_images):.4f}")

#     train_pbar.close()
#     # return running_loss / total_images, batch_losses_train
#     return running_loss / total_images


# @torch.no_grad()
# def validate_epoch(model, val_loader, device, train_mean, train_std, current_epoch=None, total_epochs=None, non_blocking=None,
#              w_mse=None, w_grad=None, w_refl=None, use_amp=None, hapke_params=None, camera_params=None, autocast_device_type=None):
#     """Validate the model with reflectance map loss"""
#     model.eval()
#     running_loss = 0.0
#     total_images = 0

#     val_pbar = tqdm(val_loader, desc=f"Validating on epoch {current_epoch}/{total_epochs}", leave=False, position=0, dynamic_ncols=True)

#     #DEBUG: track batch losses
#     # batch_losses_val = []

#     for batch_idx, (images, reflectance_maps, targets, meta) in enumerate(val_pbar):
#         # move everything to device - costly but necessary
#         images = images.to(device, non_blocking=non_blocking)
#         reflectance_maps = reflectance_maps.to(device, non_blocking=non_blocking)
#         targets = targets.to(device, non_blocking=non_blocking)
#         meta = meta.to(device, non_blocking=non_blocking)

#         # Normalize in-place (inputs are on device)
#         images = normalize_inputs(images, train_mean, train_std)
        
#         with torch.amp.autocast(device_type=autocast_device_type, enabled=use_amp):
#             # Pass target size to model for proper upsampling
#             outputs = model(images, meta, target_size=targets.shape[-2:])
#             loss = calculate_total_loss(outputs, targets, reflectance_maps, meta, device=device,
#                                         camera_params=camera_params, hapke_params=hapke_params,
#                                         w_grad=w_grad, w_refl=w_refl, w_mse=w_mse)
        
#         # # DEBUG: Print loss components for first batch
#         # if batch_idx == 0:
#         #     if hasattr(loss, 'loss_components'):
#         #         comp = loss.loss_components
#         #         print(f"[TRAIN] MSE: {comp['mse']:.3f} (w={w_mse}), Grad: {comp['grad']:.3f} (w={w_grad}), Refl: {comp['refl']:.3f} (w={w_refl})")
#         #         total = w_mse * comp['mse'] + w_grad * comp['grad'] + w_refl * comp['refl']
#         #         print(f"[TRAIN] Weighted contributions: MSE={w_mse * comp['mse']:.3f}, Grad={w_grad * comp['grad']:.3f}, Refl={w_refl * comp['refl']:.3f}, Total={total:.3f}")

#         # DEBUG: track batch losses
#         # batch_losses_val.append(loss.item())

#         # calculate and accumulate running loss
#         bsz = images.size(0)
#         running_loss += loss.item() * bsz
#         total_images += bsz
#         val_pbar.set_postfix(loss=f"{(running_loss / total_images):.4f}")
#     val_pbar.close()
#     # return running_loss / total_images, batch_losses_val
#     return running_loss / total_images

# @torch.no_grad()
# def evaluate_on_test_files(model=None, test_loader=None, device=None, train_mean=None, train_std=None, non_blocking=None,
#                            w_mse=None, w_grad=None, w_refl=None, use_amp=None, hapke_params=None, camera_params=None):
    
#     if any(v is None for v in [model, test_loader, device, train_mean, train_std, camera_params, hapke_params, w_mse, w_grad, w_refl, use_amp]):
#         raise ValueError("All arguments must be provided to evaluate_on_test_files.")
#     """Evaluate the model on test files with reflectance map loss"""
#     model.eval()
#     running_loss = 0.0
#     total_images = 0
#     abs_sum = 0.0
#     n_pixels = 0

#     test_pbar = tqdm(test_loader, desc=f"Evaluating on test files", leave=False, position=0, dynamic_ncols=True)

#     for images, reflectance_maps, targets, meta in test_pbar:
#         # move everything to device - costly but necessary
#         images = images.to(device, non_blocking=non_blocking)
#         reflectance_maps = reflectance_maps.to(device, non_blocking=non_blocking)
#         targets = targets.to(device, non_blocking=non_blocking)
#         meta = meta.to(device, non_blocking=non_blocking)

#         # Normalize in-place (inputs are on device)
#         images = normalize_inputs(images, train_mean, train_std)
        
#         with torch.amp.autocast(device_type=device.type, enabled=use_amp):
#             # Pass target size to model for proper upsampling
#             outputs = model(images, meta, target_size=targets.shape[-2:])
#             loss = calculate_total_loss(outputs, targets, reflectance_maps, meta, device=device,
#                                         camera_params=camera_params, hapke_params=hapke_params,
#                                         w_grad=w_grad, w_refl=w_refl, w_mse=w_mse)

#         # Compute Absolute Mean Error (AME)
#         abs_diff = torch.abs(outputs - targets)
#         abs_sum += abs_diff.sum().item()
#         n_pixels += abs_diff.numel()

#         # calculate running loss
#         bsz = images.size(0)
#         running_loss += loss.item() * bsz
#         total_images += bsz
#         test_pbar.set_postfix(loss=f"{(running_loss / total_images):.4f}")

#     test_pbar.close()

#     test_ame = abs_sum / n_pixels
#     test_loss = running_loss / total_images
#     return test_loss, test_ame


# def estimate_dynamic_batch_size(model, config=None,
#                                 optimizer=None, device=None, scaler=None,
#                                 start_bs=16, max_trials=10, max_batch_size=256,
#                                 use_amp=False, camera_params=None, hapke_params=None,
#                                 w_mse=None, w_grad=None, w_refl=None):
#     """
#     Estimer maksimal batchstørrelse uden OOM ved at generere syntetiske data.
#     input_shape: (C, H, W) for billeder
#     meta_shape: (M,) for meta-data (valgfri)
#     refl_shape: (R, H, W) for reflektanskort (valgfri)
#     """
#     print("Estimating maximum batch size with synthetic data...")
#     input_shape = (config["IMAGES_PER_DEM"], config["IMAGE_H"], config["IMAGE_W"])
#     # Meta per-image vector size is typically 5 (sun/camera params). If a
#     # different value exists in config, use it; otherwise default to 5.
#     meta_dim = config.get("META_DIM", 5) if config is not None else 5
#     meta_shape = (config["IMAGES_PER_DEM"], meta_dim)
#     refl_shape = (config["IMAGES_PER_DEM"], config["DEM_SIZE"], config["DEM_SIZE"])

#     model.train()

#     test_bs = start_bs
#     max_bs = 0
#     trials = 0

#     while trials < max_trials:
#         trials += 1
#         try:
#             # Generér syntetiske batches
#             images = torch.randn((test_bs,) + input_shape, device=device)
#             # Targets (DEMs) live at DEM resolution (config['DEM_SIZE']).
#             targets = torch.randn((test_bs, 1, config["DEM_SIZE"], config["DEM_SIZE"]), device=device)
#             refls = torch.randn((test_bs,) + refl_shape, device=device) if refl_shape else None
#             meta = torch.randn((test_bs,) + meta_shape, device=device) if meta_shape else None

#             optimizer_zero = optimizer.zero_grad if optimizer is not None else lambda *a, **k: None
#             optimizer_zero(set_to_none=True)

#             with torch.amp.autocast(device_type=device.type if device.type in ['cuda', 'mps'] else 'cpu', enabled=use_amp):
#                 outputs = model(images, meta) if meta is not None else model(images)

#                 # Brug din loss-funktion
#                 loss = calculate_total_loss(outputs, targets, refls, meta, device=device,
#                                             camera_params=camera_params, hapke_params=hapke_params,
#                                             w_grad=w_grad, w_refl=w_refl, w_mse=w_mse)

#             if scaler and scaler.is_enabled():
#                 scaler.scale(loss).backward()
#                 scaler.update()
#             else:
#                 loss.backward()

#             max_bs = test_bs
#             del outputs, loss, images, targets, refls, meta
#             torch.cuda.empty_cache()
#             if device.type == 'cuda':
#                 torch.cuda.synchronize()
#             elif device.type == 'mps' and hasattr(torch.mps, 'synchronize'):
#                 torch.mps.synchronize()

#             test_bs *= 2  # Dobbel batchstørrelsen
#         except RuntimeError as e:
#             if 'out of memory' in str(e).lower():
#                 print(f"OOM at batch size {test_bs}, last successful: {max_bs}")
#                 if device.type == 'cuda':
#                     torch.cuda.empty_cache()
#                 elif device.type == 'mps' and hasattr(torch.mps, 'empty_cache'):
#                     torch.mps.empty_cache()
#                 break
#             else:
#                 raise

#     if max_bs == 0:
#         max_bs = start_bs

#     print(f"Max batch size: {max_bs}")
#     if max_bs > max_batch_size:
#         print(f"Limiting max batch size to {max_batch_size}")
#         max_bs = max_batch_size
#     return max_bs


__all__ = ['DEMDataset', 
           'FluidDEMDataset',
           'DEMDatasetHDF5', 
           'calculate_total_loss', 
           'train_epoch', 
           'validate_epoch', 
           'evaluate_on_test_files',
           'estimate_dynamic_batch_size']
