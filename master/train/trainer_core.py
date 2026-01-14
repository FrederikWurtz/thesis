"""Core trainer components split out from the migrated trainer module.

Provides:
- DEMDataset, FluidDEMDataset
- UNet (+ helper blocks: MetaEncoder, FiLMLayer, DoubleConv)
- compute_reflectance_map_from_dem, calculate_total_loss
- normalize_inputs, compute_input_stats, get_device
"""

import numpy as np
import torch
import h5py
from torch.utils.data import Dataset
from tqdm import tqdm
import tempfile
import os

from master.lro_data_sim.lro_generator import generate_and_return_lro_data
from master.data_sim.generator import generate_and_return_data_CPU
from master.models.losses import calculate_total_loss
from master.train.train_utils import normalize_inputs


class DEMDatasetHDF5(Dataset):
    def __init__(self, hdf5_path):
        self.hdf5_path = hdf5_path
        # Åbn kort for at hente længde
        with h5py.File(self.hdf5_path, 'r') as f:
            self.length = f['images'].shape[0]

        # Filhåndtag åbnes senere pr. worker
        self.file = None
        self.images = None
        self.reflectance_maps = None
        self.dems = None
        self.metas = None

    def __del__(self):
        if self.file is not None:
            try:
                self.file.close()
            except:
                pass

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Sørg for at filen er åbnet (sker i worker_init_fn)
        if self.file is None:
            raise RuntimeError("HDF5 file not opened. Did you set worker_init_fn?")
        
        images_tensor = torch.from_numpy(self.images[idx]).float()
        reflectance_maps_tensor = torch.from_numpy(self.reflectance_maps[idx]).float()
        target_tensor = torch.from_numpy(self.dems[idx]).unsqueeze(0).float()
        meta_tensor = torch.from_numpy(self.metas[idx]).float()
        return images_tensor, reflectance_maps_tensor, target_tensor, meta_tensor


class SemifluidDEMDataset(Dataset):
    def __init__(self, config=None, temporary_dir=None, reuse_limit=10):
        self.config = config
        self.reuse_limit = reuse_limit
        self.cache = {}  # idx -> reuse_count
        self.temp_dir = temporary_dir

    def __len__(self):
        return int(self.config["FLUID_TRAIN_DEMS"])

    def _generate_and_save(self, idx):
        # Generer syntetisk data
        images, reflectance_maps, dem_np, metas = generate_and_return_data(config=self.config)
        images_np = np.stack(images, axis=0)
        refl_np = np.stack(reflectance_maps, axis=0)
        meta_np = np.array(metas, dtype=np.float32)

        # Gem til disk
        path = os.path.join(self.temp_dir, f"dataset_{idx}.npz")
        np.savez(path, images=images_np, refl=refl_np, dem=dem_np, meta=meta_np)

    def _load_from_disk(self, idx):
        path = os.path.join(self.temp_dir, f"dataset_{idx}.npz")
        arr = np.load(path)
        # Konverter til tensors
        images_tensor = torch.from_numpy(arr["images"])
        reflectance_maps_tensor = torch.from_numpy(arr["refl"])
        target_tensor = torch.from_numpy(arr["dem"]).unsqueeze(0)
        meta_tensor = torch.from_numpy(arr["meta"])
        return images_tensor, reflectance_maps_tensor, target_tensor, meta_tensor

    def __getitem__(self, idx):
        # Hvis vi har genereret før og reuse_count < limit → load fra disk
        if idx in self.cache and self.cache[idx] < self.reuse_limit:
            self.cache[idx] += 1
            return self._load_from_disk(idx)

        # Ellers generer nyt og nulstil tæller
        self._generate_and_save(idx)
        self.cache[idx] = 1
        return self._load_from_disk(idx)


class DEMDataset(Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Load PyTorch tensors directly
        loaded = torch.load(self.files[idx], map_location='cpu')
        
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
    def __init__(self, config=None):
        # NOTE: store simple kwargs/dicts only so the Dataset is picklable by DataLoader workers
        self.config = config
        self.epoch = 0
        self.base_seed = config["BASE_SEED"] if "BASE_SEED" in config else 42

    def set_epoch(self, epoch):
        """Set epoch for deterministic data generation."""
        self.epoch = epoch

    def __len__(self):
        return self.config["FLUID_TRAIN_DEMS"]

    def __getitem__(self, idx):
        # Set seed for reproducibility
        epoch_seed = self.base_seed + self.epoch * len(self) + idx
        torch.manual_seed(epoch_seed)
        np.random.seed(epoch_seed % (2**32 - 1))

        if self.config["USE_LRO_DEMS"]:
            images, reflectance_maps, dem_tensor, metas = generate_and_return_lro_data(config=self.config, device='cpu')

            if not torch.is_tensor(dem_tensor):
                dem_tensor = torch.from_numpy(dem_tensor)
   
            return torch.stack(images), torch.stack(reflectance_maps), dem_tensor.unsqueeze(0), torch.tensor(metas, dtype=torch.float32)

        else:
            images, reflectance_maps, dem_np, metas = generate_and_return_data_CPU(config=self.config)

            images_np = np.stack(images, axis=0)  # (5, H_img, W_img)
            refl_np = np.stack(reflectance_maps, axis=0)  # (5, H_dem, W_dem)
            meta_np = np.array(metas, dtype=np.float32)  # (5,5)

            target_tensor = torch.from_numpy(dem_np).unsqueeze(0)
            images_tensor = torch.from_numpy(images_np)
            reflectance_maps_tensor = torch.from_numpy(refl_np)
            meta_tensor = torch.from_numpy(meta_np)

            return images_tensor, reflectance_maps_tensor, target_tensor, meta_tensor


def train_epoch(model, train_loader, optimizer, scaler, device, train_mean, train_std, current_epoch=None, total_epochs=None, non_blocking=None,
                w_mse=None, w_grad=None, w_refl=None, use_amp=None, hapke_params=None, camera_params=None, autocast_device_type=None, grad_clip=None):
    """Train for one epoch with reflectance map loss"""
    model.train()
    running_loss = 0.0
    total_images = 0

    train_pbar = tqdm(train_loader, desc=f"Training on epoch {current_epoch}/{total_epochs}", leave=False, position=0, dynamic_ncols=True)

    #DEBUG: track batch losses
    # batch_losses_train = []

    for batch_idx, (images, reflectance_maps, targets, meta) in enumerate(train_pbar):
        # move everything to device - costly but necessary
        images = images.to(device, non_blocking=non_blocking)
        reflectance_maps = reflectance_maps.to(device, non_blocking=non_blocking)
        targets = targets.to(device, non_blocking=non_blocking)
        meta = meta.to(device, non_blocking=non_blocking)

        # print(f"Pre-normalization: mean={images.mean().item():.4f}, std={images.std().item():.4f}")
        # print(f"Images min value: {images.min().item():.4f}, max value: {images.max().item():.4f}")

        # Normalize in-place (inputs are on device)
        images = normalize_inputs(images, train_mean, train_std)

        # #check that normalization worked
        # print(f"Post-normalization: mean={images.mean().item():.4f}, std={images.std().item():.4f}")

        optimizer.zero_grad()
        
        with torch.amp.autocast(device_type=autocast_device_type, enabled=use_amp):
            # forward pass
            outputs = model(images, meta, target_size=targets.shape[-2:])
            # compute loss (calculate_total_loss should accept device tensors)
            loss = calculate_total_loss(outputs, targets, reflectance_maps, meta, device=device,
                                        camera_params=camera_params, hapke_params=hapke_params,
                                        w_grad=w_grad, w_refl=w_refl, w_mse=w_mse)
        
        # backward pass and step optimizer via scaler
        scaler.scale(loss).backward()

        if grad_clip is not None and grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        # # DEBUG: Print loss components for first batch
        # if batch_idx == 0:
        #     if hasattr(loss, 'loss_components'):
        #         comp = loss.loss_components
        #         print(f"[TRAIN] MSE: {comp['mse']:.3f} (w={w_mse}), Grad: {comp['grad']:.3f} (w={w_grad}), Refl: {comp['refl']:.3f} (w={w_refl})")
        #         total = w_mse * comp['mse'] + w_grad * comp['grad'] + w_refl * comp['refl']
        #         print(f"[TRAIN] Weighted contributions: MSE={w_mse * comp['mse']:.3f}, Grad={w_grad * comp['grad']:.3f}, Refl={w_refl * comp['refl']:.3f}, Total={total:.3f}")

        # DEBUG: track batch losses
        # batch_losses_train.append(loss.item())

        # accumulate running loss
        bsz = images.size(0) # batch size
        running_loss += loss.item() * bsz # accumulate loss times batch size
        total_images += bsz
        train_pbar.set_postfix(loss=f"{(running_loss / total_images):.4f}")

    train_pbar.close()
    # return running_loss / total_images, batch_losses_train
    return running_loss / total_images


@torch.no_grad()
def validate_epoch(model, val_loader, device, train_mean, train_std, current_epoch=None, total_epochs=None, non_blocking=None,
             w_mse=None, w_grad=None, w_refl=None, use_amp=None, hapke_params=None, camera_params=None, autocast_device_type=None):
    """Validate the model with reflectance map loss"""
    model.eval()
    running_loss = 0.0
    total_images = 0

    val_pbar = tqdm(val_loader, desc=f"Validating on epoch {current_epoch}/{total_epochs}", leave=False, position=0, dynamic_ncols=True)

    #DEBUG: track batch losses
    # batch_losses_val = []

    for batch_idx, (images, reflectance_maps, targets, meta) in enumerate(val_pbar):
        # move everything to device - costly but necessary
        images = images.to(device, non_blocking=non_blocking)
        reflectance_maps = reflectance_maps.to(device, non_blocking=non_blocking)
        targets = targets.to(device, non_blocking=non_blocking)
        meta = meta.to(device, non_blocking=non_blocking)

        # Normalize in-place (inputs are on device)
        images = normalize_inputs(images, train_mean, train_std)
        
        with torch.amp.autocast(device_type=autocast_device_type, enabled=use_amp):
            # Pass target size to model for proper upsampling
            outputs = model(images, meta, target_size=targets.shape[-2:])
            loss = calculate_total_loss(outputs, targets, reflectance_maps, meta, device=device,
                                        camera_params=camera_params, hapke_params=hapke_params,
                                        w_grad=w_grad, w_refl=w_refl, w_mse=w_mse)
        
        # # DEBUG: Print loss components for first batch
        # if batch_idx == 0:
        #     if hasattr(loss, 'loss_components'):
        #         comp = loss.loss_components
        #         print(f"[TRAIN] MSE: {comp['mse']:.3f} (w={w_mse}), Grad: {comp['grad']:.3f} (w={w_grad}), Refl: {comp['refl']:.3f} (w={w_refl})")
        #         total = w_mse * comp['mse'] + w_grad * comp['grad'] + w_refl * comp['refl']
        #         print(f"[TRAIN] Weighted contributions: MSE={w_mse * comp['mse']:.3f}, Grad={w_grad * comp['grad']:.3f}, Refl={w_refl * comp['refl']:.3f}, Total={total:.3f}")

        # DEBUG: track batch losses
        # batch_losses_val.append(loss.item())

        # calculate and accumulate running loss
        bsz = images.size(0)
        running_loss += loss.item() * bsz
        total_images += bsz
        val_pbar.set_postfix(loss=f"{(running_loss / total_images):.4f}")
    val_pbar.close()
    # return running_loss / total_images, batch_losses_val
    return running_loss / total_images

@torch.no_grad()
def evaluate_on_test_files(model=None, test_loader=None, device=None, train_mean=None, train_std=None, non_blocking=None,
                           w_mse=None, w_grad=None, w_refl=None, use_amp=None, hapke_params=None, camera_params=None):
    
    if any(v is None for v in [model, test_loader, device, train_mean, train_std, camera_params, hapke_params, w_mse, w_grad, w_refl, use_amp]):
        raise ValueError("All arguments must be provided to evaluate_on_test_files.")
    """Evaluate the model on test files with reflectance map loss"""
    model.eval()
    running_loss = 0.0
    total_images = 0
    abs_sum = 0.0
    n_pixels = 0

    test_pbar = tqdm(test_loader, desc=f"Evaluating on test files", leave=False, position=0, dynamic_ncols=True)

    for images, reflectance_maps, targets, meta in test_pbar:
        # move everything to device - costly but necessary
        images = images.to(device, non_blocking=non_blocking)
        reflectance_maps = reflectance_maps.to(device, non_blocking=non_blocking)
        targets = targets.to(device, non_blocking=non_blocking)
        meta = meta.to(device, non_blocking=non_blocking)

        # Normalize in-place (inputs are on device)
        images = normalize_inputs(images, train_mean, train_std)
        
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            # Pass target size to model for proper upsampling
            outputs = model(images, meta, target_size=targets.shape[-2:])
            loss = calculate_total_loss(outputs, targets, reflectance_maps, meta, device=device,
                                        camera_params=camera_params, hapke_params=hapke_params,
                                        w_grad=w_grad, w_refl=w_refl, w_mse=w_mse)

        # Compute Absolute Mean Error (AME)
        abs_diff = torch.abs(outputs - targets)
        abs_sum += abs_diff.sum().item()
        n_pixels += abs_diff.numel()

        # calculate running loss
        bsz = images.size(0)
        running_loss += loss.item() * bsz
        total_images += bsz
        test_pbar.set_postfix(loss=f"{(running_loss / total_images):.4f}")

    test_pbar.close()

    test_ame = abs_sum / n_pixels
    test_loss = running_loss / total_images
    return test_loss, test_ame


def estimate_dynamic_batch_size(model, config=None,
                                optimizer=None, device=None, scaler=None,
                                start_bs=16, max_trials=10, max_batch_size=256,
                                use_amp=False, camera_params=None, hapke_params=None,
                                w_mse=None, w_grad=None, w_refl=None):
    """
    Estimer maksimal batchstørrelse uden OOM ved at generere syntetiske data.
    input_shape: (C, H, W) for billeder
    meta_shape: (M,) for meta-data (valgfri)
    refl_shape: (R, H, W) for reflektanskort (valgfri)
    """
    print("Estimating maximum batch size with synthetic data...")
    input_shape = (config["IMAGES_PER_DEM"], config["IMAGE_H"], config["IMAGE_W"])
    # Meta per-image vector size is typically 5 (sun/camera params). If a
    # different value exists in config, use it; otherwise default to 5.
    meta_dim = config.get("META_DIM", 5) if config is not None else 5
    meta_shape = (config["IMAGES_PER_DEM"], meta_dim)
    refl_shape = (config["IMAGES_PER_DEM"], config["DEM_SIZE"], config["DEM_SIZE"])

    model.train()

    test_bs = start_bs
    max_bs = 0
    trials = 0

    while trials < max_trials:
        trials += 1
        try:
            # Generér syntetiske batches
            images = torch.randn((test_bs,) + input_shape, device=device)
            # Targets (DEMs) live at DEM resolution (config['DEM_SIZE']).
            targets = torch.randn((test_bs, 1, config["DEM_SIZE"], config["DEM_SIZE"]), device=device)
            refls = torch.randn((test_bs,) + refl_shape, device=device) if refl_shape else None
            meta = torch.randn((test_bs,) + meta_shape, device=device) if meta_shape else None

            optimizer_zero = optimizer.zero_grad if optimizer is not None else lambda *a, **k: None
            optimizer_zero(set_to_none=True)

            with torch.amp.autocast(device_type=device.type if device.type in ['cuda', 'mps'] else 'cpu', enabled=use_amp):
                outputs = model(images, meta) if meta is not None else model(images)

                # Brug din loss-funktion
                loss = calculate_total_loss(outputs, targets, refls, meta, device=device,
                                            camera_params=camera_params, hapke_params=hapke_params,
                                            w_grad=w_grad, w_refl=w_refl, w_mse=w_mse)

            if scaler and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.update()
            else:
                loss.backward()

            max_bs = test_bs
            del outputs, loss, images, targets, refls, meta
            torch.cuda.empty_cache()
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elif device.type == 'mps' and hasattr(torch.mps, 'synchronize'):
                torch.mps.synchronize()

            test_bs *= 2  # Dobbel batchstørrelsen
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f"OOM at batch size {test_bs}, last successful: {max_bs}")
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                elif device.type == 'mps' and hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
                break
            else:
                raise

    if max_bs == 0:
        max_bs = start_bs

    print(f"Max batch size: {max_bs}")
    if max_bs > max_batch_size:
        print(f"Limiting max batch size to {max_batch_size}")
        max_bs = max_batch_size
    return max_bs


__all__ = ['DEMDataset', 
           'FluidDEMDataset',
           'DEMDatasetHDF5', 
           'calculate_total_loss', 
           'train_epoch', 
           'validate_epoch', 
           'evaluate_on_test_files',
           'estimate_dynamic_batch_size']
