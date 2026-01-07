import torch
import torch.amp
import h5py
import numpy as np
import glob
import os
from tqdm import tqdm

def _move_to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        cls = list if isinstance(obj, list) else tuple
        return cls(_move_to_device(v, device) for v in obj)
    return obj


def convert_npz_to_hdf5(npz_dir, output_path):
    """
    Convert all .npz files in npz_dir into a single HDF5 file.

    Args:
        npz_dir (str): Path to directory with .npz files.
        output_path (str): Path to HDF5 output file.
    """
    # Find all npz files
    npz_files = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))
    if len(npz_files) == 0:
        raise ValueError(f"No .npz files found in {npz_dir}")

    # Open HDF5 file for writing
    with h5py.File(output_path, 'w') as h5f:
        n = len(npz_files)

        # Use first file to determine shapes
        sample = np.load(npz_files[0])
        dem_shape = sample['dem'].shape
        data_shape = sample['data'].shape  # (5, H_img, W_img)
        meta_shape = sample['meta'].shape

        # Create HDF5 datasets
        dem_ds = h5f.create_dataset('dem', shape=(n, *dem_shape), dtype='float32')
        data_ds = h5f.create_dataset('images', shape=(n, *data_shape), dtype='float32')
        refl_ds = h5f.create_dataset('reflectance_maps', shape=(n, data_shape[0], dem_shape[0], dem_shape[1]), dtype='float32')
        meta_ds = h5f.create_dataset('meta', shape=(n, *meta_shape), dtype='float32')

        # Fill data
        for i, file in enumerate(tqdm(npz_files, desc="Converting NPZ to HDF5")):
            loaded = np.load(file)
            dem_ds[i] = loaded['dem'].astype(np.float32)
            data_ds[i] = loaded['data'].astype(np.float32)
            refl_ds[i] = loaded.get('reflectance_maps', np.zeros((data_shape[0], dem_shape[0], dem_shape[1]), dtype=np.float32))
            meta_ds[i] = loaded['meta'].astype(np.float32)

    print(f"HDF5 file saved: {output_path}")


def init_distributed(backend=None):
    if backend is None:
        raise ValueError("Distributed backend must be specified.")
    
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # torchrun sets these environment variables
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend=backend, init_method='env://')
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    else:
        rank = 0
        world_size = 1
    
    return rank, world_size


def get_device():
    import torch
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    return device

def round_list(lst, ndigits=3):
    return [round(x, ndigits) for x in lst] 

# Collate function that does not move to device (to be used with DataLoader)
# This will utilize the num_workers properly without redundant copies.
def collate(batch):
    images, refls, targets, metas = zip(*batch)
    images = torch.stack(images, dim=0)      # (B, C, H, W)
    refls  = torch.stack(refls, dim=0)
    targets = torch.stack(targets, dim=0)
    metas = torch.stack(metas, dim=0)
    return images, refls, targets, metas

@torch.no_grad()
def normalize_inputs(x, mean, std):
    mean = mean.view(1, -1, 1, 1).to(x.device)
    std = std.view(1, -1, 1, 1).to(x.device)
    return (x - mean) / std.clamp_min(1e-6)

def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    dataset.file = h5py.File(dataset.hdf5_path, 'r', swmr=True)  # SWMR = sikker til flere readers
    dataset.images = dataset.file['images']
    dataset.reflectance_maps = dataset.file['reflectance_maps']
    dataset.dems = dataset.file['dem']
    dataset.metas = dataset.file['meta']

    # --- per-worker RNG seeding ---
    # Non-deterministic: use OS entropy so workers produce independent random streams each run
    import os, time, random as _random, numpy as _np, torch as _torch
    seed = (int.from_bytes(os.urandom(8), 'little')
            ^ (os.getpid() << 16)
            ^ worker_id
            ^ int(time.time() * 1e6))
    seed32 = seed % (2**32)
    _np.random.seed(seed32)
    _random.seed(seed32)
    _torch.manual_seed(seed32)
    try:
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed(seed32)
    except Exception:
        pass

# @torch.no_grad()
# def compute_input_stats(loader, images_per_dem):
#     """Compute per-channel mean and std over the dataset in loader."""
#     cnt = 0
#     mean = torch.zeros(images_per_dem)
#     M2 = torch.zeros(images_per_dem)
#     for batch in tqdm(loader, desc="Computing input stats", leave=False, position=0, dynamic_ncols=True):
#         images = batch[0]  # (B, C, H, W) or numpy array
#         if isinstance(images, np.ndarray):
#             images = torch.from_numpy(images)
#         images = images.cpu().float()
#         B = images.size(0)
#         ch_means = images.view(B, images_per_dem, -1).mean(dim=2).mean(dim=0)
#         cnt += 1
#         delta = ch_means - mean
#         mean += delta / cnt
#         M2 += delta * (ch_means - mean)
#     var = M2 / max(cnt - 1, 1)
#     std = torch.sqrt(var.clamp_min(1e-8))
#     return mean, std


@torch.no_grad()
def compute_input_stats(loader, images_per_dem):
    total_sum = torch.zeros(images_per_dem)
    total_sq_sum = torch.zeros(images_per_dem)
    total_pixels = 0
    for batch in tqdm(loader, desc="Computing input stats", leave=False, position=0, dynamic_ncols=True):
        images = batch[0]  # (B, C, H, W) or numpy array
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        images = images.cpu().float()
        B, C, H, W = images.shape
        total_sum += images.sum(dim=[0, 2, 3])
        total_sq_sum += (images ** 2).sum(dim=[0, 2, 3])
        total_pixels += B * H * W
    mean = total_sum / total_pixels
    var = (total_sq_sum / total_pixels) - mean ** 2
    std = torch.sqrt(var.clamp_min(1e-8))
    return mean, std




__all__ = ['convert_npz_to_hdf5', 
           'init_distributed', 
           'get_device', 
           'normalize_inputs', 
           'compute_input_stats', 
           'worker_init_fn', 
           'round_list', 
           'collate']