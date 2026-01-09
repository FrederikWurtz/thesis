import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from master.train.trainer_core import FluidDEMDataset, DEMDataset
from master.train.train_utils import normalize_inputs
import time

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from master.data_sim.dataset_io import list_pt_files
import os

from master.configs.config_utils import load_config_file
from master.models.losses import calculate_total_loss
from master.models.unet import UNet



def is_main():
    return int(os.environ["LOCAL_RANK"]) == 0

def ddp_setup():
    try:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        init_process_group(backend="nccl")
    except KeyError:
        raise RuntimeError("LOCAL_RANK not found in environment variables. Please run this script using torch.distributed.launch or torchrun for multi-GPU training.")


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        config: dict,
        snapshot_path: str,
        train_mean: torch.Tensor = None,
        train_std: torch.Tensor = None,
        val_data: DataLoader = None,
    ) -> None:
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model = model.to(self.gpu_id)
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = optimizer
        self.save_every = config["SAVE_EVERY"]
        self.epochs_run = 0
        self.train_loss_history = []  # Track losses
        self.val_loss_history = []    # Track validation losses
        self.snapshot_path = snapshot_path
        if os.path.exists(snapshot_path):
            if is_main():
                print("Loading snapshot")
            self._load_snapshot(snapshot_path)
        self.config = config
        self.train_mean = train_mean
        self.train_std = train_std
        self.model = DDP(self.model, device_ids=[self.gpu_id])

    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        self.train_loss_history = snapshot.get("LOSS_HISTORY", [])
        self.val_loss_history = snapshot.get("VAL_LOSS_HISTORY", [])
        if is_main():
            print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _run_epoch(self, epoch):
        t0 = time.time()
        if is_main():
            print("Running epoch {}".format(epoch))

        epoch_loss = 0.0
        total_samples = 0

        self.train_data.sampler.set_epoch(epoch)
        for images, reflectance_maps, targets, metas in self.train_data:
            images = images.to(self.gpu_id)
            metas = metas.to(self.gpu_id)
            reflectance_maps = reflectance_maps.to(self.gpu_id)
            images = normalize_inputs(images, self.train_mean, self.train_std)

            source = images, metas, reflectance_maps
            targets = targets.to(self.gpu_id)

            batch_size = images.size(0)  # Get actual batch size
            batch_loss = self._run_batch(source, targets)

            # Weight by batch size
            epoch_loss += batch_loss * batch_size
            total_samples += batch_size

        
        # Weighted average loss
        avg_loss = epoch_loss / total_samples
        
        # Gather losses and sample counts from all GPUs
        avg_loss_tensor = torch.tensor([avg_loss], device=self.gpu_id)
        total_samples_tensor = torch.tensor([total_samples], device=self.gpu_id)
        
        torch.distributed.all_reduce(avg_loss_tensor, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(total_samples_tensor, op=torch.distributed.ReduceOp.SUM)
        
        # Compute global weighted average
        global_avg_loss = avg_loss_tensor.item() / torch.distributed.get_world_size()
    
        # Store loss on main process
        if is_main():
            self.train_loss_history.append(global_avg_loss)
            total_time = time.time() - t0
            print(f"[GPU{self.gpu_id}] Epoch {epoch} | Loss: {global_avg_loss:.6f} | Samples: {total_samples_tensor.item()} | Time: {total_time:.2f}s")

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        images, metas, reflectance_maps = source
        device = images.device
        outputs = self.model(images, metas, target_size=targets.shape[-2:])
        loss = calculate_total_loss(outputs, targets, reflectance_maps, metas, device=device,
                                        camera_params=self.config["CAMERA_PARAMS"], hapke_params=self.config["HAPKE_KWARGS"],
                                        w_grad=self.config["W_GRAD"], w_refl=self.config["W_REFL"], w_mse=self.config["W_MSE"])
        loss.backward()
        self.optimizer.step()
        return loss.item()  # Return loss value

    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
            "TRAIN_LOSS_HISTORY": self.train_loss_history,  # Save loss history
            "VAL_LOSS_HISTORY": self.val_loss_history,  # Save validation loss history
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")
        
        # Also save loss history separately as CSV for easy plotting
        train_loss_file = self.snapshot_path.replace('.pt', '_train_losses.csv')
        with open(train_loss_file, 'w') as f:
            f.write("epoch,loss\n")
            for i, loss in enumerate(self.train_loss_history, start=1):
                f.write(f"{i},{loss}\n")
        # Also save validation loss history
        val_loss_file = self.snapshot_path.replace('.pt', '_val_losses.csv')
        with open(val_loss_file, 'w') as f:
            f.write("epoch,loss\n")
            for i, loss in enumerate(self.val_loss_history):
                actual_epoch = (i + 1) * self.save_every
                f.write(f"{actual_epoch},{loss}\n")

                
    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)
            
            # Validate on ALL GPUs at checkpoint intervals
            if epoch % self.save_every == 0:
                self._validate(epoch)
                
                # But only GPU 0 saves the snapshot
                if self.gpu_id == 0:
                    self._save_snapshot(epoch)

    @torch.no_grad()
    def _validate(self, epoch):
        """Run validation and return average loss"""
        if self.val_data is None:
            return None
            
        t0 = time.time()
        if is_main():
            print(f"Running validation for epoch {epoch}")
        
        self.model.eval()  # Set to evaluation mode
        val_loss = 0.0
        total_samples = 0
        
        for images, reflectance_maps, targets, metas in self.val_data:
            images = images.to(self.gpu_id)
            metas = metas.to(self.gpu_id)
            reflectance_maps = reflectance_maps.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            images = normalize_inputs(images, self.train_mean, self.train_std)
            
            batch_size = images.size(0)
            outputs = self.model(images, metas, target_size=targets.shape[-2:])
            loss = calculate_total_loss(
                outputs, targets, reflectance_maps, metas, 
                device=self.gpu_id,
                camera_params=self.config["CAMERA_PARAMS"], 
                hapke_params=self.config["HAPKE_KWARGS"],
                w_grad=self.config["W_GRAD"], 
                w_refl=self.config["W_REFL"], 
                w_mse=self.config["W_MSE"]
            )
            
            val_loss += loss.item() * batch_size
            total_samples += batch_size
        
        # Check if this GPU has no validation samples
        if total_samples == 0:
            avg_val_loss = 0.0
        else:
            avg_val_loss = val_loss / total_samples
        
        # Gather losses and sample counts from all GPUs
        avg_val_loss_tensor = torch.tensor([avg_val_loss], device=self.gpu_id)
        total_samples_tensor = torch.tensor([total_samples], device=self.gpu_id)
        
        torch.distributed.all_reduce(avg_val_loss_tensor, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(total_samples_tensor, op=torch.distributed.ReduceOp.SUM)
        
        # Check if no GPU has validation samples
        global_total_samples = total_samples_tensor.item()
        if global_total_samples == 0:
            if is_main():
                print(f"Warning: No validation samples found. Skipping validation.")
            return None
        
        # Compute global weighted average
        global_val_loss = avg_val_loss_tensor.item() / torch.distributed.get_world_size()
        
        # All GPUs store (needed for checkpoint consistency)
        self.val_loss_history.append(global_val_loss)
        
        if is_main():
            val_time = time.time() - t0
            print(f"[GPU{self.gpu_id}] Epoch {epoch} | Val Loss: {global_val_loss:.6f} | Samples: {global_total_samples} | Time: {val_time:.2f}s")
        
        return global_val_loss


def load_train_objs(config, run_path: str):
    train_set = FluidDEMDataset(config) # load your dataset
    val_path = os.path.join(run_path, 'val') # load validation dataset
    val_files = list_pt_files(val_path)
    val_set = DEMDataset(val_files)
    model = UNet(in_channels=config["IMAGES_PER_DEM"], out_channels=1)  # load your model
    optimizer = torch.optim.Adam(model.parameters(), lr=config["LR"], weight_decay=config["WEIGHT_DECAY"])
    return train_set, val_set, model, optimizer

def prepare_dataloader(dataset: Dataset, batch_size: int, num_workers: int = 2, prefetch_factor: int = 4) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset),
        num_workers=num_workers,
        prefetch_factor=prefetch_factor
    )
