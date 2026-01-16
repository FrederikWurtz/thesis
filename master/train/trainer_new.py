import warnings


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

from torch.amp import autocast, GradScaler
import torch
from torch.profiler import profile, record_function, ProfilerActivity

# 🔥 Suppress torch.compile() warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torch._dynamo')
warnings.filterwarnings('ignore', category=UserWarning, module='torch._logging')
warnings.filterwarnings('ignore', message='.*Profiler function.*will be ignored.*')



def is_main():
    return int(os.environ["LOCAL_RANK"]) == 0

def ddp_setup():
    try:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))
        # 🔥 Enable cuDNN autotuner for convolution optimization
        torch.backends.cudnn.benchmark = True
        
        # Optional: Enable TF32 for Ampere GPUs (A100, RTX 3090, etc.)
        if torch.cuda.get_device_capability()[0] >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        # 🔥 Use TensorFloat32 for faster matmul on Ampere+ GPUs
        torch.set_float32_matmul_precision('high')  # or 'medium' for even more speed
        
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
        test_data: DataLoader = None,
    ) -> None:
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model = model.to(self.gpu_id)
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.optimizer = optimizer
        self.save_every = config["SAVE_EVERY"]
        self.epochs_run = 0
        self.train_loss_history = []  # Track losses
        self.val_loss_history = []    # Track validation losses
        self.train_timings = []
        self.val_timings = []
        self.config = config
        self.train_mean = train_mean
        self.train_std = train_std
        self.model = DDP(self.model, device_ids=[self.gpu_id]) # First wrap model in DDP
        if is_main():
            print("🔥 About to compile model with torch.compile() - this may take 5-30 minutes on first run...")
        self.model = torch.compile(self.model, mode='reduce-overhead')  # Then compile with torch.compile
        if is_main():
            print("✅ Model compilation complete!")
        self.dtype = torch.bfloat16 if self.config["USE_BF16"] else torch.float16
        self.use_amp = self.config["USE_AMP"]
        self.scaler = GradScaler('cuda') if (self.use_amp and self.dtype == torch.float16) else None
        self.snapshot_path = snapshot_path # Path to save/load snapshots
        if os.path.exists(snapshot_path): 
            if is_main():
                print("Loading snapshot")
            self._load_snapshot(snapshot_path) # Then, after DDP wrapping, load snapshot if it exists

    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "OPTIMIZER_STATE": self.optimizer.state_dict(),  # Save optimizer state
            "EPOCHS_RUN": epoch,
            "TRAIN_LOSS_HISTORY": self.train_loss_history,  # Save loss history
            "VAL_LOSS_HISTORY": self.val_loss_history,  # Save validation loss history
            "TRAIN_TIMINGS": self.train_timings,
            "VAL_TIMINGS": self.val_timings,
        }

        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")
        
        # Also save loss history separately as CSV for easy plotting
        train_loss_file = self.snapshot_path.replace('snapshot.pt', 'train_losses.csv')
        with open(train_loss_file, 'w') as f:
            f.write("epoch,loss\n")
            for i, loss in enumerate(self.train_loss_history, start=0):
                f.write(f"{i},{loss}\n")
        # Also save validation loss history
        val_loss_file = self.snapshot_path.replace('snapshot.pt', 'val_losses.csv')
        with open(val_loss_file, 'w') as f:
            f.write("epoch,loss\n")
            for i, loss in enumerate(self.val_loss_history, start=0):
                actual_epoch = i * self.save_every
                f.write(f"{actual_epoch},{loss}\n")
        # Also save timings
        train_timing_file = self.snapshot_path.replace('snapshot.pt', 'train_timings.csv')
        with open(train_timing_file, 'w') as f:
            f.write("epoch,time_seconds\n")
            for i, timing in enumerate(self.train_timings, start=0):
                f.write(f"{i},{timing}\n")
        val_timing_file = self.snapshot_path.replace('snapshot.pt', 'val_timings.csv')
        with open(val_timing_file, 'w') as f:
            f.write("epoch,time_seconds\n")
            for i, timing in enumerate(self.val_timings, start=0):
                actual_epoch = i * self.save_every
                f.write(f"{actual_epoch},{timing}\n")

                
    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location=loc)

        self.model.module.load_state_dict(snapshot["MODEL_STATE"])

        # Load optimizer state and ensure all tensors are on correct device
        optimizer_state = snapshot["OPTIMIZER_STATE"]
        
        # Move optimizer state tensors to correct device
        for state in optimizer_state['state'].values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.gpu_id)

        self.optimizer.load_state_dict(optimizer_state)  # Load optimizer state

        # Load scaler state if it exists and we're using AMP
        if self.scaler is not None and "SCALER_STATE" in snapshot:
            self.scaler.load_state_dict(snapshot["SCALER_STATE"])
            if is_main():
                print("Loaded GradScaler state")

        self.epochs_run = snapshot["EPOCHS_RUN"] + 1  # Resume from NEXT epoch
        self.train_loss_history = snapshot["TRAIN_LOSS_HISTORY"]
        self.val_loss_history = snapshot["VAL_LOSS_HISTORY"]
        self.train_timings = snapshot["TRAIN_TIMINGS"]
        self.val_timings = snapshot["VAL_TIMINGS"]
        if is_main():
            print(f"Found snapshot saved at epoch {self.epochs_run - 1}.")
            print(f"Resuming model from snapshot at Epoch {self.epochs_run}")
        self.model.train()  # Set back to training mode

    def _run_epoch(self, epoch):
        t0 = time.time()
        if is_main():
            print("Running epoch {}".format(epoch))

        # 🔥 Accumulate on GPU instead of CPU
        epoch_loss = torch.zeros(1, dtype=torch.float32, device=f'cuda:{self.gpu_id}')
        total_samples = 0

        # Add detailed timing if profiling
        use_profiler = self.config.get("USE_PROFILER", False)
        if use_profiler and is_main():
            data_load_time = 0.0
            compute_time = 0.0

        # Set epoch for distributed sampler and dataset randomness, for reproducibility
        self.train_data.sampler.set_epoch(epoch)

            # If your dataset has on-the-fly generation, set its random seed based on epoch
        if hasattr(self.train_data.dataset, 'set_epoch'):
            self.train_data.dataset.set_epoch(epoch)

        for batch_idx, (images, reflectance_maps, targets, metas) in enumerate(self.train_data):
            if use_profiler and is_main():
                batch_start = time.time()
            
            images = images.to(self.gpu_id)
            metas = metas.to(self.gpu_id)
            reflectance_maps = reflectance_maps.to(self.gpu_id)
            images = normalize_inputs(images, self.train_mean, self.train_std)

            source = images, metas, reflectance_maps
            targets = targets.to(self.gpu_id)

            if use_profiler and is_main():
                data_load_time += time.time() - batch_start
                compute_start = time.time()

            batch_size = images.size(0)
            mean_batch_loss = self._run_batch(source, targets, return_tensors=True)
            
            if use_profiler and is_main():
                compute_time += time.time() - compute_start

            # 🔥 Accumulate on GPU (detach to avoid building huge computation graph)
            epoch_loss += mean_batch_loss.detach() * batch_size
            total_samples += batch_size
            
            # Print batch-level timing for first epoch
            if use_profiler and is_main() and epoch == 0 and batch_idx < 5:
                print(f"  Batch {batch_idx}: Data load: {(time.time()-batch_start)*1000:.2f}ms | "
                      f"Compute: {compute_time*1000:.2f}ms")

        # 🔥 Only sync once at the end of the epoch
        epoch_loss_value = epoch_loss.item()

        # Gather total loss sums (not averages) from all GPUs
        epoch_loss_tensor = torch.tensor([epoch_loss_value], dtype=torch.float32, device=f'cuda:{self.gpu_id}')
        total_samples_tensor = torch.tensor([total_samples], dtype=torch.int64, device=f'cuda:{self.gpu_id}')
        
        torch.distributed.all_reduce(epoch_loss_tensor, op=torch.distributed.ReduceOp.SUM) # Sum of losses across GPUs
        torch.distributed.all_reduce(total_samples_tensor, op=torch.distributed.ReduceOp.SUM) # Sum of samples across GPUs
        
        # Compute true weighted average: total_loss / total_samples
        global_avg_loss = epoch_loss_tensor.item() / total_samples_tensor.item()
    
        # Store loss on main process
        if is_main():
            self.train_loss_history.append(global_avg_loss)
            total_time = time.time() - t0
            self.train_timings.append(total_time)
            print(f"[GPU{self.gpu_id}] Epoch {epoch} | Loss: {global_avg_loss:.2e} | Samples: {total_samples_tensor.item()} | Time: {total_time:.2f}s")

    def _run_batch(self, source, targets, return_tensors: bool = False):
        self.optimizer.zero_grad()
        images, metas, reflectance_maps = source
        device = images.device
        
        with autocast('cuda', enabled=self.use_amp, dtype=self.dtype):
            outputs = self.model(images, metas, target_size=targets.shape[-2:])
            total_loss = calculate_total_loss(
                outputs, targets, reflectance_maps, metas, 
                device=device,
                camera_params=self.config["CAMERA_PARAMS"], 
                hapke_params=self.config["HAPKE_KWARGS"],
                w_grad=self.config["W_GRAD"], 
                w_refl=self.config["W_REFL"], 
                w_mse=self.config["W_MSE"],
                height_norm=self.config["HEIGHT_NORMALIZATION"] + self.config["HEIGHT_NORMALIZATION_PM"], # the maximum possible height for normalization
                return_components=False
            )
        
        # # Check loss component values
        # if is_main():
        #     print(f"    Loss components: MSE={loss_mse.item():.6f}, Grad={loss_grad.item():.6f}, Refl={loss_refl.item():.6f}, Total={total_loss.item():.6f}")

        # # 🔍 Diagnostic checks
        # if torch.isnan(total_loss) or torch.isinf(total_loss):
        #     print(f"⚠️ NaN/Inf detected in loss at epoch {self.epochs_run}")
        #     print(f"Output stats: min={outputs.min():.4f}, max={outputs.max():.4f}, mean={outputs.mean():.4f}")
        #     raise RuntimeError("NaN detected in loss!")
        
        total_loss.backward()
        
        # 🔍 Check gradients
        total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["GRAD_CLIP"])
        # if torch.isnan(total_norm) or torch.isinf(total_norm):
        #     print(f"⚠️ NaN/Inf in gradients! Norm: {total_norm:.4f}")
        #     raise RuntimeError("NaN detected in gradients!")
        
        if is_main() and total_norm > self.config["GRAD_CLIP"] * 0.8:
            print(f"⚠️ Large gradient norm: {total_norm:.4f} (clipped at {self.config['GRAD_CLIP']})")
        
        self.optimizer.step()

        if return_tensors:
            return total_loss
        else:
            return total_loss.item()

    def train(self, max_epochs: int):
        # Enable profiling for first few batches
        use_profiler = self.config["USE_PROFILER"]
        
        if use_profiler and is_main():
            prof = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    os.path.join(os.path.dirname(self.snapshot_path), '../profiler')
                ),
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            )
            prof.start()


        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)
            
            if use_profiler and is_main() and epoch == 0:
                prof.step()

            # Validate on ALL GPUs at checkpoint intervals
            if epoch % self.save_every == 0:
                self._validate(epoch)
                
                # But only GPU 0 saves the snapshot
                if self.gpu_id == 0:
                    self._save_snapshot(epoch)

        if use_profiler and is_main():
            prof.stop()
            print(f"Profiler trace saved to: {os.path.dirname(self.snapshot_path)}/../profiler")


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
            
            with autocast('cuda', enabled=self.use_amp, dtype=self.dtype):
                outputs = self.model(images, metas, target_size=targets.shape[-2:])
                loss = calculate_total_loss(
                    outputs, targets, reflectance_maps, metas, 
                    device=self.gpu_id,
                    camera_params=self.config["CAMERA_PARAMS"], 
                    hapke_params=self.config["HAPKE_KWARGS"],
                    w_grad=self.config["W_GRAD"], 
                    w_refl=self.config["W_REFL"], 
                    w_mse=self.config["W_MSE"],
                    height_norm=self.config["HEIGHT_NORMALIZATION"] + self.config["HEIGHT_NORMALIZATION_PM"], # the maximum possible height for normalization
                    return_components=False
                )
            
            val_loss += loss.item() * batch_size
            total_samples += batch_size
        
        
        # Gather losses and sample counts from all GPUs
        val_loss_tensor = torch.tensor([val_loss], device=self.gpu_id)
        total_samples_tensor = torch.tensor([total_samples], device=self.gpu_id)
        
        torch.distributed.all_reduce(val_loss_tensor, op=torch.distributed.ReduceOp.SUM) # Sum of validation losses across GPUs
        torch.distributed.all_reduce(total_samples_tensor, op=torch.distributed.ReduceOp.SUM) # Sum of validation samples across GPUs
        
        # Check if no GPU has validation samples
        global_total_samples = total_samples_tensor.item()
        if global_total_samples == 0:
            if is_main():
                print(f"Warning: No validation samples found. Skipping validation.")
            return None

        # Compute the true weighted average: total_val_loss / total_samples
        global_avg_val_loss = val_loss_tensor.item() / total_samples_tensor.item()

        
        if is_main():
            self.val_loss_history.append(global_avg_val_loss)
            val_time = time.time() - t0
            self.val_timings.append(val_time)
            print(f"[GPU{self.gpu_id}] Epoch {epoch} | Val Loss: {global_avg_val_loss:.2e} | Samples: {global_total_samples} | Time: {val_time:.2f}s")
        
        self.model.train()  # Set back to training mode
        return global_avg_val_loss

    @torch.no_grad()
    def test(self):
        """Run testing and return average loss and AME"""
        if self.test_data is None:
            if is_main():
                print("No test data provided. Skipping testing.")
            return None, None
            
        t0 = time.time()
        epoch = self.epochs_run
        if is_main():
            print(f"Evaluating on test dataset, after epoch {epoch}")
        
        self.model.eval()  # Set to evaluation mode
        test_loss = 0.0
        total_ame = 0.0
        total_samples = 0
        
        for images, reflectance_maps, targets, metas in self.test_data:
            images = images.to(self.gpu_id)
            metas = metas.to(self.gpu_id)
            reflectance_maps = reflectance_maps.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            images = normalize_inputs(images, self.train_mean, self.train_std)
            
            batch_size = images.size(0)

            with autocast('cuda', enabled=self.use_amp, dtype=self.dtype):
                outputs = self.model(images, metas, target_size=targets.shape[-2:])
                # Calculate loss
                loss = calculate_total_loss(
                    outputs, targets, reflectance_maps, metas, 
                    device=self.gpu_id,
                    camera_params=self.config["CAMERA_PARAMS"], 
                    hapke_params=self.config["HAPKE_KWARGS"],
                    w_grad=self.config["W_GRAD"], 
                    w_refl=self.config["W_REFL"], 
                    w_mse=self.config["W_MSE"],
                    height_norm=self.config["HEIGHT_NORMALIZATION"] + self.config["HEIGHT_NORMALIZATION_PM"], # the maximum possible height for normalization
                    return_components=False
                )
            
            # Calculate AME (Absolute Mean Error)
            ame = torch.abs(outputs - targets).mean()
            
            test_loss += loss.item() * batch_size
            total_ame += ame.item() * batch_size
            total_samples += batch_size
        
        
        # Gather losses, AMEs, and sample counts from all GPUs
        test_loss_tensor = torch.tensor([test_loss], device=self.gpu_id)
        ame_tensor = torch.tensor([total_ame], device=self.gpu_id)
        total_samples_tensor = torch.tensor([total_samples], device=self.gpu_id)
        
        torch.distributed.all_reduce(test_loss_tensor, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(ame_tensor, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(total_samples_tensor, op=torch.distributed.ReduceOp.SUM)
        
        # Check if no GPU has test samples
        global_total_samples = total_samples_tensor.item()
        if global_total_samples == 0:
            if is_main():
                print(f"Warning: No test samples found. Skipping testing.")
            return None, None
        
        # Compute global weighted averages
        global_test_loss = test_loss_tensor.item() / total_samples_tensor.item()
        global_ame = ame_tensor.item() / total_samples_tensor.item()
        
        if is_main():
            test_time = time.time() - t0
            print(f"[GPU{self.gpu_id}] Epoch {epoch} | Test Loss: {global_test_loss:.2e} | AME: {global_ame:.6f} | Samples: {global_total_samples} | Time: {test_time:.2f}s")
        
        self.model.train()  # Set back to training mode
        return global_test_loss, global_ame

def load_train_objs(config, run_path: str):
    train_set = FluidDEMDataset(config) # load your dataset
    val_path = os.path.join(run_path, 'val') # load validation dataset
    val_files = list_pt_files(val_path)
    val_set = DEMDataset(val_files)
    test_set = list_pt_files(os.path.join(run_path, 'test'))  # load test dataset
    test_set = DEMDataset(test_set)
    model = UNet(in_channels=config["IMAGES_PER_DEM"], out_channels=1)  # load your model
    optimizer = torch.optim.Adam(model.parameters(), lr=config["LR"], weight_decay=config["WEIGHT_DECAY"])
    return train_set, val_set, test_set, model, optimizer

def prepare_dataloader(dataset: Dataset, batch_size: int, num_workers: int = 2, prefetch_factor: int = 4) -> DataLoader:
    rank = int(os.environ["LOCAL_RANK"])
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset),
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=True,  # 🔥 Keep workers alive between epochs
        pin_memory_device=f'cuda:{rank}'  # 🔥 Pin directly to target GPU
    )
