import sys
import warnings

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from master.data_sim.generator import generate_and_save_data_pooled_multi_gpu
from master.train.trainer_core import FluidDEMDataset, DEMDataset, SemiFluidDEMDataset, is_main
from master.train.train_utils import normalize_inputs
import time
import subprocess

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from master.data_sim.dataset_io import list_pt_files
import os

from master.configs.config_utils import load_config_file
from master.models.losses import calculate_total_loss, calculate_total_loss_multi_band
from master.models.unet import UNet

from torch.amp import autocast, GradScaler
import torch
from torch.profiler import profile, record_function, ProfilerActivity

# 🔥 Suppress torch.compile() warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torch._dynamo')
warnings.filterwarnings('ignore', category=UserWarning, module='torch._logging')
warnings.filterwarnings('ignore', message='.*Profiler function.*will be ignored.*')



class Trainer_multiGPU:
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

    def _run_epoch(self, epoch, return_val=False):
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

        # Also set epoch in dataset to ensure deterministic data generation
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

        if return_val:
            return global_avg_loss

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
    def test(self, data_loader: DataLoader = None):
        """Run testing and return average loss and AME"""
        if self.test_data is None:
            if is_main():
                print("No test data provided. Skipping testing.")
            return None, None
        
        # Allow custom data loader for testing
        data_loader = self.test_data if data_loader is None else data_loader

        t0 = time.time()
        epoch = self.epochs_run
        if is_main():
            print(f"Evaluating on test dataset, after epoch {epoch}")
        
        self.model.eval()  # Set to evaluation mode
        test_loss = 0.0
        total_ame = 0.0
        total_samples = 0
        
        for images, reflectance_maps, targets, metas in data_loader:
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

def load_train_objs(config, run_path: str, epoch_shared=None):
    if config["USE_SEMIFLUID"]:
        if is_main():
            print("Using SemiFluidDEMDataset for training.")
        train_set = SemiFluidDEMDataset(config, epoch_shared=epoch_shared) # load your dataset
    else:
        train_set = FluidDEMDataset(config, epoch_shared=epoch_shared) # load your dataset
    val_path = os.path.join(run_path, 'val') # load validation dataset
    val_files = list_pt_files(val_path)
    val_set = DEMDataset(val_files, config=config)
    test_set = list_pt_files(os.path.join(run_path, 'test'))  # load test dataset
    test_set = DEMDataset(test_set, config=config)
    if config["USE_MULTI_BAND"]:
        out_channels = 3 # DEM, w band and theta_bar band
        model = UNet(in_channels=config["IMAGES_PER_DEM"], out_channels=out_channels, w_range=(config["W_MIN"], config["W_MAX"]), theta_range=(config["THETA_BAR_MIN"], config["THETA_BAR_MAX"]))  # load your model
    else:
        out_channels = 1
        model = UNet(in_channels=config["IMAGES_PER_DEM"], out_channels=out_channels)  # load your model
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



def generate_fluid_data(run_name: str, epoch: int):
    print(f"[Rank0] Generating fluid data for epoch {epoch}...", flush=True)
    cmd = [
        sys.executable,
        "generate_fluid_data.py",
        "--run", run_name,
        "--epoch", str(epoch),
    ]
    print(f"[Rank0] Spawning generator process: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, check=True)
    print(f"[Rank0] Generator finished with code {result.returncode}", flush=True)



class Trainer_multiGPU_multi_band:
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
        self.train_mean = train_mean.to(self.gpu_id)
        self.train_std = train_std.to(self.gpu_id)
        self.model = DDP(self.model, device_ids=[self.gpu_id]) # First wrap model in DDP
        if torch.distributed.get_world_size() == 1:
            print("🔥 About to compile model with torch.compile() on single GPU...")
            self.model = torch.compile(self.model, mode='reduce-overhead')
        else:
            if is_main():
                print("⚠️ Skipping torch.compile() in multi-GPU mode to avoid known compile issues on multiple GPUs.")
        self.dtype = torch.bfloat16 if self.config["USE_BF16"] else torch.float16
        self.use_amp = self.config["USE_AMP"]
        self.scaler = GradScaler('cuda') if (self.use_amp and self.dtype == torch.float16) else None
        self.snapshot_path = snapshot_path # Path to save/load snapshots
        if os.path.exists(snapshot_path): 
            if is_main():
                print("Loading snapshot")
            self._load_snapshot(snapshot_path) # Then, after DDP wrapping, load snapshot if it exists
        self.debug = self.config["DEBUG"]

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

    def _run_epoch(self, epoch, return_val=False):
        t0 = time.time()
        
        if is_main() and self.debug:
            print(f"--- Epoch {epoch} start ---")
            check_params_for_nans(self.model, tag=f"epoch_{epoch}_start")

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

        # DEBUG: set to eval mode for testing
        # self.model.eval()
        
        # Set epoch for distributed sampler and dataset randomness, for reproducibility
        self.train_data.sampler.set_epoch(epoch)

        # Also set epoch in dataset to ensure deterministic data generation
        self.train_data.dataset.set_epoch(epoch)

        for batch_idx, (images, reflectance_map_targets, dem_targets, metas, w_targets, theta_targets, lro_metas) in enumerate(self.train_data):
            if use_profiler and is_main():
                batch_start = time.time()
            
            images = images.to(self.gpu_id)
            metas = metas.to(self.gpu_id)
            reflectance_map_targets = reflectance_map_targets.to(self.gpu_id)
            images = normalize_inputs(images, self.train_mean, self.train_std)
            w_targets = w_targets.to(self.gpu_id)
            theta_targets = theta_targets.to(self.gpu_id)
            dem_targets = dem_targets.to(self.gpu_id)

            source = images, metas
            targets = dem_targets, reflectance_map_targets, w_targets, theta_targets

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
        # print(f"Debug: epoch_loss_tensor={epoch_loss_tensor.item():.10f}, total_samples_tensor={total_samples_tensor.item()}")
        if not total_samples_tensor.item() == 0:
            global_avg_loss = epoch_loss_tensor.item() / total_samples_tensor.item()
        else:
            global_avg_loss = 0  # or some default value, but this would indicate a problem
    
        # Store loss on main process
        if is_main():
            self.train_loss_history.append(global_avg_loss)
            total_time = time.time() - t0
            self.train_timings.append(total_time)
            print(f"[GPU{self.gpu_id}] Epoch {epoch} | Loss: {global_avg_loss:.2e} | Samples: {total_samples_tensor.item()} | Time: {total_time:.2f}s | Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        if return_val:
            return global_avg_loss
        
        if is_main() and self.debug:
            print(f"--- Epoch {epoch} end ---")
            check_params_for_nans(self.model, tag=f"epoch_{epoch}_end")

    def _run_batch(self, source, targets, return_tensors: bool = False):
        self.optimizer.zero_grad()
        images, metas = source
        dem_targets, reflectance_map_targets, w_targets, theta_targets = targets
        device = images.device
        
        if is_main() and self.debug:
            print(f"Use_amp: {self.use_amp}, dtype: {self.dtype}")
            
        with autocast('cuda', enabled=self.use_amp, dtype=self.dtype):
            outputs = self.model(images, metas, target_size=dem_targets.shape[-2:])
            dem_outputs = outputs[:, 0:1, :, :]
            w_outputs = outputs[:, 1:2, :, :]
            theta_outputs = outputs[:, 2:3, :, :]

            if is_main() and self.debug:
                # 🔍 Check outputs for NaNs/Inf
                for name, t in [
                    ("dem_outputs", dem_outputs),
                    ("w_outputs", w_outputs),
                    ("theta_outputs", theta_outputs),
                ]:
                    if torch.isnan(t).any() or torch.isinf(t).any():
                        print(f"🚨 NaN/Inf i model-output '{name}'")
                        print(f"    min={t.min().item():.6e}, max={t.max().item():.6e}, mean={t.mean().item():.6e}")
                        raise RuntimeError(f"NaN/Inf i output {name}")

            total_loss_list = calculate_total_loss_multi_band(
                dem_outputs, dem_targets, reflectance_map_targets, metas, w_outputs, w_targets, theta_outputs, theta_targets,
                device=device,
                config=self.config,
                return_components=True
            )
        
        loss_mse, loss_grad, loss_refl, loss_w, loss_theta, total_loss = total_loss_list

        if is_main() and self.debug:
            # Check loss component values
            print(f"    Loss components: MSE={loss_mse.item():.6f}, Grad={loss_grad.item():.6f}, Refl={loss_refl.item():.6f}, w_band={loss_w.item():.6f}, theta_band={loss_theta.item():.6f}, Total={total_loss.item():.6f}")
            # 🔍 Check loss values
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print("🚨 NaN/Inf i total_loss i _run_batch")
                print(f"    MSE={loss_mse}, Grad={loss_grad}, Refl={loss_refl}, w={loss_w}, theta={loss_theta}")
                raise RuntimeError("NaN/Inf i total_loss")

        total_loss.backward()
        
        if is_main() and self.debug:
            # 🔍 Check gradients før clipping
            for name, p in self.model.named_parameters():
                if p.grad is not None:
                    if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                        print(f"🚨 NaN/Inf i gradient for '{name}' FØR clipping")
                        print(f"    grad min={p.grad.min().item():.6e}, max={p.grad.max().item():.6e}, mean={p.grad.mean().item():.6e}")
                        raise RuntimeError(f"NaN/Inf i grad for {name}")


        # Clip gradients – dette returnerer norm FØR clipping
        clipped_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                    self.config["GRAD_CLIP"])

        # Log hvornår clipping sker
        if is_main() and self.debug:
            if clipped_norm > self.config["GRAD_CLIP"]:
                print(f"✂️  Gradient clipping: norm before={clipped_norm:.4f} "
                    f"clip={self.config['GRAD_CLIP']}")

        # (optional) check for NaNs i clipped gradients
        if is_main() and self.debug:
            if torch.isnan(clipped_norm) or torch.isinf(clipped_norm):
                print(f"🚨 NaN/Inf in gradient norm *after clipping*: {clipped_norm}")
                raise RuntimeError("NaN detected in gradients!")
        
        self.optimizer.step()

        if is_main() and self.debug:
            # 🔍 Check parametre EFTER step
            for name, p in self.model.named_parameters():
                if torch.isnan(p).any() or torch.isinf(p).any():
                    print(f"🚨 NaN/Inf i parameter '{name}' EFTER optimizer.step()")
                    print(f"    param min={p.data.min().item():.6e}, max={p.data.max().item():.6e}, mean={p.data.mean().item():.6e}")
                    raise RuntimeError(f"NaN/Inf i param efter step: {name}")


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
        
        for images, reflectance_map_targets, dem_targets, metas, w_targets, theta_targets, lro_metas in self.val_data:
            images = images.to(self.gpu_id)
            metas = metas.to(self.gpu_id)
            reflectance_map_targets = reflectance_map_targets.to(self.gpu_id)
            dem_targets = dem_targets.to(self.gpu_id)
            w_targets = w_targets.to(self.gpu_id)
            theta_targets = theta_targets.to(self.gpu_id)
            images = normalize_inputs(images, self.train_mean, self.train_std)
            batch_size = images.size(0)
            
            with autocast('cuda', enabled=self.use_amp, dtype=self.dtype):
                outputs = self.model(images, metas, target_size=dem_targets.shape[-2:])
                dem_outputs = outputs[:, 0:1, :, :]
                w_outputs = outputs[:, 1:2, :, :]
                theta_outputs = outputs[:, 2:3, :, :]

                if dem_outputs.shape != dem_targets.shape:
                    raise ValueError(f"Shape mismatch between dem_outputs {dem_outputs.shape} and dem_targets {dem_targets.shape}")
                if w_outputs.shape != w_targets.shape:
                    raise ValueError(f"Shape mismatch between w_outputs {w_outputs.shape} and w_targets {w_targets.shape}")
                if theta_outputs.shape != theta_targets.shape:
                    raise ValueError(f"Shape mismatch between theta_outputs {theta_outputs.shape} and theta_targets {theta_targets.shape}")

                total_loss = calculate_total_loss_multi_band(
                    dem_outputs, dem_targets, reflectance_map_targets, metas, w_outputs, w_targets, theta_outputs, theta_targets,
                    device=self.gpu_id,
                    config=self.config,
                    return_components=False
                )
            
            val_loss += total_loss.item() * batch_size
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

        # print(f"Debug: val_loss_tensor={val_loss_tensor.item():.10f}, total_samples_tensor={total_samples_tensor.item()}")
        # Compute the true weighted average: total_val_loss / total_samples
        global_avg_val_loss = val_loss_tensor.item() / total_samples_tensor.item()

        
        if is_main():
            self.val_loss_history.append(global_avg_val_loss)
            val_time = time.time() - t0
            self.val_timings.append(val_time)
            print(f"[GPU{self.gpu_id}] Epoch {epoch} | Val Loss: {global_avg_val_loss:.2e} | Samples: {global_total_samples} | Time: {val_time:.2f}s | Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        self.model.train()  # Set back to training mode
        return global_avg_val_loss

    @torch.no_grad()
    def test(self, data_loader: DataLoader = None):
        """Run testing and return average loss and AME"""
        if self.test_data is None:
            if is_main():
                print("No test data provided. Skipping testing.")
            return None, None
        
        # Allow custom data loader for testing
        data_loader = self.test_data if data_loader is None else data_loader

        t0 = time.time()
        epoch = self.epochs_run
        if is_main():
            print(f"Evaluating on test dataset, after epoch {epoch}")
        
        self.model.eval()  # Set to evaluation mode
        test_loss = 0.0
        dem_total_ame = 0.0
        w_total_ame = 0.0
        theta_total_ame = 0.0
        total_samples = 0
        
        for images, reflectance_map_targets, dem_targets, metas, w_targets, theta_targets, lro_metas in data_loader:
            images = images.to(self.gpu_id)
            metas = metas.to(self.gpu_id)
            reflectance_map_targets = reflectance_map_targets.to(self.gpu_id)
            dem_targets = dem_targets.to(self.gpu_id)
            w_targets = w_targets.to(self.gpu_id)
            theta_targets = theta_targets.to(self.gpu_id)
            images = normalize_inputs(images, self.train_mean, self.train_std)
            
            batch_size = images.size(0)

            with autocast('cuda', enabled=self.use_amp, dtype=self.dtype):
                outputs = self.model(images, metas, target_size=dem_targets.shape[-2:])
                dem_outputs = outputs[:, 0:1, :, :]
                w_outputs = outputs[:, 1:2, :, :]
                theta_outputs = outputs[:, 2:3, :, :]
                # Calculate loss
                total_loss = calculate_total_loss_multi_band(
                    dem_outputs, dem_targets, reflectance_map_targets, metas, w_outputs, w_targets, theta_outputs, theta_targets,
                    device=self.gpu_id,
                    config=self.config,
                    return_components=False
                )
            
            # Calculate AME (Absolute Mean Error) for DEM, w band, and theta band
            dem_ame = torch.abs(dem_outputs - dem_targets).mean()
            w_ame = torch.abs(w_outputs - w_targets).mean()
            theta_ame = torch.abs(theta_outputs - theta_targets).mean()
            
            # Accumulate losses and AMEs
            test_loss += total_loss.item() * batch_size
            dem_total_ame += dem_ame.item() * batch_size
            w_total_ame += w_ame.item() * batch_size
            theta_total_ame += theta_ame.item() * batch_size
            total_samples += batch_size
        
        
        # Gather losses, AMEs, and sample counts from all GPUs
        test_loss_tensor = torch.tensor([test_loss], device=self.gpu_id)
        dem_ame_tensor = torch.tensor([dem_total_ame], device=self.gpu_id)
        w_ame_tensor = torch.tensor([w_total_ame], device=self.gpu_id)
        theta_ame_tensor = torch.tensor([theta_total_ame], device=self.gpu_id)
        total_samples_tensor = torch.tensor([total_samples], device=self.gpu_id)
        
        torch.distributed.all_reduce(test_loss_tensor, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(dem_ame_tensor, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(w_ame_tensor, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(theta_ame_tensor, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(total_samples_tensor, op=torch.distributed.ReduceOp.SUM)
        
        # Check if no GPU has test samples
        global_total_samples = total_samples_tensor.item()
        if global_total_samples == 0:
            if is_main():
                print(f"Warning: No test samples found. Skipping testing.")
            return None, None
        
        # Compute global weighted averages
        global_test_loss = test_loss_tensor.item() / total_samples_tensor.item()
        global_dem_ame = dem_ame_tensor.item() / total_samples_tensor.item()
        global_w_ame = w_ame_tensor.item() / total_samples_tensor.item()
        global_theta_ame = theta_ame_tensor.item() / total_samples_tensor.item()
        
        if is_main():
            test_time = time.time() - t0
            print(f"[GPU{self.gpu_id}] Epoch {epoch} | Test Loss: {global_test_loss:.2e} | DEM AME: {global_dem_ame:.6f} | W AME: {global_w_ame:.6f} | Theta AME: {global_theta_ame:.6f} | Samples: {global_total_samples} | Time: {test_time:.2f}s")
        
        # print(f"Debug: test_loss_tensor={test_loss_tensor.item():.10f}, total_samples_tensor={total_samples_tensor.item()}")
        self.model.train()  # Set back to training mode
        return global_test_loss, (global_dem_ame, global_w_ame, global_theta_ame)


def check_params_for_nans(model, tag=""):
    has_issue = False
    for name, p in model.named_parameters():
        if p is None:
            continue
        if torch.isnan(p).any() or torch.isinf(p).any():
            print(f"🚨 [{tag}] NaN/Inf i parameter: {name}")
            has_issue = True
    return has_issue



class Trainer_singleGPU:
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
        # Select device: prefer MPS, then CPU
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
            print("Using MPS device for training.")
        else:
            self.device = torch.device('cpu')
            print("Using CPU for training.")
        self.model = model.to(self.device)
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.optimizer = optimizer
        self.save_every = config["SAVE_EVERY"]
        self.epochs_run = 0
        self.train_loss_history = []
        self.val_loss_history = []
        self.train_timings = []
        self.val_timings = []
        self.config = config
        self.train_mean = train_mean
        self.train_std = train_std
        # No DDP or torch.compile for single GPU/CPU
        self.dtype = torch.bfloat16 if self.config.get("USE_BF16", False) else torch.float16
        self.use_amp = self.config.get("USE_AMP", False)
        self.scaler = GradScaler(self.device) if (self.use_amp and self.dtype == torch.float16 and self.device.type == 'cuda') else None
        self.snapshot_path = snapshot_path
        if os.path.exists(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(snapshot_path)

    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.state_dict(),
            "OPTIMIZER_STATE": self.optimizer.state_dict(),
            "EPOCHS_RUN": epoch,
            "TRAIN_LOSS_HISTORY": self.train_loss_history,
            "VAL_LOSS_HISTORY": self.val_loss_history,
            "TRAIN_TIMINGS": self.train_timings,
            "VAL_TIMINGS": self.val_timings,
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")
        # Save loss/timing CSVs as in multiGPU
        train_loss_file = self.snapshot_path.replace('snapshot.pt', 'train_losses.csv')
        with open(train_loss_file, 'w') as f:
            f.write("epoch,loss\n")
            for i, loss in enumerate(self.train_loss_history, start=0):
                f.write(f"{i},{loss}\n")
        val_loss_file = self.snapshot_path.replace('snapshot.pt', 'val_losses.csv')
        with open(val_loss_file, 'w') as f:
            f.write("epoch,loss\n")
            for i, loss in enumerate(self.val_loss_history, start=0):
                actual_epoch = i * self.save_every
                f.write(f"{actual_epoch},{loss}\n")
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
        loc = self.device
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        optimizer_state = snapshot["OPTIMIZER_STATE"]
        for state in optimizer_state['state'].values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)
        self.optimizer.load_state_dict(optimizer_state)
        if self.scaler is not None and "SCALER_STATE" in snapshot:
            self.scaler.load_state_dict(snapshot["SCALER_STATE"])
            print("Loaded GradScaler state")
        self.epochs_run = snapshot["EPOCHS_RUN"] + 1
        self.train_loss_history = snapshot["TRAIN_LOSS_HISTORY"]
        self.val_loss_history = snapshot["VAL_LOSS_HISTORY"]
        self.train_timings = snapshot["TRAIN_TIMINGS"]
        self.val_timings = snapshot["VAL_TIMINGS"]
        print(f"Found snapshot saved at epoch {self.epochs_run - 1}.")
        print(f"Resuming model from snapshot at Epoch {self.epochs_run}")
        self.model.train()

    def _run_epoch(self, epoch, return_val=False):
        t0 = time.time()
        print(f"Running epoch {epoch}")
        self.train_data.dataset.set_epoch(epoch) # Set epoch for dataset randomness
        epoch_loss = torch.zeros(1, dtype=torch.float32, device=self.device)
        total_samples = 0
        for batch_idx, (images, reflectance_maps, targets, metas) in enumerate(self.train_data):
            images = images.to(self.device)
            metas = metas.to(self.device)
            reflectance_maps = reflectance_maps.to(self.device)
            images = normalize_inputs(images, self.train_mean, self.train_std)
            targets = targets.to(self.device)
            batch_size = images.size(0)
            mean_batch_loss = self._run_batch((images, metas, reflectance_maps), targets, return_tensors=True)
            epoch_loss += mean_batch_loss.detach() * batch_size
            total_samples += batch_size
        epoch_loss_value = epoch_loss.item()
        global_avg_loss = epoch_loss_value / total_samples if total_samples > 0 else float('nan')
        self.train_loss_history.append(global_avg_loss)
        total_time = time.time() - t0
        self.train_timings.append(total_time)
        print(f"Epoch {epoch} | Loss: {global_avg_loss:.2e} | Samples: {total_samples} | Time: {total_time:.2f}s")
        if return_val:
            return global_avg_loss

    def _run_batch(self, source, targets, return_tensors: bool = False):
        self.optimizer.zero_grad()
        images, metas, reflectance_maps = source
        device = self.device
        # AMP only for CUDA, not MPS/CPU
        amp_enabled = self.use_amp and device.type == 'cuda'
        with autocast(device.type, enabled=amp_enabled, dtype=self.dtype):
            outputs = self.model(images, metas, target_size=targets.shape[-2:])
            total_loss = calculate_total_loss(
                outputs, targets, reflectance_maps, metas,
                device=device,
                camera_params=self.config["CAMERA_PARAMS"],
                hapke_params=self.config["HAPKE_KWARGS"],
                w_grad=self.config["W_GRAD"],
                w_refl=self.config["W_REFL"],
                w_mse=self.config["W_MSE"],
                height_norm=self.config["HEIGHT_NORMALIZATION"] + self.config["HEIGHT_NORMALIZATION_PM"],
                return_components=False
            )
        total_loss.backward()
        total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["GRAD_CLIP"])
        if total_norm > self.config["GRAD_CLIP"] * 0.8:
            print(f"⚠️ Large gradient norm: {total_norm:.4f} (clipped at {self.config['GRAD_CLIP']})")
        self.optimizer.step()
        if return_tensors:
            return total_loss
        else:
            return total_loss.item()

    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)
            if epoch % self.save_every == 0:
                self._validate(epoch)
                self._save_snapshot(epoch)

    @torch.no_grad()
    def _validate(self, epoch):
        if self.val_data is None:
            return None
        t0 = time.time()
        print(f"Running validation for epoch {epoch}")
        self.model.eval()
        val_loss = 0.0
        total_samples = 0
        for images, reflectance_maps, targets, metas in self.val_data:
            images = images.to(self.device)
            metas = metas.to(self.device)
            reflectance_maps = reflectance_maps.to(self.device)
            targets = targets.to(self.device)
            images = normalize_inputs(images, self.train_mean, self.train_std)
            batch_size = images.size(0)
            amp_enabled = self.use_amp and self.device.type == 'cuda'
            with autocast(self.device.type, enabled=amp_enabled, dtype=self.dtype):
                outputs = self.model(images, metas, target_size=targets.shape[-2:])
                loss = calculate_total_loss(
                    outputs, targets, reflectance_maps, metas,
                    device=self.device,
                    camera_params=self.config["CAMERA_PARAMS"],
                    hapke_params=self.config["HAPKE_KWARGS"],
                    w_grad=self.config["W_GRAD"],
                    w_refl=self.config["W_REFL"],
                    w_mse=self.config["W_MSE"],
                    height_norm=self.config["HEIGHT_NORMALIZATION"] + self.config["HEIGHT_NORMALIZATION_PM"],
                    return_components=False
                )
            val_loss += loss.item() * batch_size
            total_samples += batch_size
        global_total_samples = total_samples
        if global_total_samples == 0:
            print(f"Warning: No validation samples found. Skipping validation.")
            return None
        global_avg_val_loss = val_loss / global_total_samples
        self.val_loss_history.append(global_avg_val_loss)
        val_time = time.time() - t0
        self.val_timings.append(val_time)
        print(f"Epoch {epoch} | Val Loss: {global_avg_val_loss:.2e} | Samples: {global_total_samples} | Time: {val_time:.2f}s")
        self.model.train()
        return global_avg_val_loss

    @torch.no_grad()
    def test(self, data_loader: DataLoader = None):
        if self.test_data is None:
            print("No test data provided. Skipping testing.")
            return None, None
        data_loader = self.test_data if data_loader is None else data_loader
        t0 = time.time()
        epoch = self.epochs_run
        print(f"Evaluating on test dataset, after epoch {epoch}")
        self.model.eval()
        test_loss = 0.0
        total_ame = 0.0
        total_samples = 0
        for images, reflectance_maps, targets, metas in data_loader:
            images = images.to(self.device)
            metas = metas.to(self.device)
            reflectance_maps = reflectance_maps.to(self.device)
            targets = targets.to(self.device)
            images = normalize_inputs(images, self.train_mean, self.train_std)
            batch_size = images.size(0)
            amp_enabled = self.use_amp and self.device.type == 'cuda'
            with autocast(self.device.type, enabled=amp_enabled, dtype=self.dtype):
                outputs = self.model(images, metas, target_size=targets.shape[-2:])
                loss = calculate_total_loss(
                    outputs, targets, reflectance_maps, metas,
                    device=self.device,
                    camera_params=self.config["CAMERA_PARAMS"],
                    hapke_params=self.config["HAPKE_KWARGS"],
                    w_grad=self.config["W_GRAD"],
                    w_refl=self.config["W_REFL"],
                    w_mse=self.config["W_MSE"],
                    height_norm=self.config["HEIGHT_NORMALIZATION"] + self.config["HEIGHT_NORMALIZATION_PM"],
                    return_components=False
                )
            ame = torch.abs(outputs - targets).mean()
            test_loss += loss.item() * batch_size
            total_ame += ame.item() * batch_size
            total_samples += batch_size
        global_total_samples = total_samples
        if global_total_samples == 0:
            print(f"Warning: No test samples found. Skipping testing.")
            return None, None
        global_test_loss = test_loss / global_total_samples
        global_ame = total_ame / global_total_samples
        test_time = time.time() - t0
        print(f"Epoch {epoch} | Test Loss: {global_test_loss:.2e} | AME: {global_ame:.6f} | Samples: {global_total_samples} | Time: {test_time:.2f}s | Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.model.train()
        return global_test_loss, global_ame