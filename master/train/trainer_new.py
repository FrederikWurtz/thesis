
import warnings

import torch

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from master.train.trainer_core import is_main
from master.train.train_utils import normalize_inputs
import time

from torch.nn.parallel import DistributedDataParallel as DDP
import os

from master.models.losses import calculate_total_loss, calculate_total_loss_multi_band
from master.train.checkpoints import save_file_as_ini, read_file_from_ini
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
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        config: dict,
        snapshot_path: str,
        train_mean: torch.Tensor = None,
        train_std: torch.Tensor = None,
        val_loader: DataLoader = None,
        test_loader: DataLoader = None,
    ) -> None:
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model = model.to(self.gpu_id)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        # Setup ReduceLROnPlateau scheduler using config defaults
        try:
            lr_factor = self.config.get("LR_FACTOR", 0.5)
        except Exception:
            lr_factor = 0.5
        try:
            lr_patience = self.config.get("LR_PATIENCE", 3)
        except Exception:
            lr_patience = 3
        try:
            lr_min = self.config.get("LR_MIN", 1e-7)
        except Exception:
            lr_min = 1e-7
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=lr_factor,
            patience=lr_patience,
            min_lr=lr_min,
            verbose=(is_main() and self.debug),
        )
        # Keep a copy of last known LRs to detect changes
        self._last_lrs = [pg.get('lr', None) for pg in self.optimizer.param_groups]
        # Path to record LR-change epochs
        self.lr_changes_ini = os.path.join(os.path.dirname(self.snapshot_path), 'lr_changes.ini')
        self.save_every = config["SAVE_EVERY"]
        self.epochs_run = 0
        self.train_loss_history = []  # Track losses
        self.val_loss_history = []    # Track validation losses
        self.train_timings = []
        self.val_timings = []
        self.config = config
        self.train_mean = train_mean
        self.train_std = train_std
        self.debug = config["DEBUG"]
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
        self.train_loader.sampler.set_epoch(epoch)

        # Also set epoch in dataset to ensure deterministic data generation
        self.train_loader.dataset.set_epoch(epoch)

        for batch_idx, (images, reflectance_maps, targets, metas) in enumerate(self.train_loader):
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
            outputs = self.model(images, metas)
            total_loss = calculate_total_loss(
                outputs, targets, reflectance_maps, metas, 
                device=device,
                camera_params=self.config["CAMERA_PARAMS"], 
                hapke_params=self.config["HAPKE_KWARGS"],
                w_grad=self.config["W_GRAD"], 
                w_refl=self.config["W_REFL"], 
                w_mse=self.config["W_MSE"],
                height_norm=self.config["HEIGHT_NORMALIZATION"] + self.config["HEIGHT_NORMALIZATION_PM"], # the maximum possible height for normalization
                return_components=False,
                debug=self.debug
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
        if self.val_loader is None:
            return None
            
        t0 = time.time()
        if is_main():
            print(f"Running validation for epoch {epoch}")
        
        self.model.eval()  # Set to evaluation mode
        val_loss = 0.0
        total_samples = 0
        
        for images, reflectance_maps, targets, metas in self.val_loader:
            images = images.to(self.gpu_id)
            metas = metas.to(self.gpu_id)
            reflectance_maps = reflectance_maps.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            images = normalize_inputs(images, self.train_mean, self.train_std)
            batch_size = images.size(0)
            
            with autocast('cuda', enabled=self.use_amp, dtype=self.dtype):
                outputs = self.model(images, metas)
                loss = calculate_total_loss(
                    outputs, targets, reflectance_maps, metas, 
                    device=self.gpu_id,
                    camera_params=self.config["CAMERA_PARAMS"], 
                    hapke_params=self.config["HAPKE_KWARGS"],
                    w_grad=self.config["W_GRAD"], 
                    w_refl=self.config["W_REFL"], 
                    w_mse=self.config["W_MSE"],
                    height_norm=self.config["HEIGHT_NORMALIZATION"] + self.config["HEIGHT_NORMALIZATION_PM"], # the maximum possible height for normalization
                    return_components=False,
                    debug=self.debug
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
        if self.test_loader is None:
            if is_main():
                print("No test data provided. Skipping testing.")
            return None, None
        
        # Allow custom data loader for testing
        data_loader = self.test_loader if data_loader is None else data_loader

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
                outputs = self.model(images, metas)
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
                    return_components=False,
                    debug=self.debug
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

class Trainer_multiGPU_multi_band:
    def __init__(
        self,
        model: torch.nn.Module = None,
        train_loader: DataLoader = None,
        optimizer: torch.optim.Optimizer = None,
        config: dict = None,
        snapshot_path: str = None,
        train_mean: torch.Tensor = None,
        train_std: torch.Tensor = None,
        val_loader: DataLoader = None,
        test_loader: DataLoader = None,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
    ) -> None:
        if any(param is None for param in [model, train_loader, optimizer, config, snapshot_path, train_mean, train_std]):
            raise ValueError("Model, train_loader, optimizer, config, snapshot_path, train_mean, and train_std must all be provided for Trainer_multiGPU_multi_band.")
        
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model = model.to(self.gpu_id)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
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
            print("🔥 Compiling model with torch.compile() on single GPU. Will take some time...")
            self.model = torch.compile(self.model, mode='reduce-overhead')
        else:
            if is_main():
                print("⚠️ Skipping torch.compile() in multi-GPU mode to avoid known compile issues on multiple GPUs.")
        self.dtype = torch.bfloat16 if self.config["USE_BF16"] else torch.float16
        self.use_amp = self.config["USE_AMP"]
        self.scaler = GradScaler('cuda') if (self.use_amp and self.dtype == torch.float16) else None
        self.snapshot_path = snapshot_path # Path to save/load snapshots
        self.last_snapshot_path = snapshot_path
        self.best_snapshot_path = snapshot_path.replace("snapshot.pt", "snapshot_best.pt")
        # Keep a copy of last known LRs to detect changes
        self._last_lrs = [pg.get('lr', None) for pg in self.optimizer.param_groups]
        # Path to record LR-change epochs
        self.lr_changes_ini = os.path.join(os.path.dirname(self.snapshot_path), 'lr_changes.ini')
        self.best_val_loss = float("inf")
        if os.path.exists(snapshot_path): 
            if is_main():
                print("Loading snapshot")
            self._load_snapshot(snapshot_path) # Then, after DDP wrapping, load snapshot if it exists
            
        self.debug = self.config["DEBUG"]

    def _save_snapshot(self, epoch, val_loss=None):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "OPTIMIZER_STATE": self.optimizer.state_dict(),  # Save optimizer state
            "SCHEDULER_STATE": self.scheduler.state_dict() if hasattr(self, 'scheduler') and self.scheduler is not None else None,
            "EPOCHS_RUN": epoch,
            "TRAIN_LOSS_HISTORY": self.train_loss_history,  # Save loss history
            "VAL_LOSS_HISTORY": self.val_loss_history,  # Save validation loss history
            "TRAIN_TIMINGS": self.train_timings,
            "VAL_TIMINGS": self.val_timings,
            "BEST_VAL_LOSS": self.best_val_loss,
        }

        # always save the latest snapshot
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")
        
        # Save best only if improved
        if val_loss is not None and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            snapshot["BEST_VAL_LOSS"] = self.best_val_loss
            torch.save(snapshot, self.best_snapshot_path)
            print(f"Epoch {epoch} | New best val loss {val_loss:.2e} | Best snapshot saved at {self.best_snapshot_path}")
        
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
        # restore scheduler state if present
        if hasattr(self, 'scheduler') and self.scheduler is not None and "SCHEDULER_STATE" in snapshot and snapshot["SCHEDULER_STATE"] is not None:
            try:
                self.scheduler.load_state_dict(snapshot["SCHEDULER_STATE"])
                if is_main() and self.debug:
                    print("Loaded scheduler state from snapshot")
            except Exception as e:
                print(f"Warning: Failed to load scheduler state: {e}")

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
        self.best_val_loss = snapshot["BEST_VAL_LOSS"]
        self.model.train()  # Set back to training mode

    def _record_lr_change(self, epoch, new_lrs):
        """Append (epoch, new_lrs) to the LR changes INI file as a list entry.

        Each entry is stored as a tuple: (epoch, (lr1, lr2, ...)).
        """
        try:
            existing = read_file_from_ini(self.lr_changes_ini, ftype=list)
        except FileNotFoundError:
            existing = []

        # Normalize values to basic Python types
        try:
            lr_tuple = tuple(float(x) for x in new_lrs)
        except Exception:
            # fallback: stringify
            lr_tuple = tuple(str(x) for x in new_lrs)

        entry = (int(epoch), lr_tuple)
        existing.append(entry)

        # save back to INI (save_file_as_ini handles lists)
        save_file_as_ini(existing, self.lr_changes_ini)
        print(f"Epoch {epoch} | LR change detected. Recorded new LRs: {lr_tuple} in {self.lr_changes_ini}")


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
        self.train_loader.sampler.set_epoch(epoch)

        # Also set epoch in dataset to ensure deterministic data generation
        self.train_loader.dataset.set_epoch(epoch)

        for batch_idx, (images, reflectance_map_targets, dem_targets, metas, w_targets, theta_targets, lro_metas) in enumerate(self.train_loader):
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
            outputs = self.model(images, metas)
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
                return_components=True,
                debug=self.debug
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
                val_loss = self._validate(epoch)
                # Step scheduler on validation loss if available
                if hasattr(self, 'scheduler') and self.scheduler is not None and val_loss is not None:
                    try:
                        self.scheduler.step(val_loss)
                        # detect LR reduction (compare first param-group lr by default)
                        new_lrs = [group.get('lr', None) for group in self.optimizer.param_groups]
                        # consider change if any lr strictly decreased
                        decreased = any((nl is not None and ll is not None and nl < ll - 1e-12) for nl, ll in zip(new_lrs, self._last_lrs))
                        if decreased:
                            try:
                                self._record_lr_change(epoch, new_lrs)
                            except Exception as e:
                                print(f"Warning: failed to record lr change: {e}")
                        # update last lr snapshot
                        self._last_lrs = new_lrs
                        if is_main() and self.debug:
                            # log current LR(s)
                            lrs = [group.get('lr', None) for group in self.optimizer.param_groups]
                            print(f"Scheduler stepped. Current LRs: {lrs}")
                    except Exception as e:
                        print(f"Warning: scheduler step failed: {e}")
                # But only GPU 0 saves the snapshot
                if self.gpu_id == 0:
                    self._save_snapshot(epoch, val_loss=val_loss)

        if use_profiler and is_main():
            prof.stop()
            print(f"Profiler trace saved to: {os.path.dirname(self.snapshot_path)}/../profiler")


    @torch.no_grad()
    def _validate(self, epoch):
        """Run validation and return average loss"""
        if self.val_loader is None:
            return None
            
        t0 = time.time()
        if is_main():
            print(f"Running validation for epoch {epoch}")
        
        self.model.eval()  # Set to evaluation mode
        val_loss = 0.0
        total_samples = 0
        
        for images, reflectance_map_targets, dem_targets, metas, w_targets, theta_targets, lro_metas in self.val_loader:
            images = images.to(self.gpu_id)
            metas = metas.to(self.gpu_id)
            reflectance_map_targets = reflectance_map_targets.to(self.gpu_id)
            dem_targets = dem_targets.to(self.gpu_id)
            w_targets = w_targets.to(self.gpu_id)
            theta_targets = theta_targets.to(self.gpu_id)
            images = normalize_inputs(images, self.train_mean, self.train_std)
            batch_size = images.size(0)
            
            with autocast('cuda', enabled=self.use_amp, dtype=self.dtype):
                outputs = self.model(images, metas)
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
                    return_components=False,
                    debug=self.debug
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
        if self.test_loader is None:
            if is_main():
                print("No test data provided. Skipping testing.")
            return None, None
        
        # Allow custom data loader for testing
        data_loader = self.test_loader if data_loader is None else data_loader

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
                outputs = self.model(images, metas)
                dem_outputs = outputs[:, 0:1, :, :]
                w_outputs = outputs[:, 1:2, :, :]
                theta_outputs = outputs[:, 2:3, :, :]
                # Calculate loss
                total_loss = calculate_total_loss_multi_band(
                    dem_outputs, dem_targets, reflectance_map_targets, metas, w_outputs, w_targets, theta_outputs, theta_targets,
                    device=self.gpu_id,
                    config=self.config,
                    return_components=False,
                    debug=self.debug
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
        model: torch.nn.Module = None,
        train_loader: DataLoader = None,
        optimizer: torch.optim.Optimizer = None,
        config: dict = None,
        snapshot_path: str = None,
        train_mean: torch.Tensor = None,
        train_std: torch.Tensor = None,
        val_loader: DataLoader = None,
        test_loader: DataLoader = None,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
    ) -> None:
        # Select device: prefer MPS, then CPU
        if any(param is None for param in [model, train_loader, optimizer, config, snapshot_path, train_mean, train_std]):
            raise ValueError("Model, train_loader, optimizer, config, snapshot_path, train_mean, and train_std must all be provided for Trainer_singleGPU.")
        
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
            print("Using MPS device for training.")
        else:
            self.device = torch.device('cpu')
            print("Using CPU for training.")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.config = config
        self.debug = self.config["DEBUG"]
        self.scheduler = scheduler
        self.save_every = config["SAVE_EVERY"]
        self.epochs_run = 0
        self.train_loss_history = []
        self.val_loss_history = []
        self.train_timings = []
        self.val_timings = []
        self.train_mean = train_mean.to(self.device)
        self.train_std = train_std.to(self.device)
        # No DDP or torch.compile for single GPU/CPU
        self.dtype = torch.bfloat16 if self.config.get("USE_BF16", False) else torch.float16
        self.use_amp = self.config.get("USE_AMP", False)
        self.scaler = GradScaler(self.device) if (self.use_amp and self.dtype == torch.float16 and self.device.type == 'cuda') else None
        self.snapshot_path = snapshot_path
        self.last_snapshot_path = snapshot_path
        self.best_snapshot_path = snapshot_path.replace("snapshot.pt", "snapshot_best.pt")
        # Keep a copy of last known LRs to detect changes
        self._last_lrs = [pg.get('lr', None) for pg in self.optimizer.param_groups]
        # Path to record LR-change epochs
        self.lr_changes_ini = os.path.join(os.path.dirname(self.snapshot_path), 'lr_changes.ini')
        self.best_val_loss = float("inf")
        self.batch_number = 0
        if os.path.exists(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(snapshot_path)

    def _save_snapshot(self, epoch, val_loss=None):
        snapshot = {
            "MODEL_STATE": self.model.state_dict(),
            "OPTIMIZER_STATE": self.optimizer.state_dict(),  # Save optimizer state
            "SCHEDULER_STATE": self.scheduler.state_dict() if hasattr(self, 'scheduler') and self.scheduler is not None else None,
            "EPOCHS_RUN": epoch,
            "TRAIN_LOSS_HISTORY": self.train_loss_history,  # Save loss history
            "VAL_LOSS_HISTORY": self.val_loss_history,  # Save validation loss history
            "TRAIN_TIMINGS": self.train_timings,
            "VAL_TIMINGS": self.val_timings,
            "BEST_VAL_LOSS": self.best_val_loss,
        }

        # always save the latest snapshot
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")
        
        # Save best only if improved
        if val_loss is not None and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            snapshot["BEST_VAL_LOSS"] = self.best_val_loss
            torch.save(snapshot, self.best_snapshot_path)
            print(f"Epoch {epoch} | New best val loss {val_loss:.3e} | Best snapshot saved at {self.best_snapshot_path}")
        
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
        loc = self.device
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        optimizer_state = snapshot["OPTIMIZER_STATE"]
        for state in optimizer_state['state'].values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)
        self.optimizer.load_state_dict(optimizer_state)
        # restore scheduler state if present
        if hasattr(self, 'scheduler') and self.scheduler is not None and "SCHEDULER_STATE" in snapshot and snapshot["SCHEDULER_STATE"] is not None:
            try:
                self.scheduler.load_state_dict(snapshot["SCHEDULER_STATE"])
                if is_main() and self.debug:
                    print("Loaded scheduler state from snapshot")
            except Exception as e:
                print(f"Warning: Failed to load scheduler state: {e}")
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
        self.best_val_loss = snapshot["BEST_VAL_LOSS"]
        self.model.train()

    def _record_lr_change(self, epoch, new_lrs):
        """Append (epoch, new_lrs) to the LR changes INI file as a list entry.

        Each entry is stored as a tuple: (epoch, (lr1, lr2, ...)).
        """
        try:
            existing = read_file_from_ini(self.lr_changes_ini, ftype=list)
        except FileNotFoundError:
            existing = []

        # Normalize values to basic Python types
        try:
            lr_tuple = tuple(float(x) for x in new_lrs)
        except Exception:
            # fallback: stringify
            lr_tuple = tuple(str(x) for x in new_lrs)

        entry = (int(epoch), lr_tuple)
        existing.append(entry)

        # save back to INI (save_file_as_ini handles lists)
        save_file_as_ini(existing, self.lr_changes_ini)
        print(f"Epoch {epoch} | LR change detected. Recorded new LRs: {lr_tuple} in {self.lr_changes_ini}")

    def _run_epoch(self, epoch, return_val=False):
        t0 = time.time()
        print(f"Running epoch {epoch}")
        self.train_loader.dataset.set_epoch(epoch)  # Set epoch for dataset randomness
        epoch_loss = torch.zeros(1, dtype=torch.float32, device=self.device)
        total_samples = 0

        pbar = tqdm(self.train_loader, desc="Training", unit="batch", dynamic_ncols=True, leave=False)

        for images, reflectance_map_targets, dem_targets, metas, w_targets, theta_targets, lro_metas in pbar:
            images = images.to(self.device)
            metas = metas.to(self.device)
            reflectance_map_targets = reflectance_map_targets.to(self.device)
            dem_targets = dem_targets.to(self.device)
            w_targets = w_targets.to(self.device)
            theta_targets = theta_targets.to(self.device)

            images = normalize_inputs(images, self.train_mean, self.train_std)

            batch_size = images.size(0)
            mean_batch_loss, loss_parts = self._run_batch(
                (images, metas),
                (dem_targets, reflectance_map_targets, w_targets, theta_targets),
                return_tensors=True,
                pbar=pbar
            )
            epoch_loss += mean_batch_loss.detach() * batch_size
            total_samples += batch_size

        loss_mse, loss_grad, loss_refl, loss_w, loss_theta = loss_parts
        epoch_loss_value = epoch_loss.item()
        global_avg_loss = epoch_loss_value / total_samples if total_samples > 0 else float("nan")
        self.train_loss_history.append(global_avg_loss)
        total_time = time.time() - t0
        self.train_timings.append(total_time)
        print(f"Epoch {epoch} | Train Loss: {global_avg_loss:.3e} | Samples: {total_samples} | Time: {total_time:.2f}s" f"| Loss parts - MSE: {loss_mse.item()*self.config['W_MSE']:.3f}, Grad: {loss_grad.item()*self.config['W_GRAD']:.3f}, Refl: {loss_refl.item()*self.config['W_REFL']:.3f}, w: {loss_w.item()*self.config['W_W']:.3f}, theta: {loss_theta.item()*self.config['W_THETA']:.3f}")

        if return_val:
            return global_avg_loss

    def _run_batch(self, source, targets, return_tensors: bool = False, pbar=None):
        self.optimizer.zero_grad()
        images, metas = source
        dem_targets, reflectance_map_targets, w_targets, theta_targets = targets
        device = self.device

        amp_enabled = self.use_amp and device.type == "cuda"
        with autocast(device.type, enabled=amp_enabled, dtype=self.dtype):
            outputs = self.model(images, metas)
            dem_outputs = outputs[:, 0:1, :, :]
            w_outputs = outputs[:, 1:2, :, :]
            theta_outputs = outputs[:, 2:3, :, :]

            total_loss_list = calculate_total_loss_multi_band(
                dem_outputs,
                dem_targets,
                reflectance_map_targets,
                metas,
                w_outputs,
                w_targets,
                theta_outputs,
                theta_targets,
                device=device,
                config=self.config,
                return_components=True,
                debug=self.debug
            )

        
        if self.debug:
            loss_mse, loss_grad, loss_refl, loss_w, loss_theta, total_loss = total_loss_list
            for name, loss in [
                ("mse", loss_mse),
                ("grad", loss_grad),
                ("refl", loss_refl),
                ("w", loss_w),
                ("theta", loss_theta),
                ("total", total_loss)
            ]:
                loss.backward(retain_graph=True)
                first_grad = self.model.encs[0].conv1.weight.grad  # Check gradient of first conv layer as a proxy
                print(name, "Has NaN:", torch.isnan(first_grad).any(), "Has Inf:", torch.isinf(first_grad).any())
                self.optimizer.zero_grad()  # Reset gradients for next loss component
        
        loss_mse, loss_grad, loss_refl, loss_w, loss_theta, total_loss = total_loss_list
        total_loss.backward()
        
        if self.batch_number % 5 == 0:
            pbar.set_postfix(
                            mse=f"{loss_mse.item()*self.config['W_MSE']:.3f}",
                            grad=f"{loss_grad.item()*self.config['W_GRAD']:.3f}",
                            refl=f"{loss_refl.item()*self.config['W_REFL']:.3f}",
                            w=f"{loss_w.item()*self.config['W_W']:.3f}",
                            theta=f"{loss_theta.item()*self.config['W_THETA']:.3f}",
                            total=f"{total_loss.item():.3f}",
                        )        
        self.batch_number += 1
        
        if self.debug:
            for name, param in self.model.named_parameters():
                if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                    raise RuntimeError(f"NaN/Inf grad in {name}")

        total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["GRAD_CLIP"])
        
        if self.debug:
            for name, param in self.model.named_parameters():
                if torch.isnan(param).any() or torch.isinf(param).any():
                    raise RuntimeError(f"NaN/Inf param after backward in {name}")
    
        if total_norm > self.config["GRAD_CLIP"]:
            tqdm.write(f"⚠️ Large gradient norm: {total_norm:.4f} (clipped at {self.config['GRAD_CLIP']})")
        
        self.optimizer.step()
        
        if self.debug:
            for name, param in self.model.named_parameters():
                if torch.isnan(param).any() or torch.isinf(param).any():
                    raise RuntimeError(f"NaN/Inf param after optimizer.step in {name}")

        if return_tensors:
            return total_loss, (loss_mse, loss_grad, loss_refl, loss_w, loss_theta)
        else:
            return total_loss.item(), (loss_mse.item(), loss_grad.item(), loss_refl.item(), loss_w.item(), loss_theta.item())

    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)
            if epoch % self.save_every == 0:
                val_loss = self._validate(epoch)
                # Step scheduler on validation loss if available
                if hasattr(self, 'scheduler') and self.scheduler is not None and val_loss is not None:
                    try:
                        self.scheduler.step(val_loss)
                        # detect LR reduction (compare first param-group lr by default)
                        new_lrs = [group.get('lr', None) for group in self.optimizer.param_groups]
                        # consider change if any lr strictly decreased
                        decreased = any((nl is not None and ll is not None and nl < ll - 1e-12) for nl, ll in zip(new_lrs, self._last_lrs))
                        if decreased:
                            try:
                                self._record_lr_change(epoch, new_lrs)
                            except Exception as e:
                                print(f"Warning: failed to record lr change: {e}")
                        # update last lr snapshot
                        self._last_lrs = new_lrs
                        if is_main() and self.debug:
                            # log current LR(s)
                            lrs = [group.get('lr', None) for group in self.optimizer.param_groups]
                            print(f"Scheduler stepped. Current LRs: {lrs}")
                    except Exception as e:
                        print(f"Warning: scheduler.step failed: {e}")
                self._save_snapshot(epoch, val_loss=val_loss)
            self.epochs_run += 1
        self.epochs_run = max_epochs  # Ensure epochs_run is updated to max_epochs at the end

    @torch.no_grad()
    def _validate(self, epoch):
        if self.val_loader is None:
            return None

        t0 = time.time()
        print(f"Running validation for epoch {epoch}")
        self.model.eval()
        val_loss = 0.0
        total_samples = 0

        for images, reflectance_map_targets, dem_targets, metas, w_targets, theta_targets, lro_metas in tqdm(self.val_loader, desc="Validation", unit="batch", dynamic_ncols=True, leave=False):
            images = images.to(self.device)
            metas = metas.to(self.device)
            reflectance_map_targets = reflectance_map_targets.to(self.device)
            dem_targets = dem_targets.to(self.device)
            w_targets = w_targets.to(self.device)
            theta_targets = theta_targets.to(self.device)

            images = normalize_inputs(images, self.train_mean, self.train_std)
            batch_size = images.size(0)

            amp_enabled = self.use_amp and self.device.type == "cuda"
            with autocast(self.device.type, enabled=amp_enabled, dtype=self.dtype):
                outputs = self.model(images, metas)
                dem_outputs = outputs[:, 0:1, :, :]
                w_outputs = outputs[:, 1:2, :, :]
                theta_outputs = outputs[:, 2:3, :, :]

                if dem_outputs.shape != dem_targets.shape:
                    raise ValueError(f"Shape mismatch between dem_outputs {dem_outputs.shape} and dem_targets {dem_targets.shape}")
                if w_outputs.shape != w_targets.shape:
                    raise ValueError(f"Shape mismatch between w_outputs {w_outputs.shape} and w_targets {w_targets.shape}")
                if theta_outputs.shape != theta_targets.shape:
                    raise ValueError(f"Shape mismatch between theta_outputs {theta_outputs.shape} and theta_targets {theta_targets.shape}")

                total_loss_list = calculate_total_loss_multi_band(
                    dem_outputs,
                    dem_targets,
                    reflectance_map_targets,
                    metas,
                    w_outputs,
                    w_targets,
                    theta_outputs,
                    theta_targets,
                    device=self.device,
                    config=self.config,
                    return_components=True,
                    debug=self.debug
                )

            loss_mse, loss_grad, loss_refl, loss_w, loss_theta, total_loss = total_loss_list
            
            val_loss += total_loss.item() * batch_size
            total_samples += batch_size

        if total_samples == 0:
            print("Warning: No validation samples found. Skipping validation.")
            return None

        global_avg_val_loss = val_loss / total_samples
        self.val_loss_history.append(global_avg_val_loss)
        val_time = time.time() - t0
        self.val_timings.append(val_time)
        print(f"Epoch {epoch} | Val Loss: {global_avg_val_loss:.3e} | Samples: {total_samples} | Time: {val_time:.2f}s | Loss parts - MSE: {loss_mse.item()*self.config['W_MSE']:.3f}, Grad: {loss_grad.item()*self.config['W_GRAD']:.3f}, Refl: {loss_refl.item()*self.config['W_REFL']:.3f}, w: {loss_w.item()*self.config['W_W']:.3f}, theta: {loss_theta.item()*self.config['W_THETA']:.3f}")

        self.model.train()
        return global_avg_val_loss
    
    @torch.no_grad()
    def test(self, data_loader: DataLoader = None):
        if self.test_loader is None:
            print("No test data provided. Skipping testing.")
            return None, None

        data_loader = self.test_loader if data_loader is None else data_loader
        t0 = time.time()
        epoch = self.epochs_run
        print(f"Evaluating on test dataset, after epoch {epoch}")

        self.model.eval()
        test_loss = 0.0
        dem_total_ame = 0.0
        w_total_ame = 0.0
        theta_total_ame = 0.0
        total_samples = 0

        for images, reflectance_map_targets, dem_targets, metas, w_targets, theta_targets, lro_metas in data_loader:
            images = images.to(self.device)
            metas = metas.to(self.device)
            reflectance_map_targets = reflectance_map_targets.to(self.device)
            dem_targets = dem_targets.to(self.device)
            w_targets = w_targets.to(self.device)
            theta_targets = theta_targets.to(self.device)

            images = normalize_inputs(images, self.train_mean, self.train_std)
            batch_size = images.size(0)

            amp_enabled = self.use_amp and self.device.type == "cuda"
            with autocast(self.device.type, enabled=amp_enabled, dtype=self.dtype):
                outputs = self.model(images, metas)
                dem_outputs = outputs[:, 0:1, :, :]
                w_outputs = outputs[:, 1:2, :, :]
                theta_outputs = outputs[:, 2:3, :, :]

                loss = calculate_total_loss_multi_band(
                    dem_outputs,
                    dem_targets,
                    reflectance_map_targets,
                    metas,
                    w_outputs,
                    w_targets,
                    theta_outputs,
                    theta_targets,
                    device=self.device,
                    config=self.config,
                    return_components=False,
                    debug=self.debug
                )

            dem_ame = torch.abs(dem_outputs - dem_targets).mean()
            w_ame = torch.abs(w_outputs - w_targets).mean()
            theta_ame = torch.abs(theta_outputs - theta_targets).mean()

            test_loss += loss.item() * batch_size
            dem_total_ame += dem_ame.item() * batch_size
            w_total_ame += w_ame.item() * batch_size
            theta_total_ame += theta_ame.item() * batch_size
            total_samples += batch_size

        if total_samples == 0:
            print("Warning: No test samples found. Skipping testing.")
            return None, None

        global_test_loss = test_loss / total_samples
        global_dem_ame = dem_total_ame / total_samples
        global_w_ame = w_total_ame / total_samples
        global_theta_ame = theta_total_ame / total_samples

        test_time = time.time() - t0
        print(
            f"Epoch {epoch} | Test Loss: {global_test_loss:.2e} | "
            f"DEM AME: {global_dem_ame:.6f} | W AME: {global_w_ame:.6f} | "
            f"Theta AME: {global_theta_ame:.6f} | Samples: {total_samples} | Time: {test_time:.2f}s | "
            f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}"
        )

        self.model.train()
        return global_test_loss, (global_dem_ame, global_w_ame, global_theta_ame)