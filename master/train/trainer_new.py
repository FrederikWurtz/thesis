import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from master.train.trainer_core import FluidDEMDataset
import time

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

from master.configs.config_utils import load_config_file
from master.models.losses import calculate_total_loss
from master.models.unet import UNet


def ddp_setup():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    init_process_group(backend="nccl")

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        config: dict,
        snapshot_path: str,
    ) -> None:
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model = model.to(self.gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = config["SAVE_EVERY"]
        self.epochs_run = 0
        self.snapshot_path = snapshot_path
        if os.path.exists(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(snapshot_path)
        self.config = config
        self.model = DDP(self.model, device_ids=[self.gpu_id])

    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _run_epoch(self, epoch):
        t0 = time.time()
        print("Running epoch")
        #b_sz = len(next(iter(self.train_data))[0])
        #print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        for images, reflectance_maps, targets, metas in self.train_data:
            images = images.to(self.gpu_id)
            metas = metas.to(self.gpu_id)
            reflectance_maps = reflectance_maps.to(self.gpu_id)
            source = images, metas, reflectance_maps
            #source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            self._run_batch(source, targets)
        total_time = time.time() - t0
        print("Epoch took {:.2f}".format(total_time))

    def _run_batch(self, source, targets):
        print("Running batch")
        self.optimizer.zero_grad()
        images, metas, reflectance_maps = source
        device = images.device
        outputs = self.model(images, metas, target_size=targets.shape[-2:])
        loss = calculate_total_loss(outputs, targets, reflectance_maps, metas, device=device,
                                        camera_params=self.config["CAMERA_PARAMS"], hapke_params=self.config["HAPKE_KWARGS"],
                                        w_grad=self.config["W_GRAD"], w_refl=self.config["W_REFL"], w_mse=self.config["W_MSE"])
        loss.backward()
        self.optimizer.step()


    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)

def load_train_objs(config, fluid: bool = True):
    if fluid:
        train_set = FluidDEMDataset(config) # load your dataset
    if not fluid:
        raise NotImplementedError("Only fluid training is implemented in this script.")
    model = UNet(in_channels=config["IMAGES_PER_DEM"], out_channels=1)  # load your model
    optimizer = torch.optim.Adam(model.parameters(), lr=config["LR"], weight_decay=config["WEIGHT_DECAY"])
    return train_set, model, optimizer


def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset),
        num_workers=2,
        prefetch_factor=4
    )
