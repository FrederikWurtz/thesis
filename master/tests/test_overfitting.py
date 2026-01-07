import os
import shutil
from master.configs.config_utils import load_config_file
import torch
from master.data_sim import dataset_io
from master.train.trainer_core import FluidDEMDataset, DEMDataset, train_epoch, validate_epoch
from master.train.train_utils import collate, compute_input_stats, get_device
from master.models.unet import UNet
from torch.utils.data import DataLoader

def main():
    run_dir = 'runs/run_3'
    config_path = os.path.join(run_dir, 'stats', 'config.ini')
    config = load_config_file(config_path)
    test_dir = os.path.join(run_dir, 'test')

    train_files = dataset_io.list_npz_files(test_dir)
    train_files = train_files[:10]  # Use only a few files to test overfitting
    config["TRAIN_FILES"] = train_files
    config["VAL_FILES"] = train_files
    config["BATCH_SIZE"] = 2

    pin_memory = False
    sampler = None
    device = get_device()
    model = UNet(in_channels=config["IMAGES_PER_DEM"], out_channels=1)
    model.to(device)
    train_ds = DEMDataset(train_files)
    train_loader = DataLoader(train_ds, batch_size=config["BATCH_SIZE"], shuffle=False, num_workers=config["NUM_WORKERS"], 
                                collate_fn=collate, pin_memory=pin_memory, sampler=sampler, persistent_workers=True, prefetch_factor=4)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["LR"])
    scaler = torch.cuda.amp.GradScaler(enabled=False)

    train_mean, train_std = compute_input_stats(train_loader, images_per_dem=config["IMAGES_PER_DEM"])
    hapke_params = config['HAPKE_KWARGS']
    w_mse = config['W_MSE']
    w_grad = config['W_GRAD']
    w_refl = config['W_REFL']
    camera_params = {'image_width': config["IMAGE_W"], 'image_height': config["IMAGE_H"], 'focal_length': config["FOCAL_LENGTH"]}

    for epoch in range(50):  # Train for a few epochs to test overfitting
        print(f'Starting epoch {epoch+1}/50')
        train_loss_val = train_epoch(model, train_loader, optimizer, scaler, device, train_mean, train_std, current_epoch=epoch, total_epochs=50, non_blocking=True,
                                    w_mse=w_mse, w_grad=w_grad, w_refl=w_refl, use_amp=False, hapke_params=hapke_params, camera_params=camera_params, autocast_device_type=device.type,)

        print(f'Epoch {epoch+1} training loss: {train_loss_val:.6f}')


        

    print('Overfitting test completed.')
    print('Final training loss:', train_loss_val)

if __name__ == "__main__":
    main()