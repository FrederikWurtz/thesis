import torch
from master.train import UNet


def test_unet_forward_smoke():
    model = UNet(in_channels=5, out_channels=1)
    model.eval()
    B = 1
    C = 5
    H = 64
    W = 64
    images = torch.randn(B, C, H, W)
    # meta: (B, 5 images, 5 values)
    meta = torch.zeros(B, 5, 5)
    with torch.no_grad():
        out = model(images, meta, target_size=(H, W))
    assert out.shape == (B, 1, H, W)

if __name__ == "__main__":
    test_unet_forward_smoke()
    print("test_unet_forward_smoke passed.")
