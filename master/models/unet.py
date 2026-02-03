"""UNet model and FiLM/Meta encoder building blocks."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MetaEncoder(nn.Module):
    def __init__(self, meta_dim=5, hidden_dim=64, out_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(meta_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, meta):
        B, N, D = meta.shape
        meta_flat = meta.view(B * N, D)
        encoded = self.mlp(meta_flat)
        encoded = encoded.view(B, N, -1)
        return encoded.mean(dim=1)


class FiLMLayer(nn.Module):
    def __init__(self, feature_channels, cond_dim):
        super().__init__()
        self.gamma_fc = nn.Linear(cond_dim, feature_channels)
        self.beta_fc = nn.Linear(cond_dim, feature_channels)

    def forward(self, x, cond):
        gamma = self.gamma_fc(cond).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta_fc(cond).unsqueeze(-1).unsqueeze(-1)
        return gamma * x + beta


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels=None, out_channels=1, features=(64, 128, 256, 512), 
                 meta_dim=5, meta_hidden=64, meta_out=128, upsample_factor=4, w_range=None, theta_range=None):
        super().__init__()
        self.upsample_factor = upsample_factor
        self.w_range = w_range
        self.theta_range = theta_range
        self.encs = nn.ModuleList()
        self.pools = nn.ModuleList()
        ch = in_channels
        for f in features:
            self.encs.append(DoubleConv(ch, f))
            self.pools.append(nn.MaxPool2d(2))
            ch = f
        self.bottleneck_conv = DoubleConv(features[-1], features[-1] * 2)
        self.meta_encoder = MetaEncoder(meta_dim=meta_dim, hidden_dim=meta_hidden, out_dim=meta_out)
        self.film = FiLMLayer(feature_channels=features[-1] * 2, cond_dim=meta_out)
        self.upconvs = nn.ModuleList()
        self.decs = nn.ModuleList()
        ch = features[-1] * 2
        for f in reversed(features):
            self.upconvs.append(nn.ConvTranspose2d(ch, f, kernel_size=2, stride=2))
            self.decs.append(DoubleConv(ch, f))
            ch = f
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        self.final_upsample = nn.Upsample(scale_factor=upsample_factor, mode='bilinear', align_corners=False)

    def forward(self, x, meta, target_size=None):
        skips = []
        for enc, pool in zip(self.encs, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)
        x = self.bottleneck_conv(x)
        meta_encoded = self.meta_encoder(meta)
        x = self.film(x, meta_encoded)
        for up, dec, skip in zip(self.upconvs, self.decs, reversed(skips)):
            x = up(x)
            if x.size(-1) != skip.size(-1) or x.size(-2) != skip.size(-2):
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = dec(x)
        x = self.final_conv(x)
        if target_size is not None:
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        else:
            x = self.final_upsample(x)

        # Apply activation functions to constrain outputs to physical ranges
        if self.final_conv.out_channels == 3 and self.w_range is not None and self.theta_range is not None:
            dem = x[:, 0:1, :, :]  # No activation for DEM
            
            # Scale w to [w_min, w_max]
            w_raw = x[:, 1:2, :, :]
            w = torch.sigmoid(w_raw) * (self.w_range[1] - self.w_range[0]) + self.w_range[0]
            
            # Scale theta to [theta_min, theta_max]
            theta_raw = x[:, 2:3, :, :]
            theta = torch.sigmoid(theta_raw) * (self.theta_range[1] - self.theta_range[0]) + self.theta_range[0]
            
            x = torch.cat([dem, w, theta], dim=1)

        return x


__all__ = ['UNet']
