"""UNet model and FiLM/Meta encoder building blocks."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


# class MetaEncoder(nn.Module):
#     def __init__(self, meta_dim=5, hidden_dim=64, out_dim=128):
#         super().__init__()
#         self.mlp = nn.Sequential(
#             nn.Linear(meta_dim, hidden_dim),
#             nn.ReLU(inplace=True),
#             nn.Linear(hidden_dim, out_dim),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, meta):
#         B, N, D = meta.shape
#         meta_flat = meta.view(B * N, D)
#         encoded = self.mlp(meta_flat)
#         encoded = encoded.view(B, N, -1)
#         return encoded.mean(dim=1) # Average over the N x 5 meta numbers - not good!

class MetaEncoder(nn.Module):
    def __init__(self, meta_dim=5, num_images=5, hidden_dim=64, out_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(meta_dim * num_images, hidden_dim),  # 5*5=25 -> 64
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),  # 64 -> 128
            nn.ReLU(inplace=True)
        )

    def forward(self, meta):
        B, N, D = meta.shape  # (B, 5, 5)
        meta_flat = meta.view(B, -1)  # (B, 25)
        encoded = self.mlp(meta_flat)  # (B, 128)
        return encoded


class FiLMLayer(nn.Module):
    def __init__(self, feature_channels, cond_dim):
        super().__init__()
        self.gamma_fc = nn.Linear(cond_dim, feature_channels)
        self.beta_fc = nn.Linear(cond_dim, feature_channels)

    def forward(self, x, cond):
        gamma = self.gamma_fc(cond).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta_fc(cond).unsqueeze(-1).unsqueeze(-1)
        return (1 + gamma) * x + beta

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, norm="group", num_groups=8):
        super().__init__()

        if norm == "batch":
            norm_layer1 = nn.BatchNorm2d(out_ch)
            norm_layer2 = nn.BatchNorm2d(out_ch)
        elif norm == "group":
            # Sørg for at num_groups dividerer out_ch
            assert out_ch % num_groups == 0, \
                f"out_ch={out_ch} skal være delelig med num_groups={num_groups} for GroupNorm"
            norm_layer1 = nn.GroupNorm(num_groups, out_ch)
            norm_layer2 = nn.GroupNorm(num_groups, out_ch)
        elif norm == "instance":
            norm_layer1 = nn.InstanceNorm2d(out_ch, affine=True)
            norm_layer2 = nn.InstanceNorm2d(out_ch, affine=True)
        elif norm is None:
            norm_layer1 = nn.Identity()
            norm_layer2 = nn.Identity()
        else:
            raise ValueError(f"Ukendt norm-type: {norm}")

        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            norm_layer1,
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            norm_layer2,
            nn.ReLU(inplace=True),
        )


# def icnr_init(weight, scale=2):
#     out_ch, in_ch, kH, kW = weight.shape
#     new_out_ch = out_ch // (scale ** 2)

#     subkernel = torch.zeros(new_out_ch, in_ch, kH, kW)
#     init.kaiming_normal_(subkernel)

#     subkernel = subkernel.repeat(scale ** 2, 1, 1, 1)
#     weight.data.copy_(subkernel)



class FinalLearnedUpsample(nn.Module):
    def __init__(self, ch, upsample_factor=4):
        super().__init__()
        num_stages = int(math.log2(upsample_factor))

        layers = []
        for _ in range(num_stages):
            layers.extend([
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False),
            ])

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)



# class DoubleConv(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super().__init__()
#         self.block = nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#         )

#     def forward(self, x):
#         return self.block(x)


# class UNet(nn.Module):
#     def __init__(self, in_channels=None, out_channels=3, features=(64, 128, 256, 512), 
#                  meta_dim=5, meta_hidden=64, meta_out=128, upsample_factor=4, w_range=None, theta_range=None, norm="batch", num_groups=8):
#         super().__init__()
#         self.upsample_factor = upsample_factor
#         self.w_range = w_range
#         self.theta_range = theta_range
#         self.encs = nn.ModuleList()
#         self.pools = nn.ModuleList()
#         ch = in_channels
#         for f in features:
#             self.encs.append(DoubleConv(ch, f, norm=norm, num_groups=num_groups))
#             self.pools.append(nn.MaxPool2d(2))
#             ch = f
#         self.bottleneck_conv = DoubleConv(features[-1], features[-1] * 2, norm=norm, num_groups=num_groups)
#         self.meta_encoder = MetaEncoder(meta_dim=meta_dim, num_images=in_channels, hidden_dim=meta_hidden, out_dim=meta_out)
#         self.film = FiLMLayer(feature_channels=features[-1] * 2, cond_dim=meta_out)
#         self.upconvs = nn.ModuleList()
#         self.decs = nn.ModuleList()
#         ch = features[-1] * 2
#         for f in reversed(features):
#             self.upconvs.append(nn.ConvTranspose2d(ch, f, kernel_size=2, stride=2, bias=False))
#             self.decs.append(DoubleConv(ch, f, norm=norm, num_groups=num_groups))
#             ch = f
#         self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
#         self.final_upsample = FinalLearnedUpsample(out_channels, upsample_factor=upsample_factor) # allow a learned final upsampling instead of fixed bilinear - this is important for the 4x upsampling to get to the original resolution!
#         # self.final_upsample = nn.Upsample(scale_factor=upsample_factor, mode='bilinear', align_corners=False)

    # def forward(self, x, meta):
    #     skips = []
    #     for enc, pool in zip(self.encs, self.pools):
    #         x = enc(x)
    #         skips.append(x)
    #         x = pool(x)
    #     x = self.bottleneck_conv(x)
    #     meta_encoded = self.meta_encoder(meta)
    #     x = self.film(x, meta_encoded)
    #     for up, dec, skip in zip(self.upconvs, self.decs, reversed(skips)):
    #         x = up(x)
    #         if x.size(-1) != skip.size(-1) or x.size(-2) != skip.size(-2):
    #             x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
    #         x = torch.cat([skip, x], dim=1)
    #         x = dec(x)
    #     x = self.final_conv(x)
    #     x = self.final_upsample(x)

    #     # Apply activation functions to constrain outputs to physical ranges
    #     if self.final_conv.out_channels == 3 and self.w_range is not None and self.theta_range is not None:
    #         dem = x[:, 0:1, :, :]  # No activation for DEM
            
    #         # Scale w to [w_min, w_max]
    #         w_raw = x[:, 1:2, :, :]
    #         w = torch.sigmoid(w_raw) * (self.w_range[1] - self.w_range[0]) + self.w_range[0]
            
    #         # Scale theta to [theta_min, theta_max]
    #         theta_raw = x[:, 2:3, :, :]
    #         theta = torch.sigmoid(theta_raw) * (self.theta_range[1] - self.theta_range[0]) + self.theta_range[0]
            
    #         x = torch.cat([dem, w, theta], dim=1)

    #     return x






class DoubleConvFiLM(nn.Module):
    def __init__(self, in_ch, out_ch, norm="group", num_groups=8, cond_dim=None):
        super().__init__()

        if norm == "batch":
            norm_layer1 = nn.BatchNorm2d(out_ch)
            norm_layer2 = nn.BatchNorm2d(out_ch)
        elif norm == "group":
            assert out_ch % num_groups == 0
            norm_layer1 = nn.GroupNorm(num_groups, out_ch)
            norm_layer2 = nn.GroupNorm(num_groups, out_ch)
        elif norm == "instance":
            norm_layer1 = nn.InstanceNorm2d(out_ch, affine=True)
            norm_layer2 = nn.InstanceNorm2d(out_ch, affine=True)
        elif norm is None:
            norm_layer1 = nn.Identity()
            norm_layer2 = nn.Identity()
        else:
            raise ValueError(norm)

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.norm1 = norm_layer1
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.norm2 = norm_layer2

        self.film = FiLMLayer(out_ch, cond_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, cond): # FiLM conditioning on the meta-encoded vector, named "cond" here
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        x = self.film(x, cond)
        x = self.relu(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels=5, out_channels=3, features=(64, 128, 256, 512), 
                 meta_dim=5, meta_hidden=64, meta_out=128, upsample_factor=4, w_range=None, theta_range=None, norm="batch", num_groups=8):
        super().__init__()
        self.upsample_factor = upsample_factor
        self.w_range = w_range
        self.theta_range = theta_range
        self.encs = nn.ModuleList()
        self.pools = nn.ModuleList()
        ch = in_channels
        # first enc will use instance norm
        self.encs.append(
            DoubleConvFiLM(
                in_ch=ch,
                out_ch=features[0],
                norm="instance",
                num_groups=num_groups,
                cond_dim=meta_out,
            ))
        self.pools.append(nn.MaxPool2d(2)) 
        # then the rest will use the specified norm
        ch = features[0]
        for f in features[1:]:  # skip the first one since it's already added with instance norm
            self.encs.append(
                DoubleConvFiLM(
                    in_ch=ch,
                    out_ch=f,
                    norm=norm,
                    num_groups=num_groups,
                    cond_dim=meta_out,
                )
            )
            self.pools.append(nn.MaxPool2d(2))
            ch = f
        self.bottleneck_conv = DoubleConvFiLM(
            features[-1],
            features[-1] * 2,
            norm=norm,
            num_groups=num_groups,
            cond_dim=meta_out,
        )
        self.upconvs = nn.ModuleList()
        self.decs = nn.ModuleList()
        self.meta_encoder = MetaEncoder(meta_dim=meta_dim, num_images=in_channels, hidden_dim=meta_hidden, out_dim=meta_out)
        ch = features[-1] * 2
        for f in reversed(features):
            self.upconvs.append(nn.ConvTranspose2d(ch, f, kernel_size=2, stride=2, bias=False))
            self.decs.append(
                DoubleConvFiLM(
                    in_ch=ch,
                    out_ch=f,
                    norm=norm,
                    num_groups=num_groups,
                    cond_dim=meta_out,
                )
            )
            ch = f
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        self.final_upsample = FinalLearnedUpsample(out_channels, upsample_factor=upsample_factor) # allow a learned final upsampling instead of fixed bilinear - this is important for the 4x upsampling to get to the original resolution!
        # self.final_upsample = nn.Upsample(scale_factor=upsample_factor, mode='bilinear', align_corners=False)

    def forward(self, x, meta):
        skips = []

        # Encode metadata ONCE
        meta_encoded = self.meta_encoder(meta)

        # Encoder (FiLM everywhere)
        for enc, pool in zip(self.encs, self.pools):
            x = enc(x, meta_encoded)
            skips.append(x)
            x = pool(x)

        # Bottleneck (FiLM)
        x = self.bottleneck_conv(x, meta_encoded)

        # Decoder (FiLM)
        for up, dec, skip in zip(self.upconvs, self.decs, reversed(skips)):
            x = up(x)

            if x.size(-1) != skip.size(-1) or x.size(-2) != skip.size(-2):
                x = F.interpolate(
                    x,
                    size=skip.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            x = torch.cat([skip, x], dim=1)
            x = dec(x, meta_encoded)

        # Output
        x = self.final_conv(x)
        x = self.final_upsample(x)

        # Physical constraints
        if (
            self.final_conv.out_channels == 3
            and self.w_range is not None
            and self.theta_range is not None
        ):
            dem = x[:, 0:1]

            w = (
                torch.sigmoid(x[:, 1:2])
                * (self.w_range[1] - self.w_range[0])
                + self.w_range[0]
            )

            theta = (
                torch.sigmoid(x[:, 2:3])
                * (self.theta_range[1] - self.theta_range[0])
                + self.theta_range[0]
            )

            x = torch.cat([dem, w, theta], dim=1)

        return x


__all__ = ['UNet', 'MetaEncoder', 'FiLMLayer', 'DoubleConv', 'FinalLearnedUpsample', 'DoubleConvFiLM']
