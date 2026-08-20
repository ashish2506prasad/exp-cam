"""Mask generators ported from AM0/AM6 (CSAE and UNetCSAE)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CSAE(nn.Module):
    """Stride-2 conv autoencoder that emits mask logits at 224×224."""

    def __init__(self):
        super().__init__()
        self.enc1 = nn.Conv2d(3, 32, 3, stride=2, padding=1)
        self.enc2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.enc3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.enc4 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.dec1 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.dec2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.dec3 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.dec4 = nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1)

    def forward(self, x):
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
        z = F.relu(self.enc4(x))
        x = F.relu(self.dec1(z))
        x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        return self.dec4(x)


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNetCSAE(nn.Module):
    """Lightweight U-Net mask generator used in the paper (AM0 cell 5)."""

    def __init__(self):
        super().__init__()
        self.down1 = DoubleConv(3, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = DoubleConv(128, 256)
        self.pool4 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(256, 512)
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = DoubleConv(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = DoubleConv(64, 32)
        self.out_conv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))
        d4 = self.down4(self.pool3(d3))
        bn = self.bottleneck(self.pool4(d4))
        u4 = self.dec4(torch.cat([self.up4(bn), d4], dim=1))
        u3 = self.dec3(torch.cat([self.up3(u4), d3], dim=1))
        u2 = self.dec2(torch.cat([self.up2(u3), d2], dim=1))
        u1 = self.dec1(torch.cat([self.up1(u2), d1], dim=1))
        return self.out_conv(u1)


def last_shared_layer(mask_net: nn.Module) -> nn.Module:
    """Last layer used by all decoder branches (for GradNorm-style balancing)."""
    if isinstance(mask_net, UNetCSAE):
        return mask_net.dec1
    return mask_net.dec3
