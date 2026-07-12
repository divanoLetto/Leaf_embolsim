"""
model.py — Method 4: U-Net with Dropout2d regularisation.

Changes vs Method 2:
  - Dropout2d added after each DecoderBlock conv to prevent spatial
    memorisation of training-set embolism patterns.
  - Dropout is active only during training (nn.Dropout2d respects model.eval()).

Architecture summary
--------------------
Input:  6-channel tensor (frame_t and frame_{t+1} concatenated along C).
Output: 1-channel probability map (sigmoid), same spatial size as input.

Channel progression:
  Encoder:     6 → 32 → 64 → 128
  Bottleneck:  128 → 256
  Decoder:     256+128 → 128 → 64+64 → 64 → 32+32 → 32
  Output:      32 → 1 (sigmoid)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = ConvBNReLU(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x: torch.Tensor):
        skip = self.conv(x)
        down = self.pool(skip)
        return skip, down


class DecoderBlock(nn.Module):
    """Upsample → concat skip → ConvBNReLU → Dropout2d."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, dropout_p: float = 0.3):
        super().__init__()
        self.conv    = ConvBNReLU(in_ch + skip_ch, out_ch)
        # Dropout2d zeros entire feature-map channels, which is more effective
        # than per-pixel dropout for preventing spatial memorisation.
        self.dropout = nn.Dropout2d(p=dropout_p)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.dropout(self.conv(x))


class UNet(nn.Module):
    def __init__(self, in_channels: int = 6, base_ch: int = 32, dropout_p: float = 0.3):
        super().__init__()

        self.enc1 = EncoderBlock(in_channels, base_ch)
        self.enc2 = EncoderBlock(base_ch,      base_ch * 2)
        self.enc3 = EncoderBlock(base_ch * 2,  base_ch * 4)

        self.bottleneck = ConvBNReLU(base_ch * 4, base_ch * 8)

        self.dec3 = DecoderBlock(base_ch * 8, base_ch * 4, base_ch * 4, dropout_p)
        self.dec2 = DecoderBlock(base_ch * 4, base_ch * 2, base_ch * 2, dropout_p)
        self.dec1 = DecoderBlock(base_ch * 2, base_ch,     base_ch,     dropout_p)

        self.out_conv = nn.Conv2d(base_ch, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1, x = self.enc1(x)
        s2, x = self.enc2(x)
        s3, x = self.enc3(x)

        x = self.bottleneck(x)

        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)

        return torch.sigmoid(self.out_conv(x))
