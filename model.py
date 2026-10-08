"""小型 U-Net：三次下采样、三次上采样，以及对应尺度的跳跃连接。"""
import torch
from torch import nn
from torch.nn import functional as F


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )


class UNet(nn.Module):
    def __init__(self, base_channels=16):
        super().__init__()
        b = base_channels
        self.enc1 = DoubleConv(3, b)
        self.enc2 = DoubleConv(b, b * 2)
        self.enc3 = DoubleConv(b * 2, b * 4)
        self.bottleneck = DoubleConv(b * 4, b * 8)
        self.pool = nn.MaxPool2d(2)
        self.dec3 = DoubleConv(b * 12, b * 4)
        self.dec2 = DoubleConv(b * 6, b * 2)
        self.dec1 = DoubleConv(b * 3, b)
        self.head = nn.Conv2d(b, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        x = self.bottleneck(self.pool(e3))
        for skip, decoder in [(e3, self.dec3), (e2, self.dec2), (e1, self.dec1)]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = decoder(torch.cat([x, skip], dim=1))
        return self.head(x)  # 原始 logits；损失内部处理 sigmoid
