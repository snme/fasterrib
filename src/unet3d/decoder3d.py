import torch
import torchvision
from src.unet3d.block3d import Block3d
from torch import nn


class Decoder3d(nn.Module):
    def __init__(self, chs=(512, 256, 128, 64, 32)):
        super().__init__()
        self.chs = chs
        self.upconvs = [Up3d(chs[i], chs[i + 1]) for i in range(len(chs) - 1)]

        self.dec_blocks = nn.ModuleList(
            [Block3d(chs[i], chs[i + 1]) for i in range(len(chs) - 1)]
        )

    def forward(self, x, encoder_features):
        for i in range(len(self.chs) - 1):
            x = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x = torch.cat([x, enc_ftrs], dim=1)
            x = self.dec_blocks[i](x)
        return x

    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs


class Up3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, 2, stride=2, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x, y):
        x = self.up(x)
        return x
