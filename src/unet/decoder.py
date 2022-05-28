import torch
from src.unet.block import Block
from torch import nn


class Decoder(nn.Module):
    def __init__(self, channels=(512, 256, 128, 64, 32)):
        super().__init__()
        self.channels = channels
        self.upconvs = nn.ModuleList(
            [nn.ConvTranspose2d(channels[i], channels[i + 1], 2, 2) for i in range(len(channels) - 1)]
        )
        self.dec_blocks = nn.ModuleList(
            [Block(channels[i], channels[i + 1]) for i in range(len(channels) - 1)]
        )

    def forward(self, x, y):
        for i in range(len(self.channels) - 1):
            x = self.upconvs[i](x)
            x = torch.cat([y[i], x], dim=1)
            x = self.dec_blocks[i](x)
        return x
