from src.unet.block import Block
from torch import nn


class Encoder(nn.Module):
    def __init__(self, chs=(1, 32, 64, 128, 256, 512)):
        super().__init__()
        self.enc_blocks = nn.ModuleList(
            [Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)]
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        out = []
        for block in self.enc_blocks:
            x = block(x)
            out.append(x)
            x = self.pool(x)
        return out
