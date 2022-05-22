from torch import nn


class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding="same"),
            nn.GroupNorm(1, out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding="same"),
            nn.GroupNorm(1, out_ch),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)
