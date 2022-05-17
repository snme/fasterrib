from torch import nn


class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding="same")
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding="same")

    def forward(self, x):
        return self.relu(self.conv2(self.relu(self.conv1(x))))
