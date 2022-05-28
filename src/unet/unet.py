import torch.functional as F
from src.unet.decoder import Decoder
from src.unet.encoder import Encoder
from torch import nn


class UNet(nn.Module):
    def __init__(
        self,
        enc_chs=(1, 64, 128, 256, 512, 1024),
        dec_chs=(1024, 512, 256, 128, 64),
        num_classes=6,
    ):
        super().__init__()
        self.encoder = Encoder(enc_chs)
        self.decoder = Decoder(dec_chs)
        self.head = nn.Conv2d(dec_chs[-1], num_classes, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        encoded = self.encoder(x)
        out = self.decoder(encoded[::-1][0], encoded[::-1][1:])
        out = self.head(out)
        return out
