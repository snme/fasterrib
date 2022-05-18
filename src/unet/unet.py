import torch.functional as F
from src.unet.decoder import Decoder
from src.unet.encoder import Encoder
from torch import nn


class UNet(nn.Module):
    def __init__(
        self,
        enc_chs=(1, 64, 128, 256, 512, 1024),
        dec_chs=(1024, 512, 256, 128, 64),
        num_classes=2,
        retain_dim=False,
        out_size=(512, 512),
    ):
        super().__init__()
        self.encoder = Encoder(enc_chs)
        self.decoder = Decoder(dec_chs)
        self.head = nn.Conv2d(dec_chs[-1], num_classes, 1)
        self.retain_dim = retain_dim
        self.out_size = out_size

    def forward(self, x):
        enc_ftrs = self.encoder(x)
        out = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out = self.head(out)
        if self.retain_dim:
            out = F.interpolate(out, self.out_size)
        return out
