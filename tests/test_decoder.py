import torch
from src.unet.decoder import Decoder
from src.unet.encoder import Encoder


def test_decoder():
    decoder = Decoder()
    encoder = Encoder()

    # input image
    x = torch.randn(1, 1, 512, 512)
    enc_ftrs = encoder(x)
    out = decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
    assert list(out.shape) == [1, 64, 512, 512]
