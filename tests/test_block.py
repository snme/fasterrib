import torch
from src.unet.block import Block  # The code to test


def test_block():
    enc_block = Block(1, 64)
    x = torch.randn(1, 1, 512, 512)
    assert list(enc_block(x).shape) == [1, 64, 512, 512]
