import torch
from src.unet.encoder import Encoder  # The code to test


def test_encoder():
    encoder = Encoder()
    # input image
    x = torch.randn(1, 3, 572, 572)
    ftrs = encoder(x)
    shapes = [list(ftr.shape) for ftr in ftrs]
    print(shapes)

    assert shapes == [
        [1, 64, 568, 568],
        [1, 128, 280, 280],
        [1, 256, 136, 136],
        [1, 512, 64, 64],
        [1, 1024, 28, 28],
    ]
