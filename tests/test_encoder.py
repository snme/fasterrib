import torch
from src.unet.encoder import Encoder  # The code to test


def test_encoder():
    encoder = Encoder()
    # input image
    x = torch.randn(1, 1, 512, 512)
    ftrs = encoder(x)
    shapes = [list(ftr.shape) for ftr in ftrs]
    print(shapes)

    assert shapes == [
        [1, 64, 512, 512],
        [1, 128, 256, 256],
        [1, 256, 128, 128],
        [1, 512, 64, 64],
        [1, 1024, 32, 32],
    ]
