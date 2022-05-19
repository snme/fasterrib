import os

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset, random_split
from tqdm import tqdm

from src.rfc_dataset import RFCDataset
from src.unet.lit_unet import LitUNet
from src.unet.loss import MixedLoss
from src.unet.unet import UNet

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


def train(data_loader, val_loader=None):
    model = LitUNet(UNet())
    model = model.to(device)

    # train model
    trainer = pl.Trainer(accelerator="gpu", devices=-1, val_check_interval=100)
    trainer.fit(model=model, train_dataloaders=data_loader, val_dataloaders=val_loader)


def main():
    data = RFCDataset(
        data_dir="./data/ribfrac-challenge/training/prepared",
    )
    val_data = RFCDataset(
        data_dir="./data/ribfrac-challenge/validation/prepared",
    )
    val_data = Subset(val_data, np.arange(200))
    train_loader = DataLoader(data, batch_size=8, num_workers=8)
    val_loader = DataLoader(val_data, batch_size=8, num_workers=8)
    print(len(data))
    train(train_loader, val_loader)


if __name__ == "__main__":
    main()
