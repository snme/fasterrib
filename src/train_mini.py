import os
from datetime import datetime

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import ConcatDataset, DataLoader, Subset

from src.rfc_dataset import RFCDataset
from src.unet.hparams import HParams
from src.unet.lit_unet import LitUNet
from src.unet.unet import UNet

dirname = os.path.dirname(__file__)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

main_dir = os.path.join(dirname, "../data/ribfrac-challenge/mini/")
neg_dir = os.path.join(main_dir, "prepared/neg")
class_counts_path = os.path.join(main_dir, "class_counts.pt")
batch_size = 8

torch.cuda.empty_cache()


def train(data_loader):
    model = LitUNet(params=HParams())
    model.train()
    model = model.to(device)
    # train model
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(dirname, f"../checkpoints-mini"),
        save_top_k=1,
        monitor="train_loss",
        filename="best.ckpt",
        every_n_epochs=10,
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=-1,
        callbacks=[checkpoint_callback],
        max_epochs=1000,
        detect_anomaly=True,
    )
    trainer.fit(model=model, train_dataloaders=data_loader)


def main():
    data = RFCDataset(data_dir=os.path.join(main_dir, "prepared/pos"))
    data = Subset(data, torch.randperm(len(data))[:batch_size])
    train_loader = DataLoader(
        data,
        batch_size=batch_size,
        num_workers=24,
        shuffle=False,
        persistent_workers=True,
    )
    print("Num training examples:", len(data))
    train(train_loader)


if __name__ == "__main__":
    main()
