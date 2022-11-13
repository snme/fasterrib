import argparse
import os
from datetime import datetime

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import ConcatDataset, DataLoader, Subset

from src.config import NUM_CORES
from src.rfc_dataset import RFCDataset
from src.unet.hparams import HParams
from src.unet.lit_unet import LitUNet
from src.unet.unet import UNet

dirname = os.path.dirname(__file__)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

parser = argparse.ArgumentParser(description="Trains the ribfrac model")
parser.add_argument(
    "--wandb-api-key", help="W&B API KEY for experiment visualization", required=False
)

train_dir = os.path.join(dirname, "../data/ribfrac-challenge/training/")
class_counts_path = os.path.join(train_dir, "class_counts.pt")
dirname = os.path.dirname(__file__)
default_neg_dir = os.path.join(
    dirname, "../data/ribfrac-challenge/training/prepared/neg"
)

torch.cuda.empty_cache()


def train(hparams, data_loader, val_loader=None):
    model = LitUNet(params=hparams.dict())
    model.train()
    model = model.to(device)

    wandb_logger = WandbLogger(project="ribfrac")

    checkpoint_dir = os.path.join(
        dirname,
        f"../checkpoints-{datetime.now().strftime('%m%d-%H%M')}",
    )

    loss_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        save_top_k=2,
        monitor="val_loss",
        filename="{epoch}-{step}-{val_loss:.2f}",
    )

    bin_dice_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        save_top_k=2,
        monitor="val_bin_dice",
        filename="{epoch}-{step}-{val_bin_dice:.2f}",
        mode="max",
    )

    trainer = pl.Trainer(
        # accelerator="cpu",
        # devices=NUM_CORES,
        val_check_interval=1000,
        callbacks=[loss_callback, bin_dice_callback],
        max_epochs=100,
        strategy="ddp",
        logger=wandb_logger,
        detect_anomaly=True,
    )
    trainer.fit(model=model, train_dataloaders=data_loader, val_dataloaders=val_loader)


def main():
    hparams = HParams()

    data = RFCDataset(
        data_dir="./data/ribfrac-challenge/training/prepared/pos",
    )
    data = Subset(data, torch.randperm(len(data)))

    val_pos = RFCDataset(
        data_dir="./data/ribfrac-challenge/validation/prepared/pos",
    )
    val_neg = RFCDataset(
        data_dir="./data/ribfrac-challenge/validation/prepared/neg",
    )
    val_pos_subset = Subset(val_pos, torch.randperm(len(val_pos))[:2000])
    val_neg_subset = Subset(val_neg, torch.randperm(len(val_neg))[:2000])
    val_data = ConcatDataset([val_pos_subset, val_neg_subset])

    train_loader = DataLoader(
        data,
        batch_size=hparams.batch_size - hparams.neg_samples,
        num_workers=NUM_CORES,
        shuffle=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=hparams.batch_size,
        num_workers=NUM_CORES,
        persistent_workers=True,
    )
    print("Num training examples:", len(data))
    print("Num validation examples:", len(val_data))
    train(hparams, train_loader, val_loader)


if __name__ == "__main__":
    args = parser.parse_args()
    if args.wandb_api_key:
        os.environ["WANDB_API_KEY"] = args.wandb_api_key
    main()
