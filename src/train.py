import argparse
import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Subset

from src.rfc_dataset import RFCDataset
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
batch_size = 16


def train(data_loader, val_loader=None):
    class_counts = torch.load(class_counts_path)
    class_counts.requires_grad_(False)
    model = LitUNet(unet=UNet(), class_counts=class_counts)
    model = model.to(device)

    wandb_logger = WandbLogger(project="ribfrac")

    # train model
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(dirname, "../checkpoints"),
        save_top_k=2,
        monitor="val_loss",
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=-1,
        val_check_interval=200,
        callbacks=[checkpoint_callback],
        max_epochs=2,
        logger=wandb_logger,
    )
    trainer.fit(model=model, train_dataloaders=data_loader, val_dataloaders=val_loader)


def main():
    data = RFCDataset(
        data_dir="./data/ribfrac-challenge/training/prepared",
    )
    val_data = RFCDataset(
        data_dir="./data/ribfrac-challenge/validation/prepared",
    )
    val_indices = torch.randperm(len(val_data))[:4000]
    val_subset = Subset(val_data, val_indices)
    train_loader = DataLoader(data, batch_size=batch_size, num_workers=24, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, num_workers=24)
    print("Num training examples:", len(data))
    print("Num validation examples:", len(val_data))
    train(train_loader, val_loader)


if __name__ == "__main__":
    args = parser.parse_args()
    if args.wandb_api_key:
        os.environ["WANDB_API_KEY"] = args.wandb_api_key
    main()
