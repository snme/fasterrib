import argparse
import os

import pytorch_lightning as pl
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset

from src.rfc_dataset import RFCDataset
from src.unet.lit_unet import LitUNet
from src.unet.unet import UNet

dirname = os.path.dirname(__file__)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

parser = argparse.ArgumentParser(
    description="Evaluates the ribfrac model on the full RFC validation set"
)

batch_size = 1
torch.cuda.empty_cache()


def eval(data_loader):
    model = LitUNet.load_from_checkpoint(
        "checkpoints-0528-0951/epoch=0-step=1200.ckpt",
        unet=UNet(),
        class_counts=None,
        neg_dir=None,
    )
    model.to(device)
    model.eval()

    # test model
    trainer = pl.Trainer(
        accelerator="auto", devices=1, max_epochs=1, detect_anomaly=True
    )
    trainer.test(model=model, dataloaders=data_loader)


def main():
    val_pos = RFCDataset(
        data_dir="./data/ribfrac-challenge/validation/prepared/pos",
    )
    val_neg = RFCDataset(
        data_dir="./data/ribfrac-challenge/validation/prepared/neg",
    )
    val_data = ConcatDataset([val_pos, val_neg])
    val_data = Subset(val_data, torch.arange(100))
    print("Num validation examples:", len(val_data))
    val_loader = DataLoader(
        val_data, batch_size=8, persistent_workers=True, num_workers=12
    )

    eval(val_loader)


if __name__ == "__main__":
    args = parser.parse_args()
    with torch.no_grad():
        main()
