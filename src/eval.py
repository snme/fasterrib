import argparse
import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import ConcatDataset, DataLoader, Subset

from src.rfc_dataset import RFCDataset
from src.unet.lit_unet import LitUNet
from src.unet.unet import UNet

dirname = os.path.dirname(__file__)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

parser = argparse.ArgumentParser(description="Evaluates the ribfrac model")
parser.add_argument(
    "--wandb-api-key", help="W&B API KEY for experiment visualization", required=False
)

batch_size = 8


def eval(data_loader):
    model = LitUNet.load_from_checkpoint(
        "./checkpoints-0525-1018/epoch=6-step=35598.ckpt",
        unet=UNet(),
        class_counts=None,
    )
    model.eval()
    model = model.to(device)

    wandb_logger = WandbLogger(project="ribfrac-eval")

    # test model
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=-1,
        max_epochs=1,
        logger=wandb_logger,
        detect_anomaly=True,
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
    val_data = Subset(val_data, torch.randperm(len(val_data)))
    val_loader = DataLoader(
        val_data, batch_size=batch_size, num_workers=24, shuffle=True
    )
    print("Num validation examples:", len(val_data))
    eval(val_loader)


if __name__ == "__main__":
    args = parser.parse_args()
    if args.wandb_api_key:
        os.environ["WANDB_API_KEY"] = args.wandb_api_key
    main()
