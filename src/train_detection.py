import argparse
import os
import sys
from datetime import datetime

# or, manual
from src.engine import train_one_epoch, evaluate

import torch
# from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import ConcatDataset, DataLoader, Subset
import torchvision

from src.hparams import HParams
from src.config import NUM_CORES
from src.object_detection_dataset import ODDataset
from src.utils import collate_fn
from tqdm import tqdm

dirname = os.path.dirname(__file__)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
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

def train(hparams, data_loader, val_loader=None):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
            progress=True,
            num_classes=5
            )
    model.train()
    model = model.to(device)

    # wandb_logger = WandbLogger(project="ribfrac")

    checkpoint_dir = os.path.join(
        dirname,
        f"../checkpoints-{datetime.now().strftime('%m%d-%H%M')}",
    )

    # train model

    # trainer.fit(model=model, train_dataloaders=data_loader, val_dataloaders=val_loader)

    # construct optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    num_epochs = 1
    for epoch in tqdm(range(num_epochs)):
        # train for one epoch, printing every 10 iterations
        _, losses = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # save model
        torch.save(model, f'{checkpoint_dir}_epoch={epoch}.pkl')
        # update the learning rate
        # lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, val_loader, device=device)

    print("That's it!")

def main():
    hparams = HParams()

    data = ODDataset(
        data_dir="./data/ribfrac-challenge/training/prepared/od-c-pos",
    )
    data = Subset(data, torch.randperm(len(data)))

    val_pos = ODDataset(
        data_dir="./data/ribfrac-challenge/validation/prepared/od-c-pos",
    )
    val_neg = ODDataset(
        data_dir="./data/ribfrac-challenge/validation/prepared/od-c-neg",
    )
    val_pos_subset = Subset(val_pos, torch.randperm(len(val_pos))[:500]) #2000
    val_neg_subset = Subset(val_neg, torch.randperm(len(val_neg))[:500]) #2000
    val_data = ConcatDataset([val_pos_subset, val_neg_subset])

    train_loader = DataLoader(
        data,
        batch_size=4, #hparams.batch_size - hparams.neg_samples,
        num_workers=NUM_CORES,
        shuffle=True,
        persistent_workers=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_data,
        batch_size=4, #hparams.batch_size,
        num_workers=NUM_CORES,
        persistent_workers=True,
        collate_fn=collate_fn,
    )

    print("Num training examples:", len(data))
    print("Num validation examples:", len(val_data))

    train(hparams, train_loader, val_loader)

if __name__ == "__main__":
    args = parser.parse_args()
    if args.wandb_api_key:
        os.environ["WANDB_API_KEY"] = args.wandb_api_key
    main()
