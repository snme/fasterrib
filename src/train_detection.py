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
import wandb

from src.hparams import HParams
from src.config import NUM_CORES
from src.object_detection_dataset import ODDataset
from src.utils import collate_fn
from tqdm import tqdm
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import torch.nn.functional as F
from torch import nn


num_classes = 5
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


class predictor(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.cls_scores = nn.Sequential(
            nn.Linear(in_channels, 200),
            nn.Linear(200, 200),
            nn.Linear(200, num_classes),
            nn.LeakyReLU()
        )
        self.bbox_pred= nn.Sequential(
            nn.Linear(in_channels, 200),
            nn.Linear(200, 200),
            nn.Linear(200, 4 * num_classes),
            nn.LeakyReLU()
        )
        # self.cls_score = nn.Linear(in_channels, num_classes)
        # self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        scores = self.cls_scores(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas

def train(hparams, data_loader, val_loader=None):
    # 1e-4 is the best, thn 1e-3
    for lr in [1e-5]:# 1e-3, 1e-4, 1e-2, 1e-5, 1e-6 # [1e-5, 1e-4, 1e-6, 1e-3, 1e-2, 1e-1]:
        wandb.init()
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
                weights='DEFAULT',
                progress=True,
                )
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = predictor(in_features, num_classes)# FastRCNNPredictor(in_features, num_classes)

        model.train()
        model = model.to(device)

        # wandb_logger = WandbLogger(project="ribfrac")

        checkpoint_dir = os.path.join(
            dirname,
            f"../checkpoints/checkpoints-{datetime.now().strftime('%m%d-%H%M')}",
        )

        # train model

        # trainer.fit(model=model, train_dataloaders=data_loader, val_dataloaders=val_loader)

        # construct optimizer
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=lr,
                                    # momentum=0.9,
                                    weight_decay=0.0005)

        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                       step_size=2,
                                                       gamma=0.1)

        num_epochs = 10
        model = torch.load('final_deep_regressor.pkl') #checkpoints-1116-0102_epoch=1_lr=0.0001.pkl')
        model = model.to(device)
        print('Saving to', checkpoint_dir, 'absolute yolo')
        for epoch in tqdm(range(1, num_epochs)):
            # train for one epoch, printing every 10 iterations
            _, losses = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
            # save model
            should_save = True # epoch % 5 == 0 or epoch == num_epochs - 1
            if should_save:
                torch.save(model, f'{checkpoint_dir}_epoch={epoch}_lr={lr}.pkl')
            # torch.save(torch.Tensor(losses), f'{checkpoint_dir}_epoch={epoch}_lr={lr}_overfit_losses.pkl')
            # update the learning rate
            lr_scheduler.step()
            # evaluate on the test dataset
            evaluate(model, val_loader, device=device, to_print=should_save)

        wandb.finish()


    print("That's it!")

def main():
    hparams = HParams()

    pos_data = ODDataset(
        data_dir="./data/ribfrac-challenge/training/prepared/od-c-pos",
    )
    neg_data = ODDataset(
        data_dir="./data/ribfrac-challenge/training/prepared/od-c-neg",
    )
    total_len = len(pos_data)
    pos_data = Subset(pos_data, torch.randperm(len(pos_data))[:int(0.9 * total_len)])
    neg_data = Subset(neg_data, torch.randperm(len(neg_data))[:int(0.1 * total_len)])
    data = ConcatDataset([pos_data, neg_data])

    val_pos = ODDataset(
        data_dir="./data/ribfrac-challenge/validation/prepared/od-c-pos",
    )
    val_neg = ODDataset(
        data_dir="./data/ribfrac-challenge/validation/prepared/od-c-neg",
    )
    val_pos_subset = Subset(val_pos, torch.randperm(len(val_pos))[:400]) #2000
    val_neg_subset = Subset(val_neg, torch.randperm(len(val_neg))[:400]) #2000
    val_data = ConcatDataset([val_pos_subset, val_neg_subset])

    train_loader = DataLoader(
        data,
        batch_size=7, #hparams.batch_size - hparams.neg_samples,
        num_workers=NUM_CORES,
        shuffle=True,
        persistent_workers=True,
        collate_fn=collate_fn,
    )

    '''val_loader = DataLoader(
        pos_data,
        batch_size=7, #hparams.batch_size - hparams.neg_samples,
        num_workers=NUM_CORES,
        shuffle=True,
        persistent_workers=True,
        collate_fn=collate_fn,
    )'''

    val_loader = DataLoader(
        val_data,
        batch_size=7, #hparams.batch_size,
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
