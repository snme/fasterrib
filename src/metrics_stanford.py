import argparse
import os
import sys
from datetime import datetime

# or, manual
from src.engine import train_one_epoch, evaluate
import numpy as np

import torch
# from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import ConcatDataset, DataLoader, Subset
import torchvision
import wandb

from src.hparams import HParams
from src.config import NUM_CORES
from src.object_detection_dataset import ODDataset
from src.inference_dataset import InferenceDataset
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
            # nn.Linear(200, 200),
            nn.Linear(200, num_classes),
            nn.LeakyReLU()
        )
        self.bbox_pred= nn.Sequential(
            nn.Linear(in_channels, 200),
            # nn.Linear(200, 200),
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

def main():
    m1 = torch.load('final_deep_regressor.pkl')

    torch.manual_seed(0)
    data = InferenceDataset(
        data_dir='./data/stanford/niis'
    )

    data_loader = DataLoader(
        data,
        batch_size=1, #hparams.batch_size,
        num_workers=NUM_CORES,
        persistent_workers=True,
        collate_fn=collate_fn,
    )

    for i in [0.5]: #np.linspace(0.01, 0.95, 10):
        print('i =', i)
        print('model 1')
        evaluate(m1, val_loader, device=device, to_print=True, skip_load=True, model_no=1, nms_iou_threshold=i)

if __name__ == '__main__':
    main()
