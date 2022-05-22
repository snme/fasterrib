import os

import pytorch_lightning as pl
import torch
from src.rfc_dataset import RFCDataset
from src.unet.loss import MixedLoss
from src.unet.unet import UNet
from torch.utils.data import DataLoader
from torchmetrics import F1Score


class LitUNet(pl.LightningModule):
    def __init__(self, unet: UNet, class_counts: torch.Tensor):
        super().__init__()
        self.unet = unet
        self.class_counts = class_counts
        self.f1 = F1Score(num_classes=6, average="none")

        dirname = os.path.dirname(__file__)
        neg_dir = os.path.join(
            dirname, "../../data/ribfrac-challenge/training/prepared/neg"
        )
        self.neg_dataset = RFCDataset(data_dir=neg_dir)
        self.neg_loader = DataLoader(self.neg_dataset, shuffle=True, batch_size=4)
        self.neg_iter = iter(self.neg_loader)

    def get_neg_samples(self):
        try:
            neg_batch = next(self.neg_iter)
            return neg_batch
        except StopIteration:
            self.neg_iter = iter(self.neg_loader)
            return next(self.neg_iter)

    def training_step(self, batch, batch_idx):
        img = batch["image"]
        label = batch["label"]

        # Negative sampling
        neg_batch = self.get_neg_samples()
        img = torch.cat([img, neg_batch["image"].to(img.get_device())], dim=0)
        label = torch.cat([label, neg_batch["label"].to(img.get_device())], dim=0)

        loss_fn = MixedLoss()

        out = self.unet(img)
        loss, dice, ce, bin_dice = loss_fn(out, label, class_counts=self.class_counts)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_ce", ce, prog_bar=True)
        self.log("train_binary_dice", bin_dice, prog_bar=True)
        if not torch.isnan(dice):
            self.log("train_dice", dice, prog_bar=True)

        y_pred = torch.argmax(out, dim=1)
        assert y_pred.size() == label.size()
        f1 = self.f1(y_pred.flatten(), label.flatten())
        self.log("train_f1", {i: c for (i, c) in enumerate(f1)}, prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):
        # this is the test loop
        loss_fn = MixedLoss()
        img = batch["image"]
        label = batch["label"]
        out = self.unet(img)
        loss, dice, ce, bin_dice = loss_fn(out, label, class_counts=self.class_counts)

        y_pred = torch.argmax(out, dim=1)
        assert y_pred.size() == label.size()
        f1 = self.f1(y_pred.flatten(), label.flatten())
        return loss, dice, ce, f1, bin_dice

    def validation_epoch_end(self, outputs):
        loss = torch.stack([x[0] for x in outputs]).mean()
        dice = torch.stack([x[1] for x in outputs]).nanmean()
        ce = torch.stack([x[2] for x in outputs]).mean()
        f1 = torch.stack([x[3] for x in outputs], dim=0).nanmean(dim=0)
        bin_dice = torch.stack([x[4] for x in outputs], dim=0).mean()

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_dice", dice, prog_bar=True)
        self.log("val_bin_dice", bin_dice, prog_bar=True)
        self.log("val_ce", ce, prog_bar=True)
        self.log("val_f1", {i: c for (i, c) in enumerate(f1)}, prog_bar=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
