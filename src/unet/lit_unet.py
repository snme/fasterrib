import os
import typing as t

import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
from src.rfc_dataset import RFCDataset
from src.unet.loss import MixedLoss
from src.unet.unet import UNet
from torch.utils.data import DataLoader
from torchmetrics import ConfusionMatrix, F1Score


class LitUNet(pl.LightningModule):
    def __init__(self, unet: UNet, class_counts: t.Optional[torch.Tensor]):
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
        self.confusion_matrix = ConfusionMatrix(num_classes=6)

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

    def test_step(self, batch, _):
        mixed_loss = MixedLoss()
        img = batch["image"]
        label = batch["label"]
        out = self.unet(img)
        dice_scores = mixed_loss.get_all_dice_scores(out, label)
        bin_dice = mixed_loss.get_binary_dice_score(out, label)
        y_pred = torch.argmax(out, dim=1)
        assert y_pred.size() == label.size()
        self.confusion_matrix(y_pred.flatten(), label.flatten())
        self.f1(y_pred.flatten(), label.flatten())
        return dice_scores, bin_dice

    def test_epoch_end(self, outputs):
        dice_scores = torch.stack([x[0] for x in outputs], dim=0).nanmean(
            dim=0, keepdim=True
        )
        bin_dice = torch.cat([x[1] for x in outputs]).nanmean()
        f1 = self.f1.compute()
        confmat = self.confusion_matrix.compute()
        self.log("bin_dice", bin_dice, prog_bar=True)
        self.log("f1", {str(i): c for (i, c) in enumerate(f1)}, prog_bar=False)
        self.log(
            "dice", {str(i): c for (i, c) in enumerate(dice_scores)}, prog_bar=False
        )
        self.save_dice_barplot(dice_scores.cpu().numpy())
        self.save_confusion_matrix(confmat.cpu())

    def save_dice_barplot(self, dice_scores):
        data = pd.DataFrame(
            dice_scores,
            columns=["-1", "0", "1", "2", "3", "4"],
            index=["DICE"],
        )
        ax = sns.barplot(data=data, palette="Blues_d")
        ax.set_title("Avg DICE per class")
        ax.set_xlabel("class")
        ax.set_ylabel("DICE")
        plt.savefig("dice_barplot.png")

    def save_confusion_matrix(self, confmat):
        mask = torch.zeros_like(confmat)
        mask[0, 0] = 1
        mask[1, 1] = 1
        ax = sns.heatmap(confmat, annot=True, cmap="Blues", fmt="g", mask=mask.numpy())
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted Values")
        ax.set_ylabel("Actual Values")

        ## Ticket labels - List must be in alphabetical order
        ax.xaxis.set_ticklabels(["-1", "0", "1", "2", "3", "4"])
        ax.yaxis.set_ticklabels(["-1", "0", "1", "2", "3", "4"])

        ## Display the visualization of the Confusion Matrix.
        plt.savefig("confmat.png")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
