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
        self.confusion_matrix = ConfusionMatrix(num_classes=6)
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

        return loss

    def validation_step(self, batch, batch_idx):
        # this is the test loop
        loss_fn = MixedLoss()
        img = batch["image"]
        label = batch["label"]
        out = self.unet(img)
        loss, dice, ce, bin_dice = loss_fn(out, label, class_counts=self.class_counts)
        return loss, dice, ce, bin_dice

    def validation_epoch_end(self, outputs):
        loss = torch.stack([x[0] for x in outputs]).mean()
        dice = torch.stack([x[1] for x in outputs]).nanmean()
        ce = torch.stack([x[2] for x in outputs]).mean()
        bin_dice = torch.stack([x[3] for x in outputs], dim=0).mean()

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_dice", dice, prog_bar=True)
        self.log("val_bin_dice", bin_dice, prog_bar=True)
        self.log("val_ce", ce, prog_bar=True)

    def test_step(self, batch, _):
        mixed_loss = MixedLoss()
        img = batch["image"]
        label = batch["label"]
        out = self.unet(img)
        dice_scores = mixed_loss.get_all_dice_scores(out, label)

        softmax_out = mixed_loss.get_softmax_scores(out)
        bin_dice = mixed_loss.get_binary_dice_score(softmax_out, label)

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
        f1 = self.f1.compute().unsqueeze(0)
        confmat = self.confusion_matrix.compute()
        self.log("bin_dice", bin_dice, prog_bar=True)
        self.log("f1", {str(i): c for (i, c) in enumerate(f1[0])}, prog_bar=False)
        self.log(
            "dice", {str(i): c for (i, c) in enumerate(dice_scores[0])}, prog_bar=False
        )

        self.save_f1_plot(f1.cpu().numpy()[:, 2:])
        self.save_dice_barplot(dice_scores.cpu().numpy()[:, 2:])
        self.save_confusion_matrix(confmat.cpu()[1:, 1:])

    def forward(self, img):
        """img should have shape (N, H, W)"""
        logits = self.unet(img)  # (N, C, H, W)
        logits[:, 0] = float("-inf")  # make sure prediction for -1 class is zero
        out = torch.softmax(logits, dim=1)
        return out

    def save_f1_plot(self, f1_scores):
        data = pd.DataFrame(
            f1_scores,
            columns=["1", "2", "3", "4"],
            index=["F1"],
        )
        ax = sns.barplot(data=data, palette="Blues_d")
        ax.set_title("F1 Scores")
        ax.set_xlabel("class")
        ax.set_ylabel("F1")
        plt.savefig("f1_barplot.png")

    def save_dice_barplot(self, dice_scores):
        data = pd.DataFrame(
            dice_scores,
            columns=["1", "2", "3", "4"],
            index=["DICE"],
        )
        ax = sns.barplot(data=data, palette="Blues_d")
        ax.set_title("Avg DICE per class")
        ax.set_xlabel("class")
        ax.set_ylabel("DICE")
        plt.savefig("dice_barplot.png")

    def save_confusion_matrix(self, confmat):
        plt.clf()
        mask = torch.zeros_like(confmat)
        mask[0, 0] = True

        with sns.axes_style("white"):
            ax = sns.heatmap(
                confmat, annot=True, cmap="Blues", fmt="g", mask=mask.numpy()
            )
            ax.set_title("Confusion Matrix")
            ax.set_xlabel("Predicted Values")
            ax.set_ylabel("Actual Values")

            ## Ticket labels - List must be in alphabetical order
            ax.xaxis.set_ticklabels(["0", "1", "2", "3", "4"])
            ax.yaxis.set_ticklabels(["0", "1", "2", "3", "4"])

            ## Display the visualization of the Confusion Matrix.
            plt.savefig("confmat.jpg")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
