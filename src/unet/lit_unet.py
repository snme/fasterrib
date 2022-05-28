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
    def __init__(
        self,
        enc_chs=(1, 64, 128, 256, 512, 1024),
        dec_chs=(1024, 512, 256, 128, 64),
        num_classes=6,
        class_counts: t.Optional[torch.Tensor] = None,
        neg_dir=None,
        learning_rate=1e-6,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.unet = UNet(enc_chs=self.hparams.enc_chs, dec_chs=self.hparams.dec_chs)
        self.class_counts = self.hparams.class_counts
        self.f1 = F1Score(num_classes=self.hparams.num_classes, average="none")
        self.confusion_matrix = ConfusionMatrix(num_classes=self.hparams.num_classes)
        self.neg_dir = self.hparams.neg_dir
        self.learning_rate = self.hparams.learning_rate

        if self.neg_dir:
            print("Using negative sampling")
            self.neg_dataset = RFCDataset(data_dir=self.neg_dir)
            self.neg_loader = DataLoader(
                self.neg_dataset,
                shuffle=True,
                batch_size=4,
                persistent_workers=True,
                num_workers=8,
            )
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
        if self.neg_dir is not None:
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
        bin_dice = torch.stack([x[3] for x in outputs], dim=0).nanmean()

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_dice", dice, prog_bar=True)
        self.log("val_bin_dice", bin_dice, prog_bar=True)
        self.log("val_ce", ce, prog_bar=True)

    def test_step(self, batch, _):
        mixed_loss = MixedLoss()
        img = batch["image"]
        target = batch["label"]
        out = self.unet(img)
        dice_scores = mixed_loss.get_all_dice_scores(out, target)

        target_one_hot = torch.nn.functional.one_hot(target, num_classes=6)
        target_one_hot = target_one_hot.type(torch.int8).permute(0, 3, 1, 2)

        softmax_out = mixed_loss.get_softmax_scores(out)
        bin_dice = mixed_loss.get_binary_dice_score(
            softmax_out, target, target_one_hot=target_one_hot
        )

        y_pred = torch.argmax(out, dim=1)
        assert y_pred.size() == target.size()
        self.confusion_matrix(y_pred.flatten(), target.flatten())
        self.f1(y_pred.flatten(), target.flatten())
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
            "dice",
            {str(i): c for (i, c) in enumerate(dice_scores.squeeze())},
            prog_bar=False,
        )

        self.save_f1_plot(f1.unsqueeze(0).cpu().numpy()[:, 2:])
        self.save_dice_barplot(dice_scores.cpu().numpy()[:, 2:])
        self.save_confusion_matrix(confmat.cpu()[1:, 1:])

    def forward(self, img):
        """img should have shape (N, H, W)"""
        logits = self.unet(img)  # (N, C, H, W)
        return logits

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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
