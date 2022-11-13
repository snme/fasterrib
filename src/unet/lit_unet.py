import typing as t

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
from matplotlib.colors import LogNorm
from sklearn import metrics
from src.config import NUM_CORES
from src.rfc_dataset import RFCDataset
from src.unet.hparams import HParams
from src.unet.loss import MixedLoss
from src.unet.unet import UNet
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import AUROC, ROC, ConfusionMatrix, F1Score, MatthewsCorrCoef
from torchmetrics import functional as metricsF


class LitUNet(pl.LightningModule):
    def __init__(self, params: t.Dict):
        super().__init__()
        self.save_hyperparameters(params)

        self.params = HParams.parse_obj(self.hparams)

        self.unet = UNet(enc_chs=self.params.enc_chs, dec_chs=self.hparams.dec_chs)

        if self.params.class_counts:
            self.class_counts = nn.Parameter(
                torch.tensor(self.params.class_counts, device=self.device),
                requires_grad=False,
            )
        else:
            self.class_counts = None

        self.neg_dir = self.hparams.neg_dir
        self.learning_rate = self.hparams.learning_rate
        self.num_classes = self.hparams.num_classes

        self.f1 = F1Score(
            num_classes=self.hparams.num_classes, average="none", ignore_index=0
        )
        self.macro_f1 = F1Score(
            num_classes=self.hparams.num_classes, average="macro", ignore_index=0
        )
        self.mcc = MatthewsCorrCoef(num_classes=self.params.num_classes)
        self.confusion_matrix = ConfusionMatrix(num_classes=self.hparams.num_classes)
        self.roc = ROC(num_classes=self.num_classes, compute_on_cpu=False)
        self.auroc = AUROC(
            num_classes=self.num_classes, average=None, compute_on_cpu=False
        )

        if self.neg_dir:
            print("Using negative sampling")
            self.neg_dataset = RFCDataset(data_dir=self.neg_dir)
            self.neg_loader = DataLoader(
                self.neg_dataset,
                shuffle=True,
                batch_size=self.params.neg_samples,
                persistent_workers=True,
                num_workers=NUM_CORES,
            )
            self.neg_iter = iter(self.neg_loader)

        if self.class_counts is not None:
            print("Using class counts")

    def get_neg_samples(self):
        try:
            neg_batch = next(self.neg_iter)
            return neg_batch
        except StopIteration:
            self.neg_iter = iter(self.neg_loader)
            return next(self.neg_iter)

    def training_step(self, batch, batch_idx):
        img = batch["image"]
        target = batch["label"]

        # Negative sampling
        if self.neg_dir is not None:
            neg_batch = self.get_neg_samples()
            img = torch.cat([img, neg_batch["image"]], dim=0)
            target = torch.cat([target, neg_batch["label"]], dim=0)

        mixed_loss = MixedLoss(params=self.params)
        out = self.unet(img)

        loss, dice, ce, bin_dice = mixed_loss(
            out, target, class_counts=self.class_counts
        )

        softmax_out = mixed_loss.get_softmax_scores(out)
        y_pred = torch.argmax(softmax_out, dim=1)
        train_f1 = metricsF.f1_score(
            y_pred.flatten(),
            target.flatten(),
            average="macro",
            num_classes=6,
            ignore_index=0,
        )

        y_mcc = y_pred[target != 0]
        t_mcc = target[target != 0]
        train_mcc = metricsF.matthews_corrcoef(
            y_mcc.flatten(), t_mcc.flatten(), num_classes=6
        )

        self.log("train_macro_f1", train_f1, prog_bar=True)
        self.log("train_mcc", train_mcc, prog_bar=True)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_ce", ce, prog_bar=True)
        self.log("train_binary_dice", bin_dice, prog_bar=True)
        if not torch.isnan(dice):
            self.log("train_dice", dice, prog_bar=True)

        print('training step')
        return loss

    def validation_step(self, batch, batch_idx):
        # this is the test loop
        print('validation step')
        mixed_loss = MixedLoss(params=self.params)
        img = batch["image"]
        target = batch["label"]
        out = self.unet(img)

        loss, dice, ce, bin_dice = mixed_loss(
            out, target, class_counts=self.class_counts
        )

        softmax_out = mixed_loss.get_softmax_scores(out)
        y_pred = torch.argmax(softmax_out, dim=1)
        self.f1(y_pred.flatten(), target.flatten())
        self.macro_f1(y_pred.flatten(), target.flatten())

        return loss, dice, ce, bin_dice

    def validation_epoch_end(self, outputs):
        loss = torch.stack([x[0] for x in outputs]).mean()
        dice = torch.stack([x[1] for x in outputs]).nanmean()
        ce = torch.stack([x[2] for x in outputs]).mean()
        bin_dice = torch.stack([x[3] for x in outputs], dim=0).nanmean()
        f1 = self.f1.compute()
        self.f1.reset()

        macro_f1 = self.macro_f1.compute()
        self.macro_f1.reset()

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_dice", dice, prog_bar=True)
        self.log("val_bin_dice", bin_dice, prog_bar=True)
        self.log("val_ce", ce, prog_bar=True)
        self.log("f1", {str(i): c for (i, c) in enumerate(f1)}, prog_bar=False)
        self.log("macro_f1", macro_f1, prog_bar=False)

    def test_step(self, batch, _):
        mixed_loss = MixedLoss(params=self.params)
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

        y_pred = torch.argmax(softmax_out, dim=1)

        assert y_pred.size() == target.size()
        self.confusion_matrix(y_pred.flatten(), target.flatten())
        self.f1(y_pred.flatten(), target.flatten())
        self.macro_f1(y_pred.flatten(), target.flatten())

        if self.hparams.eval_roc:
            probs = softmax_out.permute(1, 0, 2, 3).flatten(start_dim=1).T
            self.roc(probs, target.flatten())
            self.auroc(probs, target.flatten())

        return dice_scores, bin_dice

    def test_epoch_end(self, outputs):
        dice_scores = torch.stack([x[0] for x in outputs], dim=0).nanmean(
            dim=0, keepdim=True
        )
        bin_dice = torch.cat([x[1] for x in outputs]).nanmean()
        f1 = self.f1.compute()
        macro_f1 = self.macro_f1.compute()
        confmat = self.confusion_matrix.compute()

        self.log("bin_dice", bin_dice, prog_bar=True)
        self.log("macro_f1", macro_f1, prog_bar=True)
        self.log("f1", {str(i): c for (i, c) in enumerate(f1)}, prog_bar=False)
        self.log(
            "dice",
            {str(i): c for (i, c) in enumerate(dice_scores.squeeze())},
            prog_bar=False,
        )

        if self.hparams.eval_roc:
            roc = self.roc.compute()
            auc = self.auroc.compute()
            self.plot_roc(roc, auc)

        self.save_f1_plot(f1.unsqueeze(0).cpu().numpy()[:, 2:])
        self.save_dice_barplot(dice_scores.cpu().numpy()[:, 2:])
        self.save_confusion_matrix(confmat.cpu()[1:, 1:])

    def forward(self, img):
        """img should have shape (N, H, W)"""
        logits = self.unet(img)  # (N, C, H, W)
        return logits

    def save_f1_plot(self, f1_scores):
        plt.clf()
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
        plt.clf()
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

        mask = np.zeros_like(confmat)
        mask[0, 0] = 1
        with sns.axes_style("white"):
            ax = sns.heatmap(
                confmat, annot=True, cmap="Blues", fmt=".3g", norm=LogNorm(), mask=None
            )
            ax.set_title("Confusion Matrix")
            ax.set_xlabel("Predicted Values")
            ax.set_ylabel("Actual Values")

            ## Ticket labels - List must be in alphabetical order
            ax.xaxis.set_ticklabels(["0", "1", "2", "3", "4"])
            ax.yaxis.set_ticklabels(["0", "1", "2", "3", "4"])

            ## Display the visualization of the Confusion Matrix.
            plt.savefig("confmat.png")

    def plot_roc(self, roc, auc):
        plt.clf()
        fpr, tpr, thresholds = roc

        auc = auc.cpu().numpy()

        print("auc", auc)

        fpr = [x.cpu().numpy() for x in fpr]
        tpr = [x.cpu().numpy() for x in tpr]
        thresholds = [x.cpu().numpy() for x in thresholds]

        for i in range(1, self.num_classes):
            f = fpr[i]
            t = tpr[i]
            J = t - f
            best_thresh_i = np.argmax(J)
            best_thresh = thresholds[i][best_thresh_i]
            print(f"best_threshold_{i}", best_thresh)
            plt.plot(
                f,
                t,
                label=f"ROC {i - 1} (AUC={auc[i]:.3f})",
            )

        plt.plot([0, 1], [0, 1], "k--")
        plt.xlim([-0.05, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver operating characteristic")
        plt.legend(loc="lower right")
        plt.savefig("roc.png")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
