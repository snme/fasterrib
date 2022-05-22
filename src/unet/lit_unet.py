import pytorch_lightning as pl
import torch
from src.unet.loss import MixedLoss
from src.unet.unet import UNet
from torchmetrics import F1Score


class LitUNet(pl.LightningModule):
    def __init__(self, unet: UNet, class_counts: torch.Tensor):
        super().__init__()
        self.unet = unet
        self.class_counts = class_counts
        self.f1 = F1Score(num_classes=6, average="none")

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        loss_fn = MixedLoss(10.0, 2.0)
        img = batch["image"]
        label = batch["label"]
        out = self.unet(img)
        loss, dice, ce = loss_fn(out, label, class_counts=self.class_counts)
        self.log("train_loss", loss, prog_bar=True)
        if not torch.isnan(dice):
            self.log("train_dice", dice, prog_bar=True)
        self.log("train_ce", ce, prog_bar=True)

        y_pred = torch.argmax(out, dim=1).flatten()
        y_target = torch.argmax(label, dim=1).flatten()
        f1 = self.f1(y_pred, y_target)
        self.log("train_f1", {i: c for (i, c) in enumerate(f1)}, prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):
        # this is the test loop
        loss_fn = MixedLoss(10.0, 2.0)
        img = batch["image"]
        label = batch["label"]
        out = self.unet(img)
        loss, dice, ce = loss_fn(out, label, class_counts=self.class_counts)

        y_pred = torch.argmax(out, dim=1).flatten()
        y_target = torch.argmax(label, dim=1).flatten()
        f1 = self.f1(y_pred, y_target)
        return loss, dice, ce, f1

    def validation_epoch_end(self, outputs):
        loss = torch.stack([x[0] for x in outputs]).mean()
        dice = torch.stack([x[1] for x in outputs]).nanmean()
        ce = torch.stack([x[2] for x in outputs]).mean()
        f1 = torch.stack([x[3] for x in outputs], dim=0).nanmean(dim=0)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_dice", dice, prog_bar=True)
        self.log("val_ce", ce, prog_bar=True)
        self.log("val_f1", {i: c for (i, c) in enumerate(f1)}, prog_bar=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
