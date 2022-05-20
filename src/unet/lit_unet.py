import pytorch_lightning as pl
import torch
from src.unet.loss import MixedLoss
from src.unet.unet import UNet


class LitUNet(pl.LightningModule):
    def __init__(self, unet: UNet):
        super().__init__()
        self.unet = unet

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        loss_fn = MixedLoss(10.0, 2.0)
        img = batch["image"]
        label = batch["label"]
        out = self.unet(img)
        loss, dice, ce = loss_fn(out, label)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_dice", dice, prog_bar=True)
        self.log("train_ce", ce, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # this is the test loop
        loss_fn = MixedLoss(10.0, 2.0)
        img = batch["image"]
        label = batch["label"]
        out = self.unet(img)
        loss, dice, ce = loss_fn(out, label)
        return loss, dice, ce

    def validation_epoch_end(self, outputs):
        loss = torch.stack([x[0] for x in outputs]).mean()
        dice = torch.stack([x[1] for x in outputs]).mean()
        ce = torch.stack([x[2] for x in outputs]).mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_dice", dice, prog_bar=True)
        self.log("val_ce", ce, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
