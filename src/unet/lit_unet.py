import pytorch_lightning as pl
import torch
from src.unet.loss import MixedLoss
from src.unet.unet import UNet


class LitUNet(pl.LightningModule):
    def __init__(self, unet: UNet, class_counts: torch.Tensor):
        super().__init__()
        self.unet = unet
        self.class_counts = class_counts

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        loss_fn = MixedLoss(10.0, 2.0)
        img = batch["image"]
        label = batch["label"]
        out = self.unet(img)
        loss, dice, ce = loss_fn(out, label, class_counts=self.class_counts)
        self.log("train_loss", loss, prog_bar=True)
        if dice:
            self.log("train_dice", dice, prog_bar=True)
        self.log("train_ce", ce, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # this is the test loop
        loss_fn = MixedLoss(10.0, 2.0)
        img = batch["image"]
        label = batch["label"]
        out = self.unet(img)
        loss, dice, ce = loss_fn(out, label, class_counts=self.class_counts)
        return loss, dice, ce

    def validation_epoch_end(self, outputs):
        loss = torch.stack([x[0] for x in outputs]).mean()
        ce = torch.stack([x[2] for x in outputs]).mean()

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_ce", ce, prog_bar=True)

        dices = [x[1] for x in outputs if x[1]]
        if len(dices) > 0:
            dice = torch.stack(dices).mean()
            self.log("val_dice", dice, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-6)
        return optimizer
