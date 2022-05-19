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
        loss = loss_fn(out, label)
        return loss

    def validation_step(self, batch, batch_idx):
        # this is the test loop
        loss_fn = MixedLoss(10.0, 2.0)
        img = batch["image"]
        label = batch["label"]
        out = self.unet(img)
        loss = loss_fn(out, label)
        return loss

    def validation_epoch_end(self, validation_step_outputs):
        loss = torch.stack(validation_step_outputs).mean()
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
