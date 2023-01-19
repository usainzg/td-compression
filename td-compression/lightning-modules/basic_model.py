import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicModel(pl.LightningModule):
    """
    https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#train-epoch-level-metrics
    """

    def __init__(self, model, lr) -> None:
        super().__init__()
        self.model = model
        self.lr = lr

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)
