import torch
import torch.nn as nn
import lightning.pytorch as pl
from torchmetrics.functional import accuracy


class Model(pl.LightningModule):
    def __init__(self, model, init_lr=1e-3):
        super().__init__()
        self.model = model
        self.loss = nn.CrossEntropyLoss()
        self.init_lr = init_lr

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        logits = self.model(x)
        return logits

    def training_step(self, batch, batch_idx):
        _, loss, acc = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        self.log('train_loss', loss)
        self.log('train_accuracy', acc)
        return loss

    def validation_step(self, batch, batch_idx):
        preds, loss, acc = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        self.log('val_loss', loss)
        self.log('val_accuracy', acc)
        return preds
    
    def test_step(self, batch, batch_idx):
        _, loss, acc = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        self.log('test_loss', loss)
        self.log('test_accuracy', acc)

    def _get_preds_loss_accuracy(self, batch):
        '''convenience function since train/valid/test steps are similar'''
        x, y = batch
        logits = self.model(x)
        preds = torch.argmax(logits, dim=1)
        loss = self.loss(logits, y)
        acc = accuracy(preds, y, 'multiclass', num_classes=10)
        return preds, loss, acc

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.init_lr)
        return optimizer