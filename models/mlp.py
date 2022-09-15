from typing import Optional, List
import torch
import torch.nn as nn
import pytorch_lightning as pl

from utils.configure import Config


class MLP(pl.LightningModule):
    def __init__(self,
                 input_dim: int, layers_dim: List[int], dropout: float) -> None:
        super(MLP, self).__init__()

        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(layers_dim[i-1] if i else input_dim, x),
                    nn.BatchNorm1d(x),
                    nn.LeakyReLU(),
                    nn.Dropout(dropout),
                )
                for i, x in enumerate(layers_dim)
            ]
        )
        self.head = nn.Linear(layers_dim[-1] if layers_dim else input_dim, 1)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.head(x)
        x = x.squeeze(1)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=Config.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, min_lr=0, verbose=True
        )
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('test_loss', loss)

    def predict_step(self, batch, batch_idex, dataloader_idx=0):
        x, y = batch
        y_hat = self(x)
        return y_hat


if __name__ == '__main__':
    model = MLP(8, [128, 64, 32], 0.5)
    print(model)
