import math
from typing import cast, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import pytorch_lightning as pl

from utils.configure import Config
from utils.tools import get_activation_fn, get_nonglu_activation_fn


class ResNet(pl.LightningModule):
    def __init__(self, *,
                 d_numerical: int, categories: Optional[List[int]] = None, d_embedding: int = None,
                 d: int, d_hidden_factor: float, n_layers: int,
                 activation: str, normalization: str,
                 hidden_dropout: float, residual_dropout: float, d_out: int = 1) -> None:
        super(ResNet, self).__init__()
        self.save_hyperparameters()

        def make_normalization():
            return {'batchnorm': nn.BatchNorm1d, 'layernorm': nn.LayerNorm}[
                normalization
            ](d)

        self.main_activation = get_activation_fn(activation)
        self.last_activation = get_nonglu_activation_fn(activation)
        self.residual_dropout = residual_dropout
        self.hidden_dropout = hidden_dropout

        d_in = d_numerical
        d_hidden = int(d * d_hidden_factor)

        if categories is not None:
            d_in = len(categories) * d_embedding
            category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
            self.register_buffer('category_offsets', category_offsets)
            self.category_embeddings = nn.Embedding(sum(categories), d_embedding)
            nn.init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))
            print(f'{self.category_embeddings.weight.shape}')

        self.first_layer = nn.Linear(d_in, d)
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        'norm': make_normalization(),
                        'linear0': nn.Linear(
                            d, d_hidden * (2 if activation.endswith('glu') else 1)
                        ),
                        'linear1': nn.Linear(d_hidden, d),
                    }
                )
                for _ in range(n_layers)
            ]
        )
        self.last_normalization = make_normalization()
        self.head = nn.Linear(d, d_out)
        self.criterion = nn.MSELoss()

    def forward(self, x: Tensor) -> Tensor:
        x = self.first_layer(x)
        for layer in self.layers:
            layer = cast(Dict[str, nn.Module], layer)
            z = x
            z = layer['norm'](z)
            z = layer['linear0'](z)
            z = self.main_activation(z)
            if self.hidden_dropout:
                z = F.dropout(z, p=self.hidden_dropout, training=self.training)
            z = layer['linear1'](z)
            if self.residual_dropout:
                z = F.dropout(z, p=self.residual_dropout, training=self.training)
            x = x + z
        x = self.last_normalization(x)
        x = self.last_activation(x)
        x = self.head(x)
        x = x.squeeze(-1)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.RAdam(self.parameters(), lr=Config.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.1, min_lr=0, verbose=False
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
    model = ResNet(
        d_numerical=70, d=32, d_hidden_factor=4, n_layers=2,
        activation='reglu', normalization='layernorm', hidden_dropout=0.5, residual_dropout=0.2)
    print(model)
