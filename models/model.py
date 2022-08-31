import torch
import torch.nn as nn
import pytorch_lightning as pl


class BasicNet(pl.LightningModule):
    def __init__(self, input_dim):
        super(BasicNet, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.Dropout(p=0.2),
            nn.LeakyReLU(),
            nn.Linear(32, 1)
        )
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, on_epoch=True)
        return loss


class ElasticLinear(pl.LightningModule):
    def __init__(self, loss_fn, n_inputs:int=1, learning_rate=0.05, l1_lambda=0.05, l2_lambda=0.05):
        super(ElasticLinear, self).__init__()

        self.learning_rate = learning_rate
        self.loss_fn = loss_fn
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.output_layer = torch.nn.Linear(n_inputs, 1)
        self.train_log = []

    def forward(self, x):
        outputs = self.output_layer(x)
        return outputs

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def l1_reg(self):
        l1_norm = self.output_layer.weight.abs().sum()
        return self.l1_lambda * l1_norm

    def l2_reg(self):
        l2_norm = self.output_layer.weight.pow(2).sum()
        return self.l2_lambda * l2_norm

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y) + self.l1_reg() + self.l2_reg()

        self.log('loss', loss)
        self.train_log.append(loss.detach().numpy())
        return loss
