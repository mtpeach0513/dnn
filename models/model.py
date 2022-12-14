import torch
import torch.nn as nn
import pytorch_lightning as pl

from utils.configure import Config


class Ensemble(nn.Module):
    def __init__(self, modelA, modelB, modelC, input):
        super(Ensemble, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.modelC = modelC
        self.fc1 = nn.Linear(input, 1)

    def forward(self, x):
        out1 = self.modelA(x)
        out2 = self.modelB(x)
        out3 = self.modelC(x)

        out = out1 + out2 + out3
        x = self.fc1(out)
        x = x.squeeze(-1)
        return x

class BasicNet(pl.LightningModule):
    def __init__(self, input_dim):
        super(BasicNet, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Dropout(p=.2),
            nn.Linear(32, 1)
        )
        self.criterion = nn.MSELoss()
        #self.criterion = nn.SmoothL1Loss()

    def forward(self, x):
        return self.net(x).squeeze(1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=Config.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, min_lr=Config.min_lr, patience=10, verbose=True)
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
