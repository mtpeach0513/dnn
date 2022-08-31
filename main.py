import pytorch_lightning as pl

from models.model import BasicNet
from data.dataloader import MyDataModule
from utils.configure import Config

import warnings
#warnings.filterwarnings('ignore')


if __name__ == '__main__':
    conf = Config()
    data = MyDataModule(conf)
    model = BasicNet(11)
    trainer = pl.Trainer()
    trainer.fit(model, data)
