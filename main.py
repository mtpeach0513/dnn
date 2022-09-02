import time

import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from models.model import BasicNet
from data.dataloader import MyDataModule, MyDataset
from utils.configure import Config

import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    start = time.time()
    conf = Config()

    print('========================================')
    input_dim = MyDataset().data_x.shape[1]
    print(f'input_dim: {input_dim}')
    print('========================================')

    data_module = MyDataModule(conf)
    model = BasicNet(input_dim)

    model_checkpoint = ModelCheckpoint(
        'logs',
        filename='{epoch}-{val_loss:.2f}',
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        save_last=False,
    )
    early_stopping = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=10,
    )

    trainer = pl.Trainer(
        callbacks=[model_checkpoint, early_stopping],
    )
    trainer.fit(model, data_module)
    trainer.predict(model, data_module)
    print(f'elapsed time: {time.time() - start:.2f} [sec]')
