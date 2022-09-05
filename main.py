import time

import numpy as np
import pandas as pd
import torch
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
        dirpath='lightning_logs',
        filename='{epoch}-{val_loss:.2f}',
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        save_last=False,
    )
    early_stopping = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=20,
    )

    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=1,
        auto_select_gpus=True,
        callbacks=[model_checkpoint, early_stopping],
        max_epochs=-1,
    )
    trainer.fit(model, data_module)
    trainer.test(model, data_module)
    predictions = trainer.predict(model, data_module)
    scaler = MyDataset().scaler_y
    predictions = scaler.inverse_transform(
        np.array(predictions).reshape(-1, 1)
    )
    np.save('prediction', predictions)
    print(f'elapsed time: {time.time() - start:.2f} [sec]')
