import time

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from models.model import BasicNet
from models.mlp import MLP
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

    pl.seed_everything(42)
    data_module = MyDataModule(conf)
    #model = BasicNet(input_dim)
    model = MLP(input_dim, [128, 64, 32], 0.2)

    model_checkpoint = ModelCheckpoint(
        dirpath='lightning_logs',
        filename='{epoch}-{val_loss:.3f}',
        monitor='val_loss',
        mode='min',
        save_top_k=3,
        save_last=True,
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
        deterministic=True,
        callbacks=[
            model_checkpoint,
            #early_stopping
        ],
        max_epochs=500,
        min_epochs=conf.num_epochs,
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
