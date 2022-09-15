import time

import numpy as np
import pytorch_lightning as pl

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

    model = BasicNet.load_from_checkpoint(
        'lightning_logs/version_5/last.ckpt',
        input_dim=input_dim
    )
    trainer = pl.Trainer(deterministic=True, logger=False)
    predictions = trainer.predict(model, data_module)
    scaler = MyDataset().scaler_y
    predictions = scaler.inverse_transform(
        np.array(predictions).reshape(-1, 1)
    )
    np.save('prediction', predictions)
    print(f'elapsed time: {time.time() - start:.2f} [sec]')
