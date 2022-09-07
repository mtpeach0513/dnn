import time

import numpy as np
import pytorch_lightning as pl

from data.dataloader import MyKFoldDataModule, MyDatasetForKFold
from models.model import BasicNetForKFold

import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    start = time.time()

    print('========================================')
    input_dim = MyDatasetForKFold().data_x.shape[1]
    print(f'input_dim: {input_dim}')
    print('========================================')

    pl.seed_everything(42)
    datamodule = MyKFoldDataModule()
    scaler = MyDatasetForKFold().scaler_y
    preds = []
    for i in range(1, 6):
        model = BasicNetForKFold.load_from_checkpoint(
            f'lightning_logs/model.{i}.pt',
            input_dim=input_dim
        )
        trainer = pl.Trainer(deterministic=True)
        pred = trainer.predict(model, datamodule)
        predictions = scaler.inverse_transform(
            np.array(pred).reshape(-1, 1)
        )
        preds.append(predictions)
    prediction = np.mean(np.array(preds), axis=0)
    np.save('prediction.npy', prediction)

    print(f'elapsed time: {time.time() - start:.2f} [sec]')
