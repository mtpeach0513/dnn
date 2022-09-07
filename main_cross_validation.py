import pytorch_lightning as pl

from data.dataloader import MyKFoldDataModule, MyDatasetForKFold, MyDatasetForKFoldPredict
from models.model import BasicNetForKFold
from models.loop import KFoldLoop

import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':

    pl.seed_everything(42)

    input_dim = MyDatasetForKFold().data_x.shape[1]
    model = BasicNetForKFold(input_dim)
    datamodule = MyKFoldDataModule()

    trainer = pl.Trainer(
        max_epochs=1000,
        min_epochs=100,
        deterministic=True,
        limit_train_batches=1,
        limit_val_batches=1,
        limit_test_batches=1,
        num_sanity_val_steps=0,
        devices=1,
        accelerator="auto",
    )
    internal_fit_loop = trainer.fit_loop
    trainer.fit_loop = KFoldLoop(5, export_path="lightning_logs")
    trainer.fit_loop.connect(internal_fit_loop)
    trainer.fit(model, datamodule)
