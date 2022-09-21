import argparse
import os
import time
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from models.mlp import MLP
from data.dataloader import MyDataModule, MyDataset
from utils.configure import Config

import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='mlp', help='model of experiment, options: [mlp]')
    parser.add_argument(
        '--stage', type=str, default='train', choices=['train', 'fit', 'test', 'pred', 'predict'],
        help='model behaviour stage, option: [train(fit), test(pred, predict)]')

    parser.add_argument('--data_path', type=str, default='train.csv', help='data file')
    parser.add_argument('--test_path', type=str, default='test.csv', help='test data file')

    parser.add_argument('--layers_dim', type=List[int], default=[64, 32], help='dimensions of model layers')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout prob')
    parser.add_argument('--location', type=str, default='miharu', help='location of experiment')
    parser.add_argument('--version', type=int, default=None, help='experiment version')
    args = parser.parse_args()
    args.root_path = os.path.join('data', args.location)

    conf = Config()

    print('\n========================================\n')
    dataset = MyDataset(root_path=args.root_path, data_path=args.data_path)
    args.input_dim = dataset.data_x.shape[1]
    print('Args in experiment:')
    print(args)
    print('\n========================================\n')

    pl.seed_everything(42)

    data_module = MyDataModule(conf, root_path=args.root_path, data_path=args.data_path)

    # when train the model
    if args.stage in ['train', 'fit']:
        model = MLP(args.input_dim, args.layers_dim, args.dropout)
        logger = TensorBoardLogger(save_dir='lightning_logs', name=args.location, version=args.version)
        log_dir = logger.log_dir
    # when you want the model to predict the test data
    else:
        root_dir = os.path.join('lightning_logs', args.location)
        if args.version is None:
            existing_versions = []
            for f in os.listdir(root_dir):
                if os.path.isdir(os.path.join(root_dir, f)) and f.startswith('version_'):
                    dir_ver = f.split('_')[1].replace('/', '')
                    existing_versions.append(int(dir_ver))
            if len(existing_versions) == 1:
                ckpt_ver = 0
            else:
                ckpt_ver = max(existing_versions)
        else:
            version = args.version if isinstance(args.version, str) else f'version_{args.version}'
            if os.path.isdir(os.path.join(root_dir, version)):
                ckpt_ver = version
            else:
                raise FileNotFoundError('The directory where the checkpoint file is saved does not exist.')

        model = MLP.load_from_checkpoint(
            f'lightning_logs/{args.location}/{ckpt_ver}/last.ckpt',
            input_dim=args.input_dim, layers_dim=args.layers_dim, dropout=args.dropout
        )
        logger = False
        log_dir = f'lightning_logs/{args.location}'

    ld = '-'.join(map(str, args.layers_dim))
    model_name = f'{args.model}_{args.location}_ld{ld}_{ckpt_ver}'

    model_checkpoint = ModelCheckpoint(
        #dirpath=f'lightning_logs/{args.location}',
        dirpath=log_dir,
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
        logger=logger,
        max_epochs=500,
        min_epochs=conf.num_epochs,
    )
    if args.stage in ['train', 'fit']:
        trainer.fit(model, data_module)
        trainer.test(model, data_module)
    else:
        pass

    predictions = trainer.predict(model, data_module)
    scaler = dataset.scaler_y
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    #np.save('prediction', predictions)
    df_test = pd.read_csv(os.path.join(args.root_path, args.test_path), index_col=0, usecols=['date', 'inflow'])
    predictions = pd.Series(predictions[:, 0], index=df_test.index, name='prediction')
    output = pd.concat([df_test, predictions], axis=1)
    print(f'\n========================================\n')
    output_dir = os.path.join('output', args.location)
    os.makedirs(output_dir, exist_ok=True)
    output.to_csv(os.path.join(output_dir, f'{model_name}.csv'), encoding='utf-8-sig')
    print(f'output file was saved. output/{model_name}.csv')
    print(f'elapsed time: {time.time() - start:.2f} [sec]')
