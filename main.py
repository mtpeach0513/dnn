import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from models.mlp import MLP
from models.resnet import ResNet
from models.tabnet import TabNetModel
from data.dataloader import MyDataModule, MyDataset
from utils.configure import Config

import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model', type=str, default='mlp', choices=['mlp', 'resnet', 'tabnet'], help='model of experiment')
    parser.add_argument(
        '--stage', type=str, default='train', choices=['train', 'fit', 'test', 'pred', 'predict'],
        help='model behaviour stage, option: [train(fit), test(pred, predict)]')

    parser.add_argument('--data_path', type=str, default='train.csv', help='data file')
    parser.add_argument('--test_path', type=str, default='test.csv', help='test data file')

    # MLP hparams
    parser.add_argument('--layers_dim', nargs='+', type=int, default=[64, 32], help='dimensions of model layers')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout prob')

    # ResNet hparams
    parser.add_argument('--dim', type=int, default=32, help='dimensions of first layer')
    parser.add_argument('--hidden_factor', type=int, default=4, help='dimensional factor of hidden layers')
    parser.add_argument('--n_layers', type=int, default=3, help='layer size')
    parser.add_argument('--activation', type=str, default='reglu', choices=['reglu', 'geglu', 'relu', 'gelu'])
    parser.add_argument('--normalization', type=str, default='batchnorm', choices=['batchnorm', 'layernorm'])
    parser.add_argument('--r_dropout', type=float, default=0.2, help='residual dropout prob')

    # TabNet hparms
    parser.add_argument('--n_da', type=int, default=8,
                        help='n_d: width of the decision prediction layer.'
                             'n_a: witdh of the attention embedding for each mask. both is same value.')
    parser.add_argument('--n_steps', type=int, default=3, help='number of steps in the architecture')
    parser.add_argument('--gamma', type=float, default=1.3,
                        help='this is coefficient for feature reusage in the masks.'
                             'a value close to 1 will make mask selection least correlated between layers.')
    parser.add_argument('--n_independent', type=int, default=2, help='number of independent GLU layers at each step')
    parser.add_argument('--n_shared', type=int, default=2, help='number of shared GLU at each step')
    parser.add_argument('--lambda_sparse', type=float, default=1e-3,
                        help='the bigger this coefficient is,'
                             ' the sparser your model will be in terms of feature selection')
    parser.add_argument('--mask_type', type=str, default='sparsemax', choices=['sparsemax', 'entmax'],
                        help='this is the masking function to use for selecting features')

    # experiment params
    parser.add_argument('--max_epochs', type=int, default=100, help='max epochs')
    parser.add_argument('--location', type=str, required=True, help='location of experiment')
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

    data_module = MyDataModule(conf, root_path=args.root_path, data_path=args.data_path, test_path=args.test_path)

    # when train the model
    if args.stage in ['train', 'fit']:
        if args.model == 'mlp':
            model = MLP(args.input_dim, args.layers_dim, args.dropout)
        elif args.model == 'resnet':
            model = ResNet(
                d_numerical=args.input_dim, d=args.dim, d_hidden_factor=args.hidden_factor,
                n_layers=args.n_layers, activation=args.activation, normalization=args.normalization,
                hidden_dropout=args.dropout, residual_dropout=args.r_dropout
            )
        else:
            model = TabNetModel(
                input_dim=args.input_dim, n_d=args.n_da, n_a=args.n_da, n_steps=args.n_steps,
                gamma=args.gamma, n_independent=args.n_independent, n_shared=args.n_shared,
                lambda_sparse=args.lambda_sparse, mask_type=args.mask_type
            )
            data_module.batch_size = 1024

        logger = TensorBoardLogger(save_dir='lightning_logs', name=args.location, version=args.version)
        log_dir = logger.log_dir
        ckpt_ver = f'version_{logger.version}'
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
                ckpt_ver = 'version_0'
            else:
                ckpt_ver = max(existing_versions)
                ckpt_ver = f'version_{ckpt_ver}'
        else:
            version = args.version if isinstance(args.version, str) else f'version_{args.version}'
            if os.path.isdir(os.path.join(root_dir, version)):
                ckpt_ver = version
            else:
                raise FileNotFoundError('The directory where the checkpoint file is saved does not exist.')

        if args.model == 'mlp':
            model = MLP.load_from_checkpoint(
                f'lightning_logs/{args.location}/{ckpt_ver}/last.ckpt',
                input_dim=args.input_dim)
        elif args.model == 'resnet':
            model = ResNet.load_from_checkpoint(
                f'lightning_logs/{args.location}/{ckpt_ver}/last.ckpt',
                d_numerical=args.input_dim)
        else:
            model = TabNetModel.load_from_checkpoint(
                f'lightning_logs/{args.location}/{ckpt_ver}/last.ckpt',
                input_dim=args.input_dim)
            data_module.batch_size = 1024

        logger = False
        log_dir = f'lightning_logs/{args.location}'

    ld = '-'.join(map(str, args.layers_dim))
    try:
        div = os.path.splitext(args.data_path)[0].split('_')[1]
    except IndexError:
        div = 'all'
    if args.model == 'mlp':
        model_name = f'{args.model}_{args.location}_{div}_ld{ld}_{ckpt_ver}'
    elif args.model == 'resnet':
        model_name = f'{args.model}_{args.location}_{div}_dim{args.dim}_hf{args.hidden_factor}' \
                     f'_nl{args.n_layers}_{ckpt_ver}'
    else:
        model_name = f'{args.model}_{args.location}_{div}_nda{args.n_da}_ns{args.n_steps}' \
                     f'_{args.mask_type}_{ckpt_ver}'
    print(f'model name: {model_name}')
    print('\n========================================\n')

    model_checkpoint = ModelCheckpoint(
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
        max_epochs=args.max_epochs,
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
    predictions = predictions.mask(predictions <= 0, 0)
    output = pd.concat([df_test, predictions], axis=1)
    print(f'\n========================================\n')
    output_dir = os.path.join('output', args.location)
    os.makedirs(output_dir, exist_ok=True)
    output.to_csv(os.path.join(output_dir, f'{model_name}.csv'), encoding='utf-8-sig')
    print(f'output file was saved. output/{args.location}/{model_name}.csv')
    print(f'elapsed time: {time.time() - start:.2f} [sec]')
