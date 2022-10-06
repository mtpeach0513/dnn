import argparse
import json
import os

import optuna
from optuna.integration import PyTorchLightningPruningCallback
import pytorch_lightning as pl
import torch

from data.dataloader import MyDataModule, MyDataset
from models.resnet import ResNet
from models.tabnet import TabNetModel
from models.transformer import TransformerModel
from utils.configure import Config

import warnings
warnings.filterwarnings('ignore')


class HparamsTuning:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--model', type=str, required=True, choices=['resnet', 'tabnet', 'transformer'], help='model of experiment')
        parser.add_argument('--data_path', type=str, default='train.csv', help='data file')
        parser.add_argument('--test_path', type=str, default='test.csv', help='test data file')
        parser.add_argument('--location', type=str, required=True, help='location of experiment')
        parser.add_argument(
            "--pruning",
            "-p",
            action="store_true",
            help="Activate the pruning feature. `MedianPruner` stops unpromising "
                 "trials at the early stages of training.",
        )
        parser.add_argument('--max_epochs', type=int, default=100, help='max epochs')
        parser.add_argument('--n_trials', type=int, default=100, help='num of trials hparams tuning')
        args = parser.parse_args()
        args.root_path = os.path.join('data', args.location)

        print('\n========================================\n')
        print('Args in experiment:')
        print(args)
        print('\n========================================\n')

        self.model = args.model
        self.root_path = args.root_path
        self.data_path = args.data_path
        self.test_path = args.test_path
        self.location = args.location
        self.max_epochs = args.max_epochs
        self.n_trials = args.n_trials
        self.pruning = args.pruning
        self.conf = Config()

    def objective(self, trial: optuna.trial.Trial) -> float:
        dataset = MyDataset(root_path=self.root_path, data_path=self.data_path)
        datamodule = MyDataModule(
            config=self.conf, root_path=self.root_path,
            data_path=self.data_path, test_path=self.test_path
        )
        input_dim = dataset.data_x.shape[1]

        if self.model == 'resnet':
            # ResNet params
            dim = trial.suggest_int('dim', 4, 32, 4)
            hidden_factor = trial.suggest_int('hidden_factor', 1, 4)
            n_layers = trial.suggest_int('n_layers', 1, 64)
            activation = trial.suggest_categorical('activation', ['reglu', 'geglu', 'relu', 'gelu'])
            normalization = trial.suggest_categorical('normalization', ['batchnorm', 'layernorm'])
            dropout = trial.suggest_discrete_uniform('dropout', 0, 0.5, 0.1)
            r_dropout = trial.suggest_discrete_uniform('r_dropout', 0, 0.5, 0.1)

            model = ResNet(
                d_numerical=input_dim, d=dim, d_hidden_factor=hidden_factor,
                n_layers=n_layers, activation=activation, normalization=normalization,
                hidden_dropout=dropout, residual_dropout=r_dropout
            )

            hyperparameters = dict(
                dim=dim, hidden_factor=hidden_factor, n_layers=n_layers,
                activation=activation, normalization=normalization,
                dropout=dropout, r_dropout=r_dropout
            )

        elif self.model == 'tabnet':
            # TabNet params
            n_d_a = trial.suggest_int('n_d', 4, 64)
            n_steps = trial.suggest_int('n_steps', 3, 10)
            gamma = trial.suggest_discrete_uniform('gamma', 1.0, 2.0, 0.1)
            n_independent = trial.suggest_int('n_independent', 1, 5)
            n_shared = trial.suggest_int('n_shared', 1, 5)
            lambda_sparse = trial.suggest_loguniform('lambda_sparse', 1e-6, 1e-3)
            mask_type = trial.suggest_categorical('mask_type', ['sparsemax', 'entmax'])

            model = TabNetModel(
                input_dim=input_dim, n_d=n_d_a, n_a=n_d_a, n_steps=n_steps, gamma=gamma,
                n_independent=n_independent, n_shared=n_shared,
                lambda_sparse=lambda_sparse, mask_type=mask_type
            )
            datamodule.batch_size = 1024

            hyperparameters = dict(
                n_d=n_d_a, n_a=n_d_a, n_steps=n_steps, gamma=gamma,
                n_independent=n_independent, n_shared=n_shared,
                lambda_sparse=lambda_sparse, mask_type=mask_type
            )

        else:
            n_layers = trial.suggest_int('n_layers', 1, 4)
            d_token = trial.suggest_int('d_token', 64, 512, 64)
            residual_dropout = trial.suggest_discrete_uniform('residual_dropout', 0, 0.2, 0.1)
            attn_dropout = trial.suggest_discrete_uniform('attn_dropout', 0, 0.5, 0.1)
            ffn_dropout = trial.suggest_discrete_uniform('ffn_dropout', 0, 0.5, 0.1)
            d_ffn_factor = trial.suggest_discrete_uniform('d_ffn_factor', 2/3, 8/3, 1/3)

            model = TransformerModel(
                d_numerical=input_dim, categories=None,
                n_layers=n_layers, d_token=d_token, d_ffn_factor=d_ffn_factor,
                attn_dropout=attn_dropout, ffn_dropout=ffn_dropout, residual_dropout=residual_dropout
            )

            hyperparameters = dict(
                n_layers=n_layers, d_token=d_token, d_ffn_factor=d_ffn_factor,
                attn_dropout=attn_dropout, ffn_dropout=ffn_dropout, residual_dropout=residual_dropout
            )

        print('\n========================================\n')
        print('Hyper parameters in experiment:')
        print(hyperparameters)
        print('\n========================================\n')
        trainer = pl.Trainer(
            logger=True,
            enable_checkpointing=False,
            gpus=1 if torch.cuda.is_available() else None,
            deterministic=True,
            max_epochs=self.max_epochs,
            #callbacks=[PyTorchLightningPruningCallback(trial, monitor='val_loss')],
        )

        trainer.logger.log_hyperparams(hyperparameters)
        trainer.fit(model, datamodule)
        return trainer.callback_metrics['val_loss'].item()

    def run(self):
        pruner: optuna.pruners.BasePruner = (
            optuna.pruners.MedianPruner() if self.pruning else optuna.pruners.NopPruner()
        )

        study = optuna.create_study(pruner=pruner)
        study.optimize(self.objective, n_trials=self.n_trials)
        print(f'Number of finished trials: {len(study.trials)}')
        print('Best trials:')
        trial = study.best_trial
        print(f'    Value: {trial.value}')
        print(f'    Params:')
        for key, value in trial.params.items():
            print(f'    {key}: {value}')
        root_dir = os.path.join('lightning_logs', self.location)
        os.makedirs(root_dir, exist_ok=True)
        with open(os.path.join(root_dir, 'best_params.json'), mode='w') as f:
            json.dump(trial.params, f, indent=4)


if __name__ == '__main__':
    HparamsTuning().run()
