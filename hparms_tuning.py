import argparse
import os

import optuna
from optuna.integration import PyTorchLightningPruningCallback
import pytorch_lightning as pl
import torch

from data.dataloader import MyDataModule, MyDataset
from models.resnet import ResNet
from utils.configure import Config


class HparamsTuning:
    def __init__(self):
        parser = argparse.ArgumentParser()
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
        args = parser.parse_args()
        args.root_path = os.path.join('data', args.location)
        self.root_path = args.root_path
        self.data_path = args.data_path
        self.test_path = args.test_path
        self.location = args.location
        self.pruning = args.pruning
        self.conf = Config()

    def objective(self, trial: optuna.trial.Trial) -> float:
        dim = trial.suggest_int('dim', 1, 8)
        hidden_factor = trial.suggest_int('hidden_factor', 1, 4)
        n_layers = trial.suggest_int('n_layers', 1, 64)
        activation = trial.suggest_categorical('activation', ['reglu', 'geglu', 'relu', 'gelu'])
        normalization = trial.suggest_categorical('normalization', ['batchnorm', 'layernorm'])
        dropout = trial.suggest_uniform('dropout', 0, 0.5)
        r_dropout = trial.suggest_uniform('r_dropout', 0, 0.5)
        dataset = MyDataset(root_path=self.root_path, data_path=self.data_path)
        input_dim = dataset.data_x.shape[1]
        model = ResNet(
            d_numerical=input_dim, d=dim, d_hidden_factor=hidden_factor,
            n_layers=n_layers, activation=activation, normalization=normalization,
            hidden_dropout=dropout, residual_dropout=r_dropout
        )
        datamodule = MyDataModule(
            config=self.conf, root_path=self.root_path,
            data_path=self.data_path, test_path=self.test_path
        )

        trainer = pl.Trainer(
            logger=True,
            enable_checkpointing=False,
            gpus=1 if torch.cuda.is_available() else None,
            deterministic=True,
            max_epochs=500,
            callbacks=[PyTorchLightningPruningCallback(trial, monitor='val_loss')],
        )
        hyperparameters = dict(
            dim=dim, hidden_factor=hidden_factor, n_layers=n_layers,
            activation=activation, normalization=normalization,
            dropout=dropout, r_dropout=r_dropout
        )
        trainer.logger.log_hyperparams(hyperparameters)
        trainer.fit(model, datamodule)
        return trainer.callback_metrics['val_loss'].item()

    def run(self):
        pruner: optuna.pruners.BasePruner = (
            optuna.pruners.MedianPruner() if self.pruning else optuna.pruners.NopPruner()
        )

        study = optuna.create_study(pruner=pruner)
        study.optimize(self.objective, n_trials=100, timeout=1800)
        print(f'Number of finished trials: {len(study.trials)}')
        print('Best trials:')
        trial = study.best_trial
        print(f'    Value: {trial.value}')
        print(f'    Params:')
        for key, value in trial.params.items():
            print(f'    {key}: {value}')


if __name__ == '__main__':
    hparmstuning = HparamsTuning()
    hparmstuning.run()
