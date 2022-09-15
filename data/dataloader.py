import os
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
import sys
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import pytorch_lightning as pl

from utils.timefeatures import time_features
from utils.tools import reduce_memory_usage
from utils.configure import Config


# setup_foldsとsetup_fold_indexメソッドを持つ抽象クラス
# 抽象クラスによりポリモーフィズムを実装できる
class BaseKFoldDataModule(pl.LightningDataModule, ABC):
    @abstractmethod
    def setup_folds(self, num_folds: int) -> None:
        pass

    @abstractmethod
    def setup_fold_index(self, fold_index: int) -> None:
        pass


# トレーニングとテストデータセットを受け取る
# setup_foldsでは指定された引数num_foldsに応じてfoldが作成される
# setup_fold_indexにより、与えられたトレーニングデータセットが現在のfold数に従って分割される
@dataclass
class MyKFoldDataModule(BaseKFoldDataModule):

    train_dataset: Optional[Dataset] = None
    test_dataset: Optional[Dataset] = None
    predict_dataset: Optional[Dataset] = None
    train_fold: Optional[Dataset] = None
    val_fold: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = MyDatasetForKFold(flag='train')
        self.test_dataset = MyDatasetForKFold(flag='test')
        self.predict_dataset = MyDatasetForKFoldPredict(flag='pred')

    def setup_folds(self, num_folds: int) -> None:
        self.num_folds = num_folds
        self.splits = [split for split in KFold(num_folds).split(range(len(self.train_dataset)))]

    def setup_fold_index(self, fold_index: int) -> None:
        train_indices, val_indices = self.splits[fold_index]
        self.train_fold = Subset(self.train_dataset, train_indices)
        self.val_fold = Subset(self.train_dataset, val_indices)

    def train_dataloader(self) -> DataLoader:
        train_loader = DataLoader(
            self.train_fold,
            batch_size=Config.batch_size,
            shuffle=True,
            num_workers=Config.num_workers,
            pin_memory=Config.pin_memory,
            drop_last=True
        )
        return train_loader

    def val_dataloader(self) -> DataLoader:
        val_loader = DataLoader(
            self.val_fold
        )
        return val_loader

    def test_dataloader(self) -> DataLoader:
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=Config.batch_size,
            shuffle=False,
            num_workers=Config.num_workers,
            pin_memory=Config.pin_memory,
            drop_last=True
        )
        return test_loader

    def predict_dataloader(self) -> DataLoader:
        predict_loader = DataLoader(
            self.predict_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=Config.num_workers,
            pin_memory=Config.pin_memory,
            drop_last=False
        )
        return predict_loader

    def __post_init__(cls):
        super().__init__()


class MyDatasetForKFold(Dataset):
    def __init__(self, root_path='data', flag='train',
                 data_path='train.csv',
                 target='inflow', scale=True,
                 cols=None):
        assert flag in ['train', 'test']
        type_map = {'train': 0, 'test': 1}
        self.set_type = type_map[flag]
        self.target = target
        self.scale = scale
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        self.scaler_y = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        df_raw = reduce_memory_usage(df_raw)
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]

        if 'id' in df_raw.columns:
            num_id = df_raw.id.nunique()
            train = int(num_id * .8)
            num_train = len(df_raw[df_raw['id'] <= train])
            df_raw.pop('id')
        else:
            num_train = int(len(df_raw) * .8)
        border1s = [0, num_train]
        border2s = [num_train, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.to_numpy())
            self.scaler_y.fit(train_data.iloc[:, -1].to_numpy().reshape(-1, 1))
            data = self.scaler.transform(df_data.to_numpy())
        else:
            data = df_data.to_numpy()

        self.data_x = data[border1:border2, :-1].astype(np.float32)
        self.data_y = data[border1:border2, -1].astype(np.float32)

    def __getitem__(self, index):
        x = self.data_x[index]
        y = self.data_y[index]
        return x, y

    def __len__(self):
        return len(self.data_x)


class MyDatasetForKFoldPredict(Dataset):
    def __init__(self, root_path='data', flag='pred',
                 data_path='test.csv',
                 target='inflow', scale=True,
                 cols=None):
        assert flag in ['pred']
        self.target = target
        self.scale = scale
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = MyDatasetForKFold().scaler
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        df_raw = reduce_memory_usage(df_raw)
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]

        if 'id' in df_raw.columns:
            df_raw.drop('id', axis=1, inplace=True)

        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        if self.scale:
            data = self.scaler.transform(df_data.to_numpy())
        else:
            data = df_data.to_numpy()

        self.data_x = data[:, :-1].astype(np.float32)
        self.data_y = data[:, -1].astype(np.float32)

    def __getitem__(self, index):
        x = self.data_x[index]
        y = self.data_y[index]
        return x, y

    def __len__(self):
        return len(self.data_x)


class MyDataModule(pl.LightningDataModule):
    def __init__(self, config: Config):
        super(MyDataModule, self).__init__()
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.pin_memory = config.pin_memory

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            self.dataset_train = MyDataset(flag='train')
            self.dataset_val = MyDataset(flag='val')
        if stage == 'test' or stage is None:
            self.dataset_test = MyDataset_Pred(flag='test')
        if stage == 'predict' or stage is None:
            self.dataset_pred = MyDataset_Pred(flag='pred')

    def train_dataloader(self):
        train_loader = DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True
        )
        return val_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True
        )
        return test_loader

    def predict_dataloader(self):
        pred_loader = DataLoader(
            self.dataset_pred,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False
        )
        return pred_loader


class MyDataset(Dataset):
    def __init__(self, root_path='data', flag='train',
                 data_path='train.csv',
                 target='inflow', scale=True, cols=None):
        assert flag in ['train', 'val']
        type_map = {'train': 0, 'val': 1}
        self.set_type = type_map[flag]
        self.target = target
        self.scale = scale
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        self.scaler_y = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        df_raw = reduce_memory_usage(df_raw)
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]

        if 'id' in df_raw.columns:
            num_id = df_raw.id.nunique()
            train = int(num_id * .8)
            num_train = len(df_raw[df_raw['id'] <= train])
            df_raw.pop('id')
        else:
            num_train = int(len(df_raw) * .8)
        border1s = [0, num_train]
        border2s = [num_train, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.to_numpy())
            self.scaler_y.fit(train_data.iloc[:, -1].to_numpy().reshape(-1, 1))
            data = self.scaler.transform(df_data.to_numpy())
        else:
            data = df_data.to_numpy()

        self.data_x = data[border1:border2, :-1].astype(np.float32)
        self.data_y = data[border1:border2, -1].astype(np.float32)

    def __getitem__(self, index):
        x = self.data_x[index]
        y = self.data_y[index]
        return x, y

    def __len__(self):
        return len(self.data_x)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class MyDataset_Pred(Dataset):
    def __init__(self, root_path='data', flag='pred',
                 data_path='test.csv',
                 target='inflow', scale=True, cols=None):
        assert flag in ['test', 'pred']
        self.target = target
        self.scale = scale
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = MyDataset().scaler
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        df_raw = reduce_memory_usage(df_raw)
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]

        if 'id' in df_raw.columns:
            df_raw.drop('id', axis=1, inplace=True)

        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        if self.scale:
            data = self.scaler.transform(df_data.to_numpy())
        else:
            data = df_data.to_numpy()

        self.data_x = data[:, :-1].astype(np.float32)
        self.data_y = data[:, -1].astype(np.float32)

    def __getitem__(self, index):
        x = self.data_x[index]
        y = self.data_y[index]
        return x, y

    def __len__(self):
        return len(self.data_x)
