import os
import sys
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from utils.timefeatures import time_features
from sklearn.preprocessing import StandardScaler
#from utils.tools import StandardScaler
from utils.tools import reduce_memory_usage
from utils.configure import Config


class MyDataset(Dataset):
    def __init__(self, root_path='data', flag='train',
                 data_path='train.csv',
                 target='inflow', scale=True, inverse=False,
                 timeenc=0, freq='h', cols=None):
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
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
            train = int(num_id * .7)
            test = int(num_id * .2)
            valid = num_id - train - test
            train_list, test_list, valid_list = [], [], []
            for i in df_raw.id.unique()[:train]:
                df = df_raw[df_raw.id == i]
                train_list.append(df)
            df_train = pd.concat(train_list, axis=0)
            num_train = len(df_train)
            for i in df_raw.id.unique()[train:train + valid]:
                df = df_raw[df_raw.id == i]
                valid_list.append(df)
            df_valid = pd.concat(valid_list, axis=0)
            num_valid = len(df_valid)
            for i in df_raw.id.unique()[-test:]:
                df = df_raw[df_raw.id == i]
                test_list.append(df)
            df_test = pd.concat(test_list, axis=0)
            num_test = len(df_test)
            df_raw.pop('id')
        else:
            num_train = int(len(df_raw) * .7)
            num_test = int(len(df_raw) * .2)
            num_valid = len(df_raw) - num_train - num_test
        border1s = [0, num_train, len(df_raw) - num_test]
        border2s = [num_train, num_train + num_valid, len(df_raw)]
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

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.data_x = data[border1:border2, :-1].astype(np.float32)
        if self.inverse:
            self.data_y = df_data.to_numpy()[border1:border2, -1].astype(np.float32)
        else:
            self.data_y = data[border1:border2, -1].astype(np.float32)
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        seq_x = self.data_x[index]
        if self.inverse:
            seq_y = np.concatenate([self.data_x[index], self.data_y[index]], 0)
        else:
            seq_y = self.data_y[index]
        seq_mark = self.data_stamp[index]
        return seq_x, seq_y, seq_mark

    def __len__(self):
        return len(self.data_x)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class MyDataset_Pred(Dataset):
    def __init__(self, root_path='data', flag='pred',
                 data_path='test.csv',
                 target='inflow', scale=True, inverse=False,
                 timeenc=0, freq='h', cols=None):
        assert flag in ['pred']
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
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

        tmp_stamp = df_raw[['date']]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.to_numpy())
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq[-1:])

        self.data_x = data[:, :-1].astype(np.float32)
        if self.inverse:
            self.data_y = df_data.to_numpy()[:, -1].astype(np.float32)
        else:
            self.data_y = data[:, -1].astype(np.float32)
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        seq_x = self.data_x[index]
        if self.inverse:
            seq_y = np.concatenate([self.data_x[index], self.data_y[index]], 0)
        else:
            seq_y = self.data_y[index]
        seq_mark = self.data_stamp[index]
        return seq_x, seq_y, seq_mark

    def __len__(self):
        return len(self.data_x)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


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
            self.dataset_test = MyDataset(flag='test')
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
