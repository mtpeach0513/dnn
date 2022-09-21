import argparse
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

import lightgbm as lgb
import numpy as np
import pandas as pd
import tqdm

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error


class Config:
    max_depth = 8
    num_leaves = int(.7 * max_depth ** 2)
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting': 'gbdt',
        'learning_rate': 0.01,
        'max_depth': max_depth,  # default -1 means no limit
        'num_leaves': num_leaves,  # default 31
        'min_child_samples': 30,  # default 20
        'max_bin': 5000,  # default 255 # ヒストグラムのビンの数
        'min_child_weight': 2,  # default 1e-3 [1, 5]
        'feature_fraction': 0.8,  # 特徴量サンプリングの割合（何％の特徴量を利用するか）
        'bagging_fraction': 0.8,  # バギングの割合（トレーニングデータの何％を利用するか）
        # 'bagging_freq': 1, # 1回の反復ごとに置換せずに再サンプリングし、トレーニングデータの80％のサンプルを抽出
        'lambda_l1': 1e-05,
        'lambda_l2': 1e-05,
        'random_state': 42,
        'verbose': -1
    }
    n_splits = 5


def prepare_dataset(root_path='data',
                    data_path='train.csv', test_path='test.csv',
                    target='inflow'):
    train = pd.read_csv(os.path.join(root_path, data_path), index_col=0)
    test = pd.read_csv(os.path.join(root_path, test_path), index_col=0)
    train_y = train.pop(target)
    test_y = test.pop(target)
    return train, test, train_y, test_y


def train(train_X, train_y, kf='kf'):
    assert kf in ['kf', 'group']
    models = []
    data_X = train_X if kf == 'kf' else train_X.id.unique()
    cv = KFold(n_splits=Config.n_splits, shuffle=True, random_state=42)
    for tr_idx, val_idx in tqdm.tqdm(cv.split(data_X), total=cv.get_n_splits(), desc='k-fold'):
        if kf == 'kf':
            X_tr, X_val = data_X.iloc[tr_idx], data_X.iloc[val_idx]
            y_tr, y_val = train_y.iloc[tr_idx], train_y.iloc[val_idx]
        else:
            # idをtrain/validに分割
            tr_groups, val_groups = train_X.id.unique()[tr_idx], train_X.id.unique()[val_idx]
            # 各レコードのidがtrain/validのどちらに属しているかによって分割
            is_tr, is_val = train_X.id.isin(tr_groups), train_X.id.isin(val_groups)
            X_tr, X_val = train_X.loc[is_tr], train_X.loc[is_val]
            y_tr, y_val = train_y.loc[is_tr], train_y.loc[is_val]
        lgb_train = lgb.Dataset(X_tr, y_tr)
        lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train)

        model = lgb.train(
            Config.params, lgb_train,
            valid_names=['train', 'valid'],
            valid_sets=[lgb_train, lgb_eval],
            verbose_eval=-1,
            num_boost_round=10000,
            early_stopping_rounds=20,
        )
        models.append(model)
    return models


def output_average_predict(models, X):
    y_preds = []
    for model in models:
        y_pred = model.predict(X, num_iteration=model.best_iteration)
        y_preds.append(y_pred)
    y_preds_average = np.mean(np.array(y_preds), axis=0)
    return y_preds_average


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='train.csv', help='data file')
    parser.add_argument('--test_path', type=str, default='test.csv', help='test data file')
    parser.add_argument('--target', type=str, default='inflow', help='target feature')
    parser.add_argument('--cv_type', type=str, default='group', help='type of cross validation, options:[kf, group]')
    parser.add_argument('--location', type=str, default='miharu', help='location of experiment')
    args = parser.parse_args()
    args.root_path = os.path.join('data', args.location)

    print(f'\n========================================\n')
    print('Args in experiment:')
    print(args)
    location = args.root_path.split('/')[-1]
    print(f'Location: {location}')
    model_name = f'lgbm_{args.location}_{os.path.splitext(args.data_path)[0]}_{args.target}_{args.cv_type}'
    model_dir = os.path.join('models', location)
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, model_name + '.pickle')

    train_X, test_X, train_y, test_y = prepare_dataset(root_path=args.root_path, data_path=args.data_path,
                                                       test_path=args.test_path, target=args.target)

    if os.path.exists(model_path):
        with open(model_path, mode='rb') as f:
            models = pickle.load(f)
    else:
        models = train(train_X, train_y, kf=args.cv_type)
        with open(model_path, mode='wb') as f:
            pickle.dump(models, f)

    print(f'\n========================================\n')
    print(f'Predict Data: {os.path.join(args.root_path, args.test_path)}')
    preds_average = output_average_predict(models, test_X)
    predictions = pd.Series(preds_average, index=test_y.index, name='predictions')
    output = pd.concat([test_y, predictions], axis=1)

    print(f'\n========================================\n')
    print(f'R2_SCORE: {r2_score(test_y, preds_average):.3f}')
    print(f'RMSE: {np.sqrt(mean_squared_error(test_y, preds_average)):.3f}')

    print(f'\n========================================\n')
    output_dir = os.path.join('output', location)
    os.makedirs(output_dir, exist_ok=True)
    output.to_csv(os.path.join(output_dir, f'{model_name}.csv'), encoding='utf-8-sig')
    print(f'output file was saved. output/{model_name}.csv')
    print(f'\n========================================\n')


if __name__ == '__main__':
    main()
