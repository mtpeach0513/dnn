import time
from typing import Tuple
import numpy as np
import pandas as pd


# [年, 月, 日, 時] -> [year, month, day, hour]に変換
# [year, month, day, hour] の4列をdate列として1列にまとめ
# date列をDateTimeIndexとしてDataFrameのindexに設定する関数
def to_datetime_idx(dataframe: pd.DataFrame) -> pd.DataFrame:
    # 年, 月, 日, 時 カラムを読込
    raw_date_cols = list(dataframe.columns[:4])     # 変更前のカラム名
    date_cols = ['year', 'month', 'day', 'hour']    # 変更後のカラム名
    # raw_date_colsの各要素がkey, date_colsの各要素がvalueとなるdictを作成
    d = {}
    for d1, d2 in zip(raw_date_cols, date_cols):
        d[d1] = d2
    # 作成したdictを元に、引数となるDataFrameの列の名前を変更
    dataframe = dataframe.rename(columns=d)
    # 名前を変更した4列を1列にまとめ、DateTimeIndexとしてDataFrameのindexに設定
    dataframe.index = pd.to_datetime(dataframe[date_cols])
    dataframe.index.name = 'date'
    dataframe = dataframe.drop(date_cols, axis=1)
    # すべての列をfloat32に変更(float64よりfloat32のほうがpytorch的に扱いやすい)
    dataframe = dataframe.astype(np.float32)
    return dataframe


# 年情報からトレーニングデータとテストデータを分割する関数
def train_test_split(dataframe: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # 流入量と雨量の名前も変更
    dataframe = dataframe.rename(
        columns={'流入量conv': 'inflow', '解析雨量全流域': 'rain'}
    )
    # 欠測(-999.0埋め)データを除去
    dataframe = dataframe[
        (dataframe['inflow'] >= 0) & (dataframe['rain'] >= 0)
        ]
    # 2010年から2017年までのデータをトレーニングデータにする
    train_data = dataframe['2010':'2017']
    # 2018年以降のデータをテストデータにする
    test_data = dataframe['2018':]
    return train_data, test_data


# データを過去方向からシフトさせた新たな列を追加する関数
def shift_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    cols, names = [], []
    # 1時間ごとに過去6時間までシフト列を作成
    # ex. rain(t-n)列には、rain列のn行前のデータが入る
    for i in [1, 2, 3, 4, 5, 6]:
        cols.append(dataframe['rain'].shift(i))
        names += [f'rain (t-{i})']
    res = pd.concat(cols, axis=1)
    res.columns = names
    new_data = pd.concat([dataframe, res], axis=1)
    return new_data


# 積算したデータを新たな列として追加する関数
def cumsum_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    cols, names = [], []
    # 3～24時間積算データを作成
    for i in [3, 6, 12, 24]:
        cols.append(dataframe['rain'].rolling(i, min_periods=1).sum())
        names += [f'rain cumsum (t={i})']
    res = pd.concat(cols, axis=1)
    res.columns = names
    new_data = pd.concat([dataframe, res], axis=1)
    return new_data


# DateTimeIndexをindexに持つDataFrameから6月～11月のデータだけを切り取る関数
def cropping(dataframe: pd.DataFrame) -> pd.DataFrame:
    data_list = []
    begin = dataframe.index[0].year   # forループの始点となる年情報
    end = dataframe.index[-1].year    # forループの終点となる年情報
    for year in range(begin, end+1):
        # 年ごとに6月～11月のデータだけ抽出し空のリストに追加
        # ついでに年ごとにid列を追加しidを振っておく
        d = dataframe[f'{year}-06':f'{year}-11'].copy()
        d['id'] = year - begin + 1
        data_list.append(d)
    cropped = pd.concat(data_list, axis=0)
    cropped = cropped[['id'] + list(dataframe.columns)]
    return cropped


# シフト、積算、切り取りを行う関数
def to_dataset(dataframe: pd.DataFrame, target: str = 'inflow') -> pd.DataFrame:
    shifted = shift_data(dataframe)
    cumsum = cumsum_data(shifted)
    res = cropping(cumsum)
    cols = list(res.columns)
    # 目的変数の列をいちばん右に移動
    cols.remove(target)
    if 'date' in cols:
        cols.remove('date')
        return res[['date'] + cols + [target]]
    else:
        return res[cols + [target]]


if __name__ == '__main__':
    start = time.time()

    df = pd.read_csv(
        'miharu3.txt', delimiter=',',
        # [年, 月, 日, 時, 流入量conv, 解析雨量全流域] のみをtxtファイルから読込
        usecols=[0, 1, 2, 3, 5, 7],
        header=0,
        #names=['year', 'month', 'day', 'hour', 'inflow', 'rain']
    )
    new = to_datetime_idx(df)
    tr, te = train_test_split(new)
    train = to_dataset(tr); test = to_dataset(te)
    train.to_csv('train.csv'); test.to_csv('test.csv')

    print(f'elapsed time: {time.time() - start:.2f} sec')
