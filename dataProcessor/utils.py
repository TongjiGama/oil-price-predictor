import numpy as np
import pandas as pd 
from pandas import DataFrame 
from torch.utils.data import Dataset, DataLoader


 



# 将 DataFrame 转换为 X 和 y 集合的函数
def create_data_set(
    _data_set:DataFrame, # DataFrame 形式的数据集
    look_back:int = 1, # 保留几天用于后续预测
    ):
    X = _data_set.iloc[:-look_back,:-1].values # 除最后一列外的所有列作为特征
    y = _data_set.iloc[:-look_back,-1].values  # 最后一列作为标签
    return X, y    

# 将Da
def get_dataloader(
    dataset, 
    batch_size=16, 
    shuffle=True,
    num_workers =4,
    ):
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        num_workers= num_workers,
        )
    return dataloader