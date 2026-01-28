import numpy as np
import pandas as pd 
import torch
from pandas import DataFrame 
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy as dc # 创建数据的深拷贝（完全独立的副本），修改副本不会影响原数据
from sklearn.preprocessing import MinMaxScaler # 从 scikit-learn 的预处理模块导入最小-最大缩放器，用于把数值特征线性缩放到给定范围


def prepare_dataframe_for_lstm(df, n_steps, inplace=False):
    """
    为LSTM准备时间序列数据
    
    参数:
    - df: 输入数据框，需要包含'Date'和'Price'列
    - n_steps: 回望窗口大小
    - inplace: 是否在原数据框上修改
    """
    df = dc(df)  # 创建数据框的深拷贝
    
    if inplace:
        df_copy = df
    else:
        df_copy = df.copy()
    
    # 确保Date是datetime类型
    df_copy['Date'] = pd.to_datetime(df_copy['Date'])
    df_copy.set_index('Date', inplace=True)
    
    # 创建时间滞后列
    for i in range(1, n_steps + 1):
        df_copy[f'Price(t-{i})'] = df_copy['Price'].shift(i)
    
    # 删除包含NaN的行
    df_copy.dropna(inplace=True)
    
    return df_copy



# 将 DataFrame 转换为 X 和 y 集合的函数
def create_dataset(df, look_back, target_column='Price', 
                               feature_columns=None, mode='train', test_size=0.05, val_size = 0.05):
    """
    创建支持多特征的LSTM数据集
    
    参数:
    - df: 原始数据框
    - look_back: 时间窗口大小
    - target_column: 要预测的目标列
    - feature_columns: 使用的特征列列表，如果为None则使用除目标列外的所有数值列
    - mode: 'train' 或 'test'
    - test_size: 测试集比例
    """
    # 确保Date是datetime并设为索引
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # 确定特征列
    if feature_columns is None:
        # 默认使用所有数值列，除了目标列
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_columns = [col for col in numeric_cols if col != target_column]
    
    # 创建时间序列数据
    X_sequences = []
    y_values = []
    
    for i in range(len(df) - look_back):
        # 特征序列：前look_back个时间步的所有特征
        feature_sequence = []
        for j in range(look_back):
            # 每个时间步的特征
            time_step_features = []
            for feature in feature_columns:
                time_step_features.append(df[feature].iloc[i + j])
            feature_sequence.append(time_step_features)
        
        X_sequences.append(feature_sequence)
        
        # 标签：下一个时间步的目标值
        y_values.append(df[target_column].iloc[i + look_back])
    
    X = np.array(X_sequences)
    y = np.array(y_values).reshape(-1, 1)
    
    # 分割数据
    split_index = int(len(X) * (1 - test_size - val_size))
    split_index_1 = int(len(X) * (1 - test_size))

    X_train = X[:split_index]
    X_val = X[split_index:split_index_1]
    X_test = X[split_index_1:]
    y_train = y[:split_index]
    y_val = y[split_index:split_index_1]
    y_test = y[split_index_1:]
    
    # 归一化
    # 重塑进行归一化
    n_features = len(feature_columns)
    X_train_reshaped = X_train.reshape(-1, n_features)
    X_val_reshaped = X_val.reshape(-1, n_features)
    X_test_reshaped = X_test.reshape(-1, n_features)
    
    feature_scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train_scaled = feature_scaler.fit_transform(X_train_reshaped)
    X_val_scaled = feature_scaler.transform(X_val_reshaped)
    X_test_scaled = feature_scaler.transform(X_test_reshaped)
    
    # 重塑回原始形状
    X_train_scaled = X_train_scaled.reshape(-1, look_back, n_features)
    X_val_scaled = X_val_scaled.reshape(-1, look_back, n_features)
    X_test_scaled = X_test_scaled.reshape(-1, look_back, n_features)
    
    target_scaler = MinMaxScaler(feature_range=(-1, 1))
    y_train_scaled = target_scaler.fit_transform(y_train)
    y_val_scaled = target_scaler.transform(y_val)
    y_test_scaled = target_scaler.transform(y_test)
    
    if mode == 'train':
        return X_train_scaled, y_train_scaled, feature_scaler, target_scaler
    elif mode == 'test':
        return X_test_scaled, y_test_scaled, feature_scaler, target_scaler   
    elif mode == 'val':
        return X_val_scaled, y_val_scaled, feature_scaler, target_scaler
    else:
        raise ValueError("mode must be 'train', 'val', or 'test'")

def create_lstm_dataset(_data_set: DataFrame, look_back, mode='train', test_size=0.05, val_size = 0.05):
    """
    创建适合LSTM的数据集
    
    参数:
    - _data_set: 原始数据框
    - look_back: 时间窗口大小
    - mode: 'train' 或 'test'
    - test_size: 测试集比例
    """
    # 准备带有时间滞后特征的DataFrame
    df = prepare_dataframe_for_lstm(_data_set, look_back, False)
    
    # 分离特征和标签
    # 特征：所有历史价格列
    # 标签：当前价格
    X = df.drop(columns=['Price']).values  # 所有历史价格特征
    y = df['Price'].values  # 当前价格标签
    
    # 归一化
    feature_scaler = MinMaxScaler(feature_range=(-1, 1))
    target_scaler = MinMaxScaler(feature_range=(-1, 1))
    
    # 拟合并转换特征
    X_scaled = feature_scaler.fit_transform(X)
    
    # 转换标签（y需要reshape为2D数组）
    y_scaled = target_scaler.fit_transform(y.reshape(-1, 1)).flatten()
    
    # 分割数据集
    split_index = int(len(X_scaled) * (1 - test_size - val_size))
    split_index_1 = int(len(X_scaled) * (1 - test_size))
    X_train = X_scaled[:split_index]
    X_val = X_scaled[split_index:split_index_1]
    X_test = X_scaled[split_index_1:]
    y_train = y_scaled[:split_index]
    y_val = y_scaled[split_index:split_index_1]
    y_test = y_scaled[split_index_1:]
    
    # 重塑为LSTM需要的形状: [samples, timesteps, features]
    # 这里timesteps = look_back, features = 1（每个时间步只有一个特征：价格）
    X_train = X_train.reshape(-1, look_back, 1)
    X_val = X_val.reshape(-1, look_back, 1)
    X_test = X_test.reshape(-1, look_back, 1)
    y_train = y_train.reshape(-1, 1)
    y_val = y_val.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    
    if mode == 'train':
        return {
            'X': X_train,
            'y': y_train,
            'feature_scaler': feature_scaler,
            'target_scaler': target_scaler
        }
    elif mode == 'test':
        return {
            'X': X_test,
            'y': y_test,
            'feature_scaler': feature_scaler,
            'target_scaler': target_scaler
        }
    elif mode == 'val':
        return {
            'X': X_val,
            'y': y_val,
            'feature_scaler': feature_scaler,
            'target_scaler': target_scaler
        }
    else:
        raise ValueError("mode must be 'train', 'val', or 'test'")

# 构造DataLoader
def get_dataloader(
    dataset, 
    batch_size=16, 
    shuffle=True,
    num_workers=0,  # 在Windows上设为0避免问题
    ):
    """
    创建DataLoader
    
    参数:
    - dataset: Dataset对象
    - batch_size: 批大小
    - shuffle: 是否打乱数据
    - num_workers: 数据加载的线程数
    """
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    return dataloader

