# datapreprocessing/utils.py
import os
import sys

# 确保路径正确，支持直接调试
if __package__ is None and __name__ == '__main__':
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    os.chdir(current_dir)

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple, List, Optional
import torch
import joblib

def create_scalers(scaler_type: str = 'standard'):
    """创建标准化器"""
    scalers = {
        'standard': StandardScaler,
        'minmax': lambda: MinMaxScaler(feature_range=(-1, 1))
    }
    return scalers.get(scaler_type, StandardScaler)()

def preprocess_data(
    data: pd.DataFrame,
    feature_columns: List[str],
    target_column: str,
    fill_method: str = 'ffill'
) -> pd.DataFrame:
    """
    预处理数据
    
    Args:
        data: 原始数据
        feature_columns: 特征列名
        target_column: 目标列名
        fill_method: 填充缺失值方法
    
    Returns:
        预处理后的DataFrame
    """
    df = data.copy()
    
    # 确保日期列是datetime类型
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    
    # 选择需要的列
    all_columns = feature_columns + [target_column]
    df = df[all_columns].copy()
    
    # 处理缺失值
    if fill_method == 'ffill':
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
    elif fill_method == 'interpolate':
        df.interpolate(method='linear', inplace=True)
        df.fillna(method='bfill', inplace=True)
    
    return df

def create_sequences(
    data: pd.DataFrame,
    sequence_length: int,
    feature_columns: List[str],
    target_column: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    创建序列数据
    
    Args:
        data: 预处理后的DataFrame
        sequence_length: 序列长度
        feature_columns: 特征列
        target_column: 目标列
    
    Returns:
        X: 特征序列 (n_samples, sequence_length, n_features)
        y: 目标值 (n_samples,)
    """
    X, y = [], []
    data_values = data.values
    
    for i in range(len(data) - sequence_length):
        X.append(data_values[i:(i + sequence_length)])
        y.append(data_values[i + sequence_length, -1])  # 最后一列是目标
    
    return np.array(X), np.array(y)

def split_data(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    分割数据为训练集、验证集和测试集
    
    Args:
        X: 特征数据
        y: 目标数据
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
    
    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须为1"
    
    n_samples = len(X)
    train_end = int(n_samples * train_ratio)
    val_end = train_end + int(n_samples * val_ratio)
    
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def scale_data(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test:np.ndarray,
    scaler_type: str = 'standard'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, object, object]:
    """
    标准化数据
    
    Args:
        X_train: 训练特征
        X_val: 验证特征
        X_test: 测试特征
        y_train: 训练标签
        y_val: 验证标签
        y_test: 测试标签
        scaler_type: 标准化类型
    
    Returns:
        标准化后的数据、X_scaler和y_scaler
    """
    # 获取原始形状
    train_shape = X_train.shape
    val_shape = X_val.shape
    test_shape = X_test.shape
    
    # 重塑为2D
    n_features = train_shape[-1]
    X_train_2d = X_train.reshape(-1, n_features)
    X_val_2d = X_val.reshape(-1, n_features)
    X_test_2d = X_test.reshape(-1, n_features)
    
    # 拟合X scaler
    scaler = create_scalers(scaler_type)
    X_train_scaled = scaler.fit_transform(X_train_2d)
    X_val_scaled = scaler.transform(X_val_2d)
    X_test_scaled = scaler.transform(X_test_2d)
    
    # 恢复原始形状
    X_train_scaled = X_train_scaled.reshape(train_shape)
    X_val_scaled = X_val_scaled.reshape(val_shape)
    X_test_scaled = X_test_scaled.reshape(test_shape)
    
    # 标准化y值
    y_scaler = create_scalers(scaler_type)
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1)).flatten()
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).flatten()
    
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train_scaled, y_val_scaled, y_test_scaled, scaler, y_scaler

def save_scaler(scaler: object, path: str):
    """保存scaler"""
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    joblib.dump(scaler, path)

def load_scaler(path: str) -> object:
    """加载scaler"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Scaler file not found: {path}")
    return joblib.load(path)