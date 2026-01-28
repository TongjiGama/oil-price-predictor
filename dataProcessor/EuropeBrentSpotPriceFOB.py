import sys
sys.path.insert(0, './')
import torch
import pandas as pd 
from pandas import DataFrame 
import numpy as np
from torch.utils.data import Dataset, DataLoader

from dataProcessor.utils import create_dataset, get_dataloader, create_lstm_dataset



class EuropeBrentSpotPriceFOB(Dataset):
    def __init__(self, path, mode='train', look_back=60, val_size = 0.05, test_size=0.05, 
                 use_multifeature=False, feature_columns=None):
        """
        LSTM数据集类
        
        参数:
        - path: 数据文件路径
        - mode: 'train' 或 'test'
        - look_back: 时间窗口大小
        - test_size: 测试集比例
        - use_multifeature: 是否使用多特征
        - feature_columns: 特征列列表（多特征模式下使用）
        """
        self.data:DataFrame = pd.read_csv(path)
        self.mode = mode
        self.look_back = look_back
        self.val_size = val_size
        self.test_size = test_size
        self.use_multifeature = use_multifeature
        # 验证数据
        self._validate_data()

        # 创建数据集
        if use_multifeature:
            # 多特征模式
            X, y, self.feature_scaler, self.target_scaler = create_dataset(
                self.data, 
                look_back=look_back,
                target_column='Price',
                feature_columns=feature_columns,
                mode=mode,
                test_size=test_size,
                val_size=val_size
            )
        else:
            # 单特征模式（仅使用价格）
            result = create_lstm_dataset(
                self.data, 
                look_back=look_back,
                mode=mode,
                test_size=test_size,
                val_size=val_size
            )
            X = result['X']
            y = result['y']
            self.feature_scaler = result['feature_scaler']
            self.target_scaler = result['target_scaler']

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)


    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    def _validate_data(self):
        """验证数据格式"""
        required_columns = ['Date', 'Price']
        for col in required_columns:
            if col not in self.data.columns:
                raise ValueError(f"数据必须包含 '{col}' 列")
        
        # 确保价格列是数值型
        if not np.issubdtype(self.data['Price'].dtype, np.number):
            self.data['Price'] = pd.to_numeric(self.data['Price'], errors='coerce')
            self.data.dropna(subset=['Price'], inplace=True)
    