# datapreprocessing/create_dataset.py
import os
import sys

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from datapreprocessing.utils import preprocess_data, create_sequences, split_data, scale_data, save_scaler
import torch
from torch.utils.data import DataLoader
from datapreprocessing.dataset import TimeSeriesDataset

class DatasetFactory:
    """数据集工厂类"""
    
    def __init__(self, args):
        self.args = args
        self.scalers = {} # 存储标准化器的字典
        
    def create_datasets(self) -> Dict[str, Any]:
        """
        创建完整的数据集
        
        Returns:
            包含数据集和元数据的字典
        """
        # 加载数据，尝试多个编码，latin-1 可以读取任何字节序列
        try:
            # 尝试用utf-8编码读取CSV文件
            raw_data = pd.read_csv(self.args.data_path, encoding='utf-8')
        except UnicodeDecodeError:
            # 如果utf-8失败，尝试gbk编码（中文编码）
            try:
                raw_data = pd.read_csv(self.args.data_path, encoding='gbk')
            except (UnicodeDecodeError, LookupError):
                # 如果以上都失败，使用latin-1编码（兼容性最强）
                raw_data = pd.read_csv(self.args.data_path, encoding='latin-1')
        
        # 预处理
        processed_data = preprocess_data(
            raw_data,
            self.args.feature_columns,
            self.args.target_column
        )
        
        # 保存日期信息，如果索引有strftime方法（说明是datetime类型），则格式化为字符串，否则直接转为字符串
        dates = processed_data.index.strftime('%Y-%m-%d') if hasattr(processed_data.index, 'strftime') else processed_data.index.astype(str)
        
        # 创建序列
        X, y = create_sequences(
            processed_data,
            self.args.sequence_length,
            self.args.feature_columns,
            self.args.target_column
        )
        
        # 获取对应的日期序列
        dates_for_sequences = dates[self.args.sequence_length:]
        
        # 分割数据
        X_train, y_train, X_val, y_val, X_test, y_test = split_data(
            X, y,
            self.args.train_ratio,
            self.args.val_ratio,
            self.args.test_ratio
        )
        
        # 分割日期
        n_train = len(X_train)
        n_val = len(X_val)
        dates_train = dates_for_sequences[:n_train]
        dates_val = dates_for_sequences[n_train:n_train+n_val]
        dates_test = dates_for_sequences[n_train+n_val:]
        
        # 标准化
        X_train_scaled, X_val_scaled, X_test_scaled, y_train_scaled, y_val_scaled, y_test_scaled, scaler, y_scaler = scale_data(
            X_train, X_val, X_test,
            y_train, y_val, y_test,
            scaler_type='standard' # 使用标准归一化
        )
        
        # 保存scaler
        if not os.path.exists('scalers'):
            os.makedirs('scalers')
        save_scaler(scaler, 'scalers/feature_scaler.pkl')
        
        # 创建PyTorch数据集
        train_dataset = TimeSeriesDataset(X_train_scaled, y_train_scaled)
        val_dataset = TimeSeriesDataset(X_val_scaled, y_val_scaled)
        test_dataset = TimeSeriesDataset(X_test_scaled, y_test_scaled)
        
        return {
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset,
            'scaler': scaler,
            'y_scaler': y_scaler,
            'dates': {
                'train': np.array(dates_train),
                'val': np.array(dates_val),
                'test': np.array(dates_test)
            },
            'metadata': {
                'input_size': train_dataset.input_size(),
                'sequence_length': train_dataset.sequence_length(),
                'n_train': len(train_dataset),
                'n_val': len(val_dataset),
                'n_test': len(test_dataset)
            }
        }
    
    @staticmethod
    def create_dataloaders(
        datasets: Dict[str, TimeSeriesDataset],
        batch_size: int,
        shuffle_train: bool = True
    ) -> Dict[str, torch.utils.data.DataLoader]:
        """
        创建DataLoader
        
        Args:
            datasets: 数据集字典
            batch_size: 批大小
            shuffle_train: 是否打乱训练集
        
        Returns:
            DataLoader字典
        """
        from torch.utils.data import DataLoader
        
        dataloaders = {}
        
        dataloaders['train'] = DataLoader(
            datasets['train'],
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=0,
            pin_memory=True
        )
        
        dataloaders['val'] = DataLoader(
            datasets['val'],
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        dataloaders['test'] = DataLoader(
            datasets['test'],
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        return dataloaders