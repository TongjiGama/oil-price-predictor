# datapreprocessing/dataset.py
import os
import sys

# 确保路径正确，支持直接调试
if __package__ is None and __name__ == '__main__':
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    os.chdir(current_dir)

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple

class TimeSeriesDataset(Dataset):
    """时间序列数据集"""
    
    def __init__(self, X: np.ndarray, y: np.ndarray, device: str = 'cpu'):
        """
        Args:
            X: 特征序列 (n_samples, seq_len, n_features)
            y: 目标值 (n_samples,)
            device: 设备类型
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).unsqueeze(1)  # 转换为(n_samples, 1)
        self.device = device
        
        if device != 'cpu':
            self.X = self.X.to(device)
            self.y = self.y.to(device)
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]
    
    def input_size(self) -> int:
        """返回输入特征维度"""
        return self.X.shape[-1]
    
    def sequence_length(self) -> int:
        """返回序列长度"""
        return self.X.shape[1]