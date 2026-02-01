# model/base_model.py
import os
import sys

# 确保路径正确，支持直接调试
if __package__ is None and __name__ == '__main__':
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    os.chdir(current_dir)

import torch.nn as nn
import torch
from abc import ABC, abstractmethod

class BaseModel(nn.Module, ABC):
    """模型基类"""
    
    def __init__(self, input_size: int, output_size: int = 1):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        
    @abstractmethod
    def forward(self, x):
        pass
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_config': self.__dict__
        }, path)
    
    @classmethod
    def load(cls, path: str, device: str = 'cpu'):
        """加载模型"""
        checkpoint = torch.load(path, map_location=device)
        model = cls(**checkpoint['model_config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        return model