# model/model_factory.py
import os
import sys

# 确保路径正确，支持直接调试
if __package__ is None and __name__ == '__main__':
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    os.chdir(current_dir)

from typing import Dict, Type
from model.LSTM import LSTMModel
from model.ARIMA import ARIMAModel
from model.base_model import BaseModel

class ModelFactory:
    """模型工厂类"""
    
    _models: Dict[str, Type[BaseModel]] = {
        'LSTM': LSTMModel,
        'ARIMA': ARIMAModel,
        # 可以在这里添加其他模型
        # 'GRU': GRUModel,
        # 'Transformer': TransformerModel,
    }
    
    @classmethod
    def create_model(cls, model_type: str, **kwargs) -> BaseModel:
        """
        创建模型
        
        Args:
            model_type: 模型类型
            **kwargs: 模型参数
        
        Returns:
            模型实例
        """
        if model_type not in cls._models:
            raise ValueError(f"未知的模型类型: {model_type}")
        
        # 特殊处理ARIMA模型的参数
        if model_type == 'ARIMA':
            # ARIMA使用不同的参数集
            arima_kwargs = {
                'input_size': kwargs.get('input_size', 1),
                'order': kwargs.get('order', (5, 1, 0)),
                'seasonal_order': kwargs.get('seasonal_order', (0, 0, 0, 0)),
                'output_size': kwargs.get('output_size', 1)
            }
            return cls._models[model_type](**arima_kwargs)
        
        # LSTM和其他深度学习模型的参数
        if model_type == 'LSTM':
            lstm_kwargs = {
                'input_size': kwargs.get('input_size'),
                'hidden_size': kwargs.get('hidden_size'),
                'num_layers': kwargs.get('num_layers'),
                'dropout': kwargs.get('dropout'),
                'output_size': kwargs.get('output_size')
            }
            return cls._models[model_type](**lstm_kwargs)

        model_class = cls._models[model_type]
        return model_class(**kwargs)
    
    @classmethod
    def register_model(cls, name: str, model_class: Type[BaseModel]):
        """注册新模型"""
        cls._models[name] = model_class