# model/ARIMA.py
import os
import sys

if __package__ is None and __name__ == '__main__':
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    os.chdir(current_dir)

import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from statsmodels.tsa.arima.model import ARIMA
from model.base_model import BaseModel
from tqdm import tqdm

# 忽略警告
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

class ARIMAModel(BaseModel):
    """ARIMA模型"""
    
    def __init__(
        self,
        input_size: int,
        order: tuple = (5, 1, 0),  # (p, d, q)
        seasonal_order: tuple = (0, 0, 0, 0),  # (P, D, Q, s)
        output_size: int = 1
    ):
        super().__init__(input_size, output_size)
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.last_train_data = None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播（预测）
        注意：ARIMA是统计模型，不支持批量预测
        """
        # 将tensor转换为numpy
        x_np = x.detach().cpu().numpy()
        
        # ARIMA需要完整的时间序列进行预测
        # 这里假设输入是完整序列，预测下一个值
        if self.model is None:
            raise RuntimeError("ARIMA模型需要先调用fit方法进行训练")
        
        # 对于每个样本进行预测
        predictions = []
        for sample in x_np:
            # sample shape: (seq_len, n_features)
            # 取最后一列（目标变量）作为时间序列
            time_series = sample[:, -1] if sample.shape[1] > 1 else sample.flatten()
            
            try:
                # 使用现有模型进行预测
                forecast = self.model.forecast(steps=1)
                if hasattr(forecast, 'iloc'):
                    pred = float(forecast.iloc[0])
                else:
                    pred = float(forecast[0])
            except:
                # 如果预测失败，使用最后一个值
                pred = time_series[-1]
            
            predictions.append(pred)
        
        return torch.tensor(predictions, dtype=torch.float32).unsqueeze(1).to(x.device)
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        训练ARIMA模型
        
        ARIMA需要连续的时间序列数据。此方法假设输入X中的每个样本
        代表一个时间步上的多元特征观测，最后一列是目标变量。
        我们将所有时间步连接成一个长时间序列进行拟合。
        
        Args:
            X: 特征序列 (n_samples, seq_len, n_features) - 最后一列是目标值
            y: 目标值 (n_samples,) - 每个样本之后的目标值
        
        数据结构说明：
        - X中每个样本是一个形如(seq_len=10, n_features)的序列
        - 为了拟合ARIMA，需要构建连续时间序列：
          * 从第一个样本的10个时间步中取出目标列的值
          * 然后加入y中该样本对应的值（第11个时间步）
          * 从第二个样本的10个时间步中取出目标列的值
          * 然后加入y中该样本对应的值
          * 以此类推...
        """
        # ARIMA是单变量模型，只使用目标值（最后一列）
        train_series = []
        
        # 从每个序列的最后一列（目标值）提取时间序列
        for i in range(len(X)):
            # X[i] shape: (seq_len, n_features)
            # 取最后一列（目标值列）的所有时间步
            if X[i].shape[1] > 0:
                target_values = X[i][:, -1].tolist()  # 取该样本的seq_len个值
                train_series.extend(target_values)
                # 添加该样本对应的下一步目标值
                train_series.append(float(y[i]))
        
        # 转换为pandas Series
        train_series = pd.Series(train_series)
        
        print(f"ARIMA拟合数据长度: {len(train_series)}")
        
        # 拟合ARIMA模型
        if self.seasonal_order != (0, 0, 0, 0):
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            self.model = SARIMAX(
                train_series,
                order=self.order,
                seasonal_order=self.seasonal_order
            ).fit(disp=False)
        else:
            self.model = ARIMA(train_series, order=self.order).fit()
        
        self.last_train_data = train_series
    
    def predict(self, X: np.ndarray, y: np.ndarray = None, return_loss: bool = False, desc: str = 'Predicting') -> np.ndarray:
        """
        批量预测
        
        与fit()对应，predict()接收相同格式的数据。每个样本都是一个序列，
        我们需要基于该序列进行一步预测。
        
        Args:
            X: 输入特征序列 (n_samples, seq_len, n_features)
            y: 真实标签 (n_samples,)，可选，如果提供则计算损失
            return_loss: 是否返回损失（需要提供y）
            desc: 进度条描述
        
        Returns:
            如果return_loss=True，返回(predictions, loss)
            否则返回predictions (n_samples, 1)
        """
        predictions = []
        total_loss = 0.0
        n_samples = len(X)
        
        for i in tqdm(range(n_samples), desc=f'ARIMA {desc}', unit='sample'):
            # 获取当前序列的目标列（最后一列）
            time_series = X[i][:, -1] if X[i].shape[1] > 1 else X[i].flatten()
            
            try:
                # 对该序列单独拟合ARIMA并预测下一个值
                # 这样确保每个样本的预测都基于其对应的序列
                temp_model = ARIMA(time_series, order=self.order).fit()
                forecast = temp_model.forecast(steps=1)
                pred = float(forecast.iloc[0]) if hasattr(forecast, 'iloc') else float(forecast[0])
            except:
                # 如果拟合失败，使用序列的最后一个值作为预测
                pred = float(time_series[-1])
            
            predictions.append(pred)
            
            # 如果提供了真实标签，计算并累加损失
            if y is not None and return_loss:
                total_loss += (y[i] - pred) ** 2
        
        predictions_array = np.array(predictions).reshape(-1, 1)
        
        if return_loss and y is not None:
            mse_loss = total_loss / n_samples
            return predictions_array, mse_loss
        
        return predictions_array
    
    def forecast(self, steps: int) -> np.ndarray:
        """多步预测"""
        if self.model is None:
            raise RuntimeError("模型未训练")
        
        forecast_result = self.model.forecast(steps=steps)
        return forecast_result.values
    
    def save(self, path: str):
        """保存模型"""
        import joblib
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'order': self.order,
            'seasonal_order': self.seasonal_order,
            'last_train_data': self.last_train_data
        }, path)
    
    @classmethod
    def load(cls, path: str, device: str = 'cpu'):
        """加载模型"""
        import joblib
        checkpoint = joblib.load(path)
        model = cls(
            input_size=1,  # ARIMA是单变量模型
            order=checkpoint['order'],
            seasonal_order=checkpoint['seasonal_order']
        )
        model.model = checkpoint['model']
        model.last_train_data = checkpoint['last_train_data']
        return model