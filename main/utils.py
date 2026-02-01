# main/utils.py
import os
import sys

# 确保路径正确，支持直接调试
if __package__ is None and __name__ == '__main__':
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    os.chdir(current_dir)

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple, Optional
import os
import csv
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
import yaml

def setup_device(device: str) -> torch.device:
    """设置设备"""
    if device == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

def set_seed(seed: int):
    """设置随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def calculate_metrics(
    predictions: np.ndarray,
    targets: np.ndarray
) -> Dict[str, float]:
    """
    计算评估指标
    
    Args:
        predictions: 预测值
        targets: 真实值
    
    Returns:
        指标字典
    """
    metrics = {
        'mse': mean_squared_error(targets, predictions),
        'rmse': np.sqrt(mean_squared_error(targets, predictions)),
        'mae': mean_absolute_error(targets, predictions),
        'mape': np.mean(np.abs((targets - predictions) / targets)) * 100,
        'r2': r2_score(targets, predictions)
    }
    
    return metrics

def plot_predictions(
    predictions: np.ndarray,
    targets: np.ndarray,
    save_path: str = None,
    title: str = 'Predictions vs Actual',
    dates: np.ndarray = None
):
    """绘制预测结果
    
    Args:
        predictions: 预测值
        targets: 真实值
        save_path: 保存路径
        title: 图标题
        dates: 日期数组 (格式: datetime or string)
    """
    plt.figure(figsize=(12, 6))
    plt.plot(targets, label='Actual', alpha=0.7)
    plt.plot(predictions, label='Predicted', alpha=0.7)
    
    if dates is not None and len(dates) == len(targets):
        # 使用日期作为x轴
        x_pos = np.arange(len(dates))
        plt.xticks(x_pos[::max(1, len(dates)//20)], 
                   dates[::max(1, len(dates)//20)], 
                   rotation=45, ha='right')
        plt.xlabel('Date')
    else:
        plt.xlabel('Time Step')
    
    plt.ylabel('Price')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def save_results(
    results: Dict[str, Any],
    save_dir: str,
    filename: str = 'results.json'
):
    """保存结果"""
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    
    # 转换numpy类型为Python类型
    def convert_to_serializable(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert_to_serializable(v) for v in obj]
        return obj
    
    serializable_results = convert_to_serializable(results)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)

def save_metrics_table(
    metrics: Dict[str, float],
    save_dir: str,
    filename: str = 'metrics.csv',
    model_type: str = None,
    extra: Optional[Dict[str, Any]] = None
):
    """保存指标到CSV表格
    
    Args:
        metrics: 评估指标字典
        save_dir: 保存目录
        filename: 文件名
        model_type: 模型类型
        extra: 额外信息字典
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    row = {}

    # 其他额外信息
    if extra:
        row.update(extra)
    
    # 评估指标
    row.update(metrics)

    if model_type:
        row.update({'model_type': model_type})

    write_header = not os.path.exists(save_path)
    with open(save_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

class EarlyStopping:
    """早停类"""
    
    def __init__(self, patience: int = 10, delta: float = 0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_state = None
        
    def __call__(self, val_loss: float, model: torch.nn.Module):
        if self.best_score is None:
            self.best_score = val_loss
            self.best_state = model.state_dict().copy()
        elif val_loss > self.best_score - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.best_state = model.state_dict().copy()
            self.counter = 0
            
        return self.early_stop