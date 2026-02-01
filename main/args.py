# main/args.py
import os
import sys

# 确保路径正确，支持直接调试
if __package__ is None and __name__ == '__main__':
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    os.chdir(current_dir)

import argparse
from dataclasses import dataclass, field
from typing import Optional, List
import yaml

@dataclass
class Args:
    # 训练参数
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    
    # 数据参数
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    sequence_length: int = 10
    shuffle: bool = True
    
    # 模型参数
    model_type: str = 'ARIMA'  # 'LSTM' or 'ARIMA'
    input_size: Optional[int] = None  # 自动推断
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    output_size: int = 1
    
    # 系统参数
    seed: int = 42
    num_workers: int = 0
    device: str = 'cpu'  # 'cuda' or 'cpu'
    
    # 路径参数
    data_path: str = 'data/RBRTEd.csv'
    save_dir: str = 'checkpoints/'
    log_dir: str = 'logs/'
    results_dir: str = 'results/'
    
    # 训练控制
    early_stopping_patience: int = 10
    save_best_only: bool = True
    checkpoint_interval: int = 5
    
    # 特征配置
    feature_columns: List[str] = field(default_factory=list)
    target_column: str = 'Price'

    # ARIMA特定参数
    arima_order: tuple = (5, 1, 0)  # (p, d, q)
    arima_seasonal_order: tuple = (0, 0, 0, 0)  # (P, D, Q, s)
    
    # 模型加载
    load_model: bool = True  # 是否加载已保存的模型

def get_args():
    parser = argparse.ArgumentParser(description='Oil Price Prediction')
    
    # 数据相关
    parser.add_argument('--data-path', type=str, default='data/RBRTEd.csv')
    parser.add_argument('--train-ratio', type=float, default=0.7)
    parser.add_argument('--val-ratio', type=float, default=0.15)
    parser.add_argument('--test-ratio', type=float, default=0.15)
    parser.add_argument('--sequence-length', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--feature-columns', type=str, nargs='+', 
                       default=[]) # nargs='+'表示接受多个值
    parser.add_argument('--target-column', type=str, default='Price')
    
    # LSTM相关参数
    parser.add_argument('--model-type', type=str, default='ARIMA')
    parser.add_argument('--hidden-size', type=int, default=64)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--input-size', type=int, default=None)
    parser.add_argument('--output-size', type=int, default=1)
    # ARIMA相关参数
    parser.add_argument('--arima-p', type=int, default=5, help='ARIMA p参数')
    parser.add_argument('--arima-d', type=int, default=1, help='ARIMA d参数')
    parser.add_argument('--arima-q', type=int, default=0, help='ARIMA q参数')
    parser.add_argument('--arima-s', type=int, default=0, help='季节性周期')
    
    # 训练相关
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--gradient-clip', type=float, default=1.0)
    parser.add_argument('--early-stopping-patience', type=int, default=10)
    
    # 系统相关
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--save-dir', type=str, default='checkpoints/')
    parser.add_argument('--log-dir', type=str, default='logs/')
    parser.add_argument('--results-dir', type=str, default='results/')
    parser.add_argument('--save-best-only', action='store_true')
    parser.add_argument('--checkpoint-interval', type=int, default=5)
    parser.add_argument('--load-model', action='store_true', help='加载已保存的模型而不是训练')
    parser.add_argument('--config', type=str, help='配置文件路径')
    
    args = parser.parse_args()

    # 处理ARIMA参数
    args.arima_order = (args.arima_p, args.arima_d, args.arima_q)
    if args.arima_s > 0:
        args.arima_seasonal_order = (0, 0, 0, args.arima_s)
    
    # 如果提供了配置文件，则加载
    if args.config:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        # 更新args，处理特殊字段
        for key, value in config.items():
            # 处理 feature_columns 可能是列表或字符串
            if key == 'feature_columns' and value is not None:
                if isinstance(value, str):
                    value = [value]
                elif not isinstance(value, list):
                    value = list(value) if value else []
            # 尝试转换为对应类型，如果失败则跳过
            try:
                setattr(args, key, value)
            except (TypeError, ValueError, AttributeError):
                # 如果属性不存在或类型不匹配，忽略
                pass
    
    # 移除 config 参数，它不属于 Args
    args_dict = vars(args)
    args_dict.pop('config', None)
    # 移除中间ARIMA参数（已合并到 arima_order / arima_seasonal_order）
    args_dict.pop('arima_p', None)
    args_dict.pop('arima_d', None)
    args_dict.pop('arima_q', None)
    args_dict.pop('arima_s', None)
    
    # ARIMA特殊处理：ARIMA只需1个epoch
    if args_dict.get('model_type') == 'LSTM':
        args_dict['epochs'] = 1
    
    # 转换 args 为 Args 数据类，并自动进行类型转换
    return Args(**args_dict)

if __name__ == '__main__':
    args = get_args()
    print(args)