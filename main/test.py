# main/test.py
import os
import sys
sys.path.insert(0,'./')


import torch
import numpy as np
from typing import Dict, Any

from main.args import Args, get_args
from datapreprocessing.create_dataset import DatasetFactory
from model.model_factory import ModelFactory
from main.utils import (
    setup_device, calculate_metrics,
    plot_predictions, save_results, save_metrics_table
)

class Tester:
    """测试器类"""
    
    def __init__(self, args: Args, model_path: str = None):
        self.args = args
        self.device = setup_device(args.device)
        
        # 加载数据
        dataset_factory = DatasetFactory(args)
        datasets = dataset_factory.create_datasets()
        
        # ARIMA模型不使用DataLoader
        if args.model_type == 'ARIMA':
            self.datasets = datasets
        else:
            self.dataloaders = DatasetFactory.create_dataloaders(
                datasets,
                batch_size=1,  # 测试时使用batch_size=1
                shuffle_train=False
            )
        
        # 保存日期信息
        self.dates = datasets.get('dates', {})
        
        # 保存scaler用于反归一化
        self.y_scaler = datasets.get('y_scaler', None)

        # 获取输入大小
        input_size = datasets['metadata']['input_size']
        if self.args.input_size is None:
            self.args.input_size = input_size
        
        # 加载模型
        self.load_model(model_path)
        
    def load_model(self, model_path: str = None):
        """加载模型"""
        # 自动检测模型类型和路径
        if model_path is None:
            if self.args.model_type == 'ARIMA':
                model_path = os.path.join(self.args.save_dir, 'best_model.pkl')
            else:
                model_path = os.path.join(self.args.save_dir, 'best_model.pth')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # ARIMA模型用.pkl格式
        if self.args.model_type == 'ARIMA':
            from model.ARIMA import ARIMAModel
            self.model = ARIMAModel.load(model_path)
            print(f"已加载ARIMA模型: {model_path}")
            return

        
        # LSTM等深度学习模型用.pth格式
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 创建模型
        self.model = ModelFactory.create_model(
            self.args.model_type,
            input_size=self.args.input_size,
            hidden_size=self.args.hidden_size,
            num_layers=self.args.num_layers,
            dropout=self.args.dropout,
            output_size=self.args.output_size
        ).to(self.device)
        
        # 加载权重
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"已加载模型: {model_path}")
    
    def test(self) -> Dict[str, Any]:
        """测试模型"""
        print("开始测试...")
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in self.dataloaders['test']:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                # 预测
                predictions = self.model(batch_x)
                
                # 收集结果
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(batch_y.cpu().numpy())
        
        # 合并结果
        predictions = np.concatenate(all_predictions, axis=0).flatten()
        targets = np.concatenate(all_targets, axis=0).flatten()
        
        # 反归一化
        if self.y_scaler is not None:
            predictions = self.y_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
            targets = self.y_scaler.inverse_transform(targets.reshape(-1, 1)).flatten()
            print("已对预测值和目标值进行反归一化")
        
        # 计算指标
        metrics = calculate_metrics(predictions, targets)
        
        # 打印结果
        print("\n测试结果 (LSTM):")
        print("-" * 50)
        for metric_name, value in metrics.items():
            print(f"{metric_name.upper()}: {value:.6f}")
        
        # 绘制结果
        plot_predictions(
            predictions,
            targets,
            save_path=os.path.join(self.args.results_dir, 'test_predictions_lstm.png'),
            title='LSTM Test Predictions vs Actual',
            dates=self.dates.get('test', None)
        )
        
        # 保存结果
        results = {
            'predictions': predictions,
            'targets': targets,
            'metrics': metrics,
            'dates': self.dates.get('test', None).tolist() if 'test' in self.dates else None,
            'args': vars(self.args),
            'model_type': 'LSTM'
        }
        
        save_results(
            results,
            self.args.results_dir,
            'test_results_lstm.json'
        )

        save_metrics_table(
            metrics,
            self.args.results_dir,
            filename='test_metrics.csv',
            model_type='LSTM',
            extra={'split': 'test'}
        )
        
        return results

def main():
    """主函数"""
    args = get_args()
    
    # 如果需要，可以从命令行指定模型路径
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default=None)
    test_args = parser.parse_args()
    
    tester = Tester(args, test_args.model_path)
    tester.test()

if __name__ == '__main__':
    main()