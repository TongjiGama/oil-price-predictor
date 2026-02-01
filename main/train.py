# main/train.py
import os
import sys
sys.path.insert(0,'./')

# 确保工作目录正确，支持直接运行和调试
if __package__ is None and __name__ == '__main__':
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    os.chdir(current_dir)

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any
import time
from tqdm import tqdm
import csv

from main.args import Args, get_args
from datapreprocessing.create_dataset import DatasetFactory
from model.model_factory import ModelFactory
from main.utils import setup_device, set_seed, calculate_metrics, save_results, save_metrics_table, plot_predictions, EarlyStopping

# module_dict = {
#     'LSTM': {
#         'module': 'model.LSTM',
#         'class': 'LSTMPredictor'
#     },
# }
# data_dict = {
#     'RBRTEd.csv': {
#         'module': 'datapreprocessing.create_dataset',
#         'class': 'EuropeBrentSpotPriceFOB'
#     },
# }

class Trainer:
    """训练器类"""
    
    def __init__(self, args: Args):
        self.args = args
        self.device = setup_device(args.device)
        set_seed(args.seed)
        
        # 创建目录
        os.makedirs(args.save_dir, exist_ok=True)
        os.makedirs(args.log_dir, exist_ok=True)
        os.makedirs(args.results_dir, exist_ok=True)
        
        # 准备数据
        print("准备数据...")
        dataset_factory = DatasetFactory(args)
        datasets = dataset_factory.create_datasets()

        # ARIMA模型需要原始数据
        self.datasets = datasets
        
        # 保存日期信息
        self.dates = datasets.get('dates', {})
        
        if args.model_type != 'ARIMA':
            # 深度学习模型使用DataLoader
            self.dataloaders = DatasetFactory.create_dataloaders(
                datasets,
                args.batch_size,
                args.shuffle
            )
        
        # 获取输入大小
        input_size = datasets['metadata']['input_size']
        if args.input_size is None:
            args.input_size = input_size
        
        # 创建或加载模型
        if args.model_type == 'ARIMA' and args.load_model:
            # 检查是否存在已保存的ARIMA模型
            model_path = os.path.join(args.save_dir, 'best_model.pkl')
            if os.path.exists(model_path):
                print(f"加载已保存的ARIMA模型: {model_path}")
                from model.ARIMA import ARIMAModel
                self.model = ARIMAModel.load(model_path, device=self.device)
                self.model_loaded = True
            else:
                print(f"警告：未找到已保存的模型 {model_path}，将创建新模型")
                self.model = ModelFactory.create_model(
                    args.model_type,
                    input_size=args.input_size,
                    hidden_size=args.hidden_size,
                    num_layers=args.num_layers,
                    dropout=args.dropout,
                    output_size=args.output_size,
                    # ARIMA特定参数
                    order = getattr(args, 'arima_order', (5,1,0)),
                    seasonal_order = getattr(args, 'arima_seasonal_order', (0,0,0,0))
                ).to(self.device)
                self.model_loaded = False
        else:
            print(f"创建{args.model_type}模型...")
            self.model = ModelFactory.create_model(
                args.model_type,
                input_size=args.input_size,
                hidden_size=args.hidden_size,
                num_layers=args.num_layers,
                dropout=args.dropout,
                output_size=args.output_size,
                # ARIMA特定参数
                order = getattr(args, 'arima_order', (5,1,0)),
                seasonal_order = getattr(args, 'arima_seasonal_order', (0,0,0,0))
            ).to(self.device)
            self.model_loaded = False
        
        
        if args.model_type == 'ARIMA':
            print("ARIMA模型，使用统计方法训练...")
            # ARIMA不需要传统的优化器和损失函数
            self.criterion = None
            self.optimizer = None
            self.scheduler = None
        else:
            # 损失函数和优化器
            self.criterion = nn.MSELoss()
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=args.learning_rate,
                weight_decay=args.weight_decay
            )
        
        if args.model_type != 'ARIMA':
            # 学习率调度器
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5
            )
            
            # 早停
            self.early_stopping = EarlyStopping(
                patience=args.early_stopping_patience
            )
        
        # 记录
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
    def train_epoch(self) -> float:
        """训练一个epoch"""
        if self.args.model_type == 'ARIMA':
            # ARIMA一次性训练
            print("训练ARIMA模型...")
            
            # 获取训练数据
            train_dataset = self.datasets['train']
            X_train = train_dataset.X.cpu().numpy()
            y_train = train_dataset.y.cpu().numpy().flatten()
            self.model.fit(X_train, y_train)
            
            # 训练损失：使用拟合残差（不进行预测）
            train_loss = float('nan')
            if hasattr(self.model, 'model') and self.model.model is not None and hasattr(self.model.model, 'resid'):
                resid = self.model.model.resid
                train_loss = float(np.mean(np.square(resid)))
            print(f"Train MSE: {train_loss:.6f}")
            print("ARIMA训练完成，训练阶段不进行预测。")
            
            return float(train_loss)

        if self.args.model_type != 'ARIMA':
            self.model.train()
            total_loss = 0
            
            for batch_x, batch_y in tqdm(self.dataloaders['train'], desc='Training'): # 进度条
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                # 前向传播
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪
                if self.args.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.args.gradient_clip
                    )
                
                self.optimizer.step()
                
                total_loss += loss.item()
            
            return total_loss / len(self.dataloaders['train'])
    
    def validate(self) -> float:
        """验证"""
        total_loss = 0.0
        self.model.eval()

        with torch.no_grad():
            for batch_x, batch_y in tqdm(self.dataloaders['val'], desc='Validation'): # 进度条
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item()

        return total_loss / len(self.dataloaders['val'])
    
    def train(self):
        """训练循环"""
        print("开始训练...")
        
        # ARIMA仅需要1个epoch，不需要循环
        if self.args.model_type == 'ARIMA':
            print("\n=== ARIMA模型训练/验证 ===")
            print("-" * 50)
            
            # 如果模型已加载，跳过训练
            if self.model_loaded:
                print("使用已加载的模型，跳过训练阶段")
                train_loss = float('nan')  # 已加载模型不计算训练损失
                val_loss = self.validate()
            else:
                print("训练新模型 (单次拟合，epochs=1)")
                # 训练
                train_loss = self.train_epoch()
            
            # 记录
            self.history['train_loss'].append(train_loss)
            
            # 打印信息
            if not self.model_loaded:
                print(f"Train Loss: {train_loss:.6f}")
            
            # # 计算训练集评估指标
            # print("\n计算训练集评估指标...")
            # train_dataset = self.datasets['train']
            # X_train = train_dataset.X.cpu().numpy()
            # y_train = train_dataset.y.cpu().numpy().flatten()
            # train_predictions, _ = self.model.predict(X_train, y_train, return_loss=True, desc='Train Metrics')
            # train_metrics = calculate_metrics(train_predictions, y_train)
            # print("训练集指标:")
            # for key, value in train_metrics.items():
            #     print(f"  {key.upper()}: {value:.6f}")
            
            # 计算验证集评估指标
            print("\n计算验证集评估指标...")
            test_dataset = self.datasets['test']
            X_test = test_dataset.X.cpu().numpy()
            y_test = test_dataset.y.cpu().numpy().flatten()
            test_predictions, _ = self.model.predict(X_test, y_test, return_loss=True, desc='Test Metrics')
            test_metrics = calculate_metrics(test_predictions, y_test)
            print("测试集指标:")
            for key, value in test_metrics.items():
                print(f"  {key.upper()}: {value:.6f}")
            
            # 保存指标到表格
            # save_metrics_table(
            #     metrics=train_metrics,
            #     save_dir=self.args.results_dir,
            #     filename='train_metrics.csv',
            #     model_type=self.args.model_type,
            #     extra={'dataset': 'train', 'order': str(self.args.order)}
            # )
            save_metrics_table(
                metrics=test_metrics,
                save_dir=self.args.results_dir,
                filename='test_metrics.csv',
                model_type=self.args.model_type,
                extra={'split': 'test', 'order': str(self.args.arima_order)}
            )
            print(f"\n指标已保存到 {self.args.results_dir}")
            
            # 绘制预测结果
            print("\n绘制预测结果...")
            # # 训练集预测图
            # train_dates = self.dates.get('train', None)
            # plot_predictions(
            #     predictions=train_predictions,
            #     targets=y_train,
            #     save_path=os.path.join(self.args.results_dir, 'train_predictions.png'),
            #     title=f'ARIMA {self.args.order} - Train Predictions vs Actual',
            #     dates=train_dates
            # )
            # 测试集预测图
            test_dates = self.dates.get('test', None)
            plot_predictions(
                predictions=test_predictions,
                targets=y_test,
                save_path=os.path.join(self.args.results_dir, 'test_predictions_arima.png'),
                title=f'ARIMA {self.args.arima_order} - Test Predictions vs Actual',
                dates=test_dates
            )
            print(f"预测结果图已保存到 {self.args.results_dir}")
            
            # 保存模型（仅在新训练时保存）
            if not self.model_loaded:
                save_path = os.path.join(self.args.save_dir, 'best_model.pkl')
                self.model.save(save_path)
                print(f"\n保存ARIMA模型到: {save_path}")
                
            else:
                print(f"\n使用已加载的模型，无需保存")
            
            print("\n完成！")
            self.save_history()
            return
        
        # LSTM等深度学习模型的正常训练循环
        best_val_loss = float('inf')
        
        for epoch in range(self.args.epochs):
            print(f"\nEpoch {epoch + 1}/{self.args.epochs}")
            print("-" * 50)
            
            # 训练
            train_loss = self.train_epoch()
            
            # 验证
            val_loss = self.validate()
            
            # 更新学习率
            self.scheduler.step(val_loss)
            
            # 记录
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rate'].append(
                self.optimizer.param_groups[0]['lr']
            )
            
            # 打印信息
            print(f"Train Loss: {train_loss:.6f}")
            print(f"Val Loss: {val_loss:.6f}")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                
                # 深度学习模型使用checkpoint
                self.save_checkpoint(
                    os.path.join(self.args.save_dir, 'best_model.pth'),
                    epoch + 1,
                    val_loss
                )
                print(f"保存最佳模型，Val Loss: {val_loss:.6f}")
            
            # 定期保存
            if (epoch + 1) % self.args.checkpoint_interval == 0:
                self.save_checkpoint(
                    os.path.join(self.args.save_dir, f'checkpoint_epoch_{epoch+1}.pth'),
                    epoch + 1,
                    val_loss
                )

            # 保存本轮模型参数统计到表格
            self.save_model_parameters_table(epoch + 1)

            # 早停检查
            if self.early_stopping(val_loss, self.model):
                print("早停触发")
                # 恢复最佳模型
                self.model.load_state_dict(self.early_stopping.best_state)
                break
        
        print("训练完成！")
        
        # 保存历史记录
        self.save_history()

    def save_model_parameters_table(self, epoch: int):
        """保存当前模型参数统计到单个CSV表格"""
        os.makedirs(self.args.results_dir, exist_ok=True)
        save_path = os.path.join(self.args.results_dir, 'model_parameters_stats.csv')

        write_header = not os.path.exists(save_path)
        with open(save_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(['epoch', 'param_name', 'mean', 'variance'])

            for name, param in self.model.named_parameters():
                data = param.detach().cpu().numpy().ravel()
                mean_value = float(data.mean())
                variance_value = float(data.var())
                writer.writerow([epoch, name, mean_value, variance_value])
    
    def save_checkpoint(self, path: str, epoch: int, val_loss: float):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'args': vars(self.args),
            'history': self.history
        }
        torch.save(checkpoint, path)
    
    def save_history(self):
        """保存训练历史"""
        save_results(
            self.history,
            self.args.results_dir,
            'training_history.json'
        )
        
        # 绘制损失曲线
        if self.args.model_type != 'ARIMA':
            self.plot_loss_curve()
    
    def plot_loss_curve(self):
        """绘制损失曲线"""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        save_path = os.path.join(self.args.results_dir, 'loss_curve.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """主函数"""
    args = get_args()
    trainer = Trainer(args)
    trainer.train()

if __name__ == '__main__':
    main()