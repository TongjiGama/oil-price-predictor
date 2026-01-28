import sys
import os
sys.path.insert(0, './')
import torch
import numpy as np
from args import Args
import matplotlib.pyplot as plt

def evaluate_model(model, test_loader, scaler, device):
    model.eval()
    criterion = torch.nn.MSELoss()
    test_loss = 0.0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            test_loss += loss.item() * batch_x.size(0)

            # 收集预测值和真实值用于后续分析
            predictions = outputs.cpu().numpy()
            targets = batch_y.cpu().numpy()
            all_predictions.extend(predictions)
            all_targets.extend(targets)
        
    avg_test_loss = test_loss / len(test_loader)

    # 反归一化
    all_predictions = scaler.inverse_transform(np.array(all_predictions).reshape(-1, 1))
    all_targets = scaler.inverse_transform(np.array(all_targets).reshape(-1, 1))

    # 计算指标
    mae = np.mean(np.abs(all_predictions - all_targets))
    mse = np.mean((all_predictions - all_targets) ** 2)
    rmse = np.sqrt(mse)

    print(f"\n<======= 测试结果 =======>")
    print(f"测试损失: {avg_test_loss:.6f}")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    # 可视化结果
    plot_predictions(all_predictions, all_targets)
    
    return avg_test_loss, all_predictions, all_targets

def plot_predictions(train_targets, train_predictions, val_targets, val_predictions, 
                     future_predictions=None, title="预测结果对比"):
    """绘制预测值与真实值对比图"""
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    total_steps = len(train_targets) + len(val_targets)
    time_steps = np.arange(total_steps)

    plt.figure(figsize=(12, 6))

    # 绘制训练集结果
    train_end = len(train_targets)
    plt.plot(time_steps[:train_end], train_targets, 
             color='blue', label='训练集真实值', alpha=0.7, linewidth=2)
    plt.plot(time_steps[:train_end], train_predictions, 
             color='green', label='训练集计算值', alpha=0.8, linewidth=1.5)
    
    # 绘制验证集结果
    if len(val_targets) > 0:
        val_start = train_end
        val_end = train_end + len(val_targets)
        plt.plot(time_steps[val_start:val_end], val_targets, 
                 color='darkblue', label='验证集真实值', alpha=0.7, linewidth=2)
        plt.plot(time_steps[val_start:val_end], val_predictions, 
                 color='orange', label='验证集计算值', alpha=0.8, linewidth=1.5)
    
    # 绘制未来预测结果
    if future_predictions is not None:
        future_start = total_steps
        future_end = future_start + len(future_predictions)
        future_steps = np.arange(future_start, future_end)
        # 用虚线连接最后一个真实值和第一个预测值
        last_real_value = val_targets[-1] if len(val_targets) > 0 else train_targets[-1]
        plt.plot([time_steps[-1], future_steps[0]], [last_real_value, future_predictions[0]], 
                 color='red', linestyle=':', alpha=0.5, linewidth=1)
        plt.plot(future_steps, future_predictions, 
                 color='red', label='未来预测值', linestyle='--', linewidth=2, alpha=0.8)

    # 标记预测起始点
        plt.axvline(x=future_start, color='gray', linestyle=':', alpha=0.5, linewidth=1)
        plt.text(future_start, plt.ylim()[1] * 0.95, '预测开始', 
                 rotation=90, verticalalignment='top', alpha=0.7)
    
    plt.xlabel('时间步', fontsize=12)
    plt.ylabel('价格', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 添加区域标识
    plt.axvspan(0, train_end, alpha=0.1, color='green', label='训练区域')
    if len(val_targets) > 0:
        plt.axvspan(train_end, total_steps, alpha=0.1, color='yellow', label='验证区域')
    
    plt.tight_layout()
    plt.savefig('complete_prediction_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_losses(train_losses, val_losses):
    """绘制训练和验证损失曲线"""
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('训练过程损失曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('loss_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_model(model, path='best_model.pth'):
    """保存模型"""
    dropout_p = 0.0
    if hasattr(model, 'dropout'):
        d = getattr(model, 'dropout')
        if isinstance(d, float):
            dropout_p = d
        elif isinstance(d, torch.nn.Dropout):
            dropout_p = d.p
    elif hasattr(model, 'lstm') and hasattr(model.lstm, 'dropout'):
        # nn.LSTM 的 dropout 属性为 float
        dropout_p = float(model.lstm.dropout)

    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'input_dim': model.lstm.input_size,
            'hidden_dim': model.hidden_size,
            'num_layers': model.num_layers,
            'output_size': model.fc.out_features,
            'dropout': dropout_p,
        }
    }, path)
    print(f"模型已保存到: {path}")

def load_model(model_class, path='best_model.pth', device='cpu'):
    """加载模型"""
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint['config']
    config.setdefault('dropout', 0.0)  # 兼容旧模型
    model = model_class(**config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model