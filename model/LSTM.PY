import torch.nn as nn 
import torch
import numpy as np
from copy import deepcopy as dc

class LSTMPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_size, dropout):
        super(LSTMPredictor, self).__init__()
        self.hidden_size = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        # 输入(B,L,input_size)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        # 全连接层
        self.fc = nn.Linear(hidden_dim, output_size)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x):
        # 初始化隐藏状态和细胞状态
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        # 前向传播 LSTM
        out, _ = self.lstm(x, (h0, c0))
        # 只取最后一个时间步的输出
        out = self.fc(out[:, -1, :])
        return out

def train_model(model, train_loader, val_loader, num_epochs, learning_rate):
    device = model.device
    # 训练模型
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5) # 学习率调度器

    train_losses = []
    val_losses = []
    outputs = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_x)
            outputs.append(output.detach().cpu().numpy())
            loss = criterion(output, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪，防止梯度爆炸
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0) # 累加损失

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # 验证模型
        model.eval()
        val_loss = 0.0
        val_outputs = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                val_output = model(batch_x)
                val_outputs.append(val_output.detach().cpu().numpy())
                loss = criterion(val_output, batch_y)
                val_loss += loss.item() * batch_x.size(0)
            
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)  # 根据验证损失调整学习率

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
    
    return train_losses, val_losses, outputs, val_outputs

def predict(model, data, scaler, sequence_length):
    model.eval()
    # 将输入序列统一成 numpy
    if torch.is_tensor(data):
        last_sequence = data[-sequence_length:].detach().cpu().numpy().reshape(-1)
    else:
        last_sequence = np.array(data[-sequence_length:]).reshape(-1)

    preds = []
    with torch.no_grad():
        for _ in range(data.shape[0]):  # 预测与输入序列长度相同步数
            inp = torch.tensor(last_sequence, dtype=torch.float32, device=model.device).view(1, -1, 1)
            pred = model(inp)
            pred_val = pred.cpu().numpy().flatten()[0]
            preds.append(pred_val)
            # 滑动窗口更新
            last_sequence = np.append(last_sequence[1:], pred_val)

    preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    return preds
