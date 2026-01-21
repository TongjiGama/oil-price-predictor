import torch.nn as nn 
import torch


class LSTMPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_size):
        super(LSTMPredictor, self).__init__()
        self.hidden_size = hidden_dim
        self.num_layers = num_layers
        # 输入(B,L,input_size)
        self.predictor = nn.LSTM(input_dim, hidden_dim  , num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.predictor(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out