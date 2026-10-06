import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, dropout, output_size, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_sizes, num_layers=num_layers, batch_first=True)
        self.hidden_size = hidden_sizes
        self.num_layers = num_layers
        self.layer_norm = nn.LayerNorm(hidden_sizes)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc0 = nn.Linear(hidden_sizes, hidden_sizes)
        self.fc = nn.Linear(hidden_sizes, output_size)
        self.hidden = None

    def forward(self, x):
        out = x
        x, self.hidden = self.lstm(x, self.hidden)
        x = self.dropout(x)
        x = x[:, -1, :]
        x = self.layer_norm(x)
        x = self.relu(x) + out[:, -1, :]
        x = x + self.relu(self.dropout(self.fc0(x)))
        x = self.fc(x)
        return x

    def reset_hidden_state(self, batch_size, device):
        self.hidden = (
            torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device),
            torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        )

    def detach_hidden(self):
        if self.hidden is not None:
            self.hidden = tuple(h.detach() for h in self.hidden)

def trim_hidden(hidden, new_batch_size):
    return tuple(h[:, :new_batch_size, :].detach() for h in hidden)

class LogCoshLoss(nn.Module):
    def forward(self, y_pred, y_true):
        return torch.mean(torch.log(torch.cosh(y_pred - y_true)))

class HuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta

    def forward(self, input, target):
        abs_error = torch.abs(input - target)
        quadratic = torch.minimum(abs_error, torch.tensor(self.delta))
        linear = abs_error - quadratic
        loss = 0.5 * quadratic ** 2 + self.delta * linear
        return torch.mean(loss)
