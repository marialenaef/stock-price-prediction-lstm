import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from collections import defaultdict
import random

class TimeSeriesDataset(Dataset):
    def __init__(self, df, tickers, seq_length, scaler, gamma=0.1, training=False):
        self.seq_length = seq_length
        self.data = []
        self.labels = []
        self.stock_ids = []

        for ticker in tickers:
            stock_data = df[df["Symbol"] == ticker]["Mid"].values
            stock_data = stock_data[~np.isnan(stock_data)]
            stock_data = scaler.transform(stock_data.reshape(-1, 1)).reshape(-1)

            if training:
                ema = np.zeros_like(stock_data)
                ema[0] = stock_data[0]
                for i in range(1, len(stock_data)):
                    ema[i] = gamma * stock_data[i] + (1 - gamma) * ema[i - 1]
                stock_data = ema

            for i in range(len(stock_data) - seq_length):
                self.data.append(stock_data[i:i + seq_length])
                self.labels.append(stock_data[i + seq_length])
                self.stock_ids.append(ticker)

        combined = list(zip(self.stock_ids, self.data, self.labels))
        combined.sort(key=lambda x: x[0])
        self.stock_ids, self.data, self.labels = zip(*combined)

        self.data = torch.tensor(np.array(self.data), dtype=torch.float32)
        self.labels = torch.tensor(np.array(self.labels), dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx].reshape(-1, 1), self.labels[idx], self.stock_ids[idx]

class SingleStockBatchSampler(Sampler):
    def __init__(self, stock_ids, batch_size, shuffle=False):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.stock_to_indices = defaultdict(list)
        for idx, stock in enumerate(stock_ids):
            self.stock_to_indices[stock].append(idx)

        self.stocks = list(self.stock_to_indices.keys())
        if self.shuffle:
            random.shuffle(self.stocks)

    def __iter__(self):
        for stock in self.stocks:
            indices = self.stock_to_indices[stock]
            if self.shuffle:
                random.shuffle(indices)
            for i in range(0, len(indices), self.batch_size):
                yield indices[i:i + self.batch_size]

    def __len__(self):
        return sum((len(indices) + self.batch_size - 1) // self.batch_size
                   for indices in self.stock_to_indices.values())

def denormalize_predictions_and_actuals(predictions, actuals, stock_ids, scaler):
    denorm_predictions = []
    denorm_actuals = []
    for pred, actual, _ in zip(predictions, actuals, stock_ids):
        pred_denorm = scaler.inverse_transform([[pred]])[0][0]
        actual_denorm = scaler.inverse_transform([[actual]])[0][0]
        denorm_predictions.append(pred_denorm)
        denorm_actuals.append(actual_denorm)
    return np.array(denorm_predictions), np.array(denorm_actuals)
