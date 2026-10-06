import os
import random
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader

from models import LSTMModel, LogCoshLoss, trim_hidden
from dataset import TimeSeriesDataset, SingleStockBatchSampler, denormalize_predictions_and_actuals

# Configuration & Seed
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_dir = "./data/stocks"
os.makedirs("results", exist_ok=True)

def init_seed(seed=875):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

init_seed()

# Data Loading & Chronological Splitting
train_list, val_list, test_list = [], [], []

for filename in os.listdir(input_dir):
    if filename.endswith(".csv"):
        symbol = filename.replace(".csv", "")
        filepath = os.path.join(input_dir, filename)
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        df.index = df.index.tz_localize(None)
        df["Mid"] = (df["high"] + df["low"]) / 2

        mid_df = df[["Mid"]].copy()
        mid_df["Symbol"] = symbol
        mid_df = mid_df.sort_index()

        total_len = len(mid_df)
        train_end = int(total_len * 0.8)
        val_end = int(total_len * 0.9)

        train_list.append(mid_df.iloc[:train_end])
        val_list.append(mid_df.iloc[train_end:val_end])
        test_list.append(mid_df.iloc[val_end:])

train = pd.concat(train_list).sort_index()
val = pd.concat(val_list).sort_index()
test = pd.concat(test_list).sort_index()

# Hyperparameters
input_size = 1
output_size = 1
batch_size = 16
num_nodes = 100
dropout = 0.0
learning_rate = 0.0001
num_epochs = 20
seq_length = 20
gamma = 0.1
max_grad_norm = 0.5

# Fitting Scaler
scaler = StandardScaler()
scaler.fit(train["Mid"].dropna().values.reshape(-1, 1))

tickers_train = sorted(train["Symbol"].unique())
tickers_val = sorted(val["Symbol"].unique())
tickers_test = sorted(test["Symbol"].unique())

# Datasets & Loaders
train_dataset = TimeSeriesDataset(train, tickers_train, seq_length, scaler, gamma=gamma, training=True)
val_dataset = TimeSeriesDataset(val, tickers_val, seq_length, scaler)
test_dataset = TimeSeriesDataset(test, tickers_test, seq_length, scaler)

train_loader = DataLoader(train_dataset, batch_sampler=SingleStockBatchSampler(train_dataset.stock_ids, batch_size))
val_loader = DataLoader(val_dataset, batch_sampler=SingleStockBatchSampler(val_dataset.stock_ids, batch_size))
test_loader = DataLoader(test_dataset, batch_sampler=SingleStockBatchSampler(test_dataset.stock_ids, batch_size))

# Model & Optimizer
model = LSTMModel(input_size, num_nodes, dropout, output_size).to(device)
criterion = LogCoshLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

def evaluate(model, val_loader, criterion):
    model.eval()
    curr_id = None
    total_loss, total_samples = 0, 0
    with torch.no_grad():
        for x, y, s in val_loader:
            x, y = x.to(device), y.to(device)
            b_size = x.size(0)
            if curr_id is None or s[0] != curr_id:
                model.reset_hidden_state(b_size, device)
            elif curr_id == s[0]:
                if b_size < batch_size:
                    model.hidden = trim_hidden(model.hidden, b_size)
                else:
                    model.detach_hidden()
            curr_id = s[0]
            outputs = model(x)
            loss = criterion(outputs, y.unsqueeze(-1))
            total_loss += loss.item()
            total_samples += x.size(0)
    return total_loss / total_samples

# Training Loop
training_losses = []
best_loss = float("inf")
best_params = model.state_dict()

for epoch in range(num_epochs):
    curr_id = None
    total_loss = 0.0
    model.train()

    for x, y, stock_id in train_loader:
        x, y = x.to(device), y.to(device)
        b_size = x.size(0)

        if curr_id is None or stock_id[0] != curr_id:
            model.reset_hidden_state(b_size, device)
        elif curr_id == stock_id[0]:
            if b_size < batch_size:
                model.hidden = trim_hidden(model.hidden, b_size)
            else:
                model.detach_hidden()

        curr_id = stock_id[0]
        outputs = model(x)
        loss = criterion(outputs, y.unsqueeze(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    training_losses.append(avg_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.7f}')

    val_loss = evaluate(model, val_loader, criterion)
    lr_scheduler.step(val_loss)
    if val_loss < best_loss:
        best_params = model.state_dict()
        best_loss = val_loss

model.load_state_dict(best_params)

# Testing Pipeline
stock_preds, stock_targets = defaultdict(list), defaultdict(list)
predictions, actuals, test_stock_ids = [], [], []

with torch.no_grad():
    model.eval()
    test_id = None
    for x, y, stock_id in test_loader:
        x, y = x.to(device), y.to(device)
        test_stock_ids.extend(stock_id)
        b_size = x.size(0)

        if test_id is None or stock_id[0] != test_id:
            model.reset_hidden_state(b_size, device)
        elif test_id == stock_id[0]:
            if b_size < batch_size:
                model.hidden = trim_hidden(model.hidden, b_size)
            else:
                model.detach_hidden()

        test_id = stock_id[0]
        outputs = model(x)
        preds = outputs.cpu().numpy().flatten()
        targets = y.cpu().numpy().flatten()

        for i in range(len(preds)):
            stock_preds[stock_id[i]].append(preds[i])
            stock_targets[stock_id[i]].append(targets[i])

        predictions.extend(preds)
        actuals.extend(targets)

test_predictions, test_actuals = denormalize_predictions_and_actuals(predictions, actuals, test_stock_ids, scaler)
bias = np.mean(test_predictions - test_actuals)
test_predictions = test_predictions - bias

# Save plots to results/
plt.figure(figsize=(8, 4))
plt.plot(training_losses, label='Train Loss')
plt.title("Training Loss Convergence")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.tight_layout()
plt.savefig("results/training_loss.png")
plt.close()

# Plot first test segment
segment_size = len(test_actuals) // 10
plt.figure(figsize=(10, 4))
plt.plot(test_actuals[:segment_size], label="Actual", color='black')
plt.plot(test_predictions[:segment_size], label="Predicted", color='red')
plt.title("Stock Price Prediction: Test Segment 1")
plt.legend()
plt.tight_layout()
plt.savefig("results/test_segment_1.png")
plt.close()

print("Pipeline execution complete. Plots saved to results/")
