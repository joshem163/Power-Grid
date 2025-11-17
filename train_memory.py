import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import time
import argparse
import psutil
import os
from load_data import load_data_SubLocal
from module import MP_NX
from load_data import process_data

# ------------------------
# Device setup
# ------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ------------------------
# MLP Model
# ------------------------
class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.3):
        super(MLP, self).__init__()
        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.lins.append(nn.Linear(in_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(nn.Linear(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.lins.append(nn.Linear(hidden_channels, out_channels))
        self.dropout = dropout

    def forward(self, x):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lins[-1](x)

# ------------------------
# Training and evaluation
# ------------------------
def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb.long())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)

def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            preds = torch.argmax(out, dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(yb.cpu())

    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.cat(all_preds).numpy()

    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred, average='binary')
    rec = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')
    return acc, pre, rec, f1

# ------------------------
# Runner (80/10/10 split)
# ------------------------
def run_train_test(X, y, hidden_dim=64, epochs=100, batch_size=32, lr=0.001):
    n_samples = X.shape[0]
    indices = np.random.permutation(n_samples)

    n_train = int(0.8 * n_samples)
    n_val = int(0.1 * n_samples)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)

    model = MLP(X.shape[1], hidden_dim, 2, num_layers=3, dropout=0.3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # ------------------------
    # Memory and time profiling
    # ------------------------
    process = psutil.Process(os.getpid())
    start_cpu_mem = process.memory_info().rss / 1e6  # MB

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        start_gpu_mem = torch.cuda.memory_allocated(device) / 1e6  # MB

    print(f"\n🚀 Starting training for {epochs} epochs...")
    start_train = time.time()
    best_val_acc = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        loss = train(model, train_loader, optimizer, criterion)
        acc_val, pre_val, rec_val, f1_val = evaluate(model, val_loader)
        if acc_val > best_val_acc:
            best_val_acc = acc_val
            best_state = model.state_dict()

    end_train = time.time()
    train_time = end_train - start_train

    # Measure GPU/CPU memory
    end_cpu_mem = process.memory_info().rss / 1e6
    cpu_mem_used = end_cpu_mem - start_cpu_mem

    if device.type == 'cuda':
        peak_gpu_mem = torch.cuda.max_memory_allocated(device) / 1e6  # MB
    else:
        peak_gpu_mem = 0

    print(f"✅ Training completed in {train_time:.2f}s | CPU Δ: {cpu_mem_used:.2f} MB | GPU peak: {peak_gpu_mem:.2f} MB")

    model.load_state_dict(best_state)

    print("\n🧪 Evaluating on test set...")
    start_test = time.time()
    acc, pre, rec, f1 = evaluate(model, test_loader)
    end_test = time.time()
    test_time = end_test - start_test

    print(f"✅ Test Accuracy: {acc:.4f}")
    print(f"Precision: {pre:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
    print(f"🕒 Test time: {test_time:.2f}s | GPU peak: {peak_gpu_mem:.2f} MB")

    return train_time, test_time, acc, pre, rec, f1, cpu_mem_used, peak_gpu_mem


# ------------------------
# Main
# ------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLP classifier using extracted topological features")
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--scenarios', type=int, default=15)
    parser.add_argument('--bus', type=str, default='SF')#SF
    args, _ = parser.parse_known_args()

    # Load dataset
    A, node_fe, Label, data, Graph = process_data(args.bus, 600)
    F_voltage = np.array([0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.90, 0.85, 0.80, 0.75, 0, -1])
    F_Flow = np.array([200, 175, 150, 125, 100, 75, 50, 30, 15, 10, 0, -1])
    start_mp=time.time()
    betti0 = MP_NX(data, Graph, F_voltage, F_Flow, args.scenarios)
    end_mp = time.time()
    mp_time = end_mp - start_mp

    X = torch.tensor(betti0, dtype=torch.float)
    y = torch.tensor(Label, dtype=torch.long).view(-1)

    print(f"\n================== Running MLP Model on {args.bus} ==================")
    train_time, test_time, acc, pre, rec, f1, cpu_mem, gpu_mem = run_train_test(
        X, y,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )

    print("\n📊 Summary:")
    print(f"MP Time: {mp_time:.2f}s | Train: {train_time:.2f}s | Test: {1000 * test_time:.2f}ms")
    print(f"CPU Memory Used: {cpu_mem:.2f} MB | GPU Peak Memory: {gpu_mem:.2f} MB")
    print(f"Accuracy: {acc:.4f} | Precision: {pre:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
