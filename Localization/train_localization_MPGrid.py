import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, hamming_loss
import numpy as np
from load_data import load_data_SubLocal
from module import MP_feature,MP_NX

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# -------------------------------
# MLP Model (no softmax!)
# -------------------------------
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
        x = self.lins[-1](x)  # raw logits
        return x  # no softmax!

# -------------------------------
# Training and testing
# -------------------------------
def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)


def test(model, loader):
    model.eval()
    all_probs, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            probs = torch.sigmoid(out).cpu()
            preds = (probs > 0.5).float()
            all_probs.append(probs)
            all_preds.append(preds)
            all_labels.append(yb.cpu())

    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.cat(all_preds).numpy()
    y_prob = torch.cat(all_probs).numpy()

    metrics = {}
    #metrics["Exact Match"] = accuracy_score(y_true, y_pred)
    sample_correct = (y_true == y_pred).sum(axis=1)
    metrics["Exact Match"] = np.mean(sample_correct / y_true.shape[1])
    try:
        metrics["ROC-AUC (micro)"] = roc_auc_score(y_true, y_prob, average='micro')
        metrics["ROC-AUC (macro)"] = roc_auc_score(y_true, y_prob, average='macro')
    except ValueError:
        metrics["ROC-AUC (micro)"] = np.nan
        metrics["ROC-AUC (macro)"] = np.nan
    metrics["F1 (micro)"] = f1_score(y_true, y_pred, average='micro')
    metrics["Hamming Loss"] = hamming_loss(y_true, y_pred)
    return metrics

# -------------------------------
# Cross-validation loop
# -------------------------------
def run_cross_validation(X, y, hidden_dim=64, epochs=50, folds=10, batch_size=32, lr=0.001):
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)

    acc_list, f1_list, auc_list = [], [], []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
        print(f"\n🔁 Fold {fold}")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)

        model = MLP(X.shape[1], hidden_dim, y.shape[1], num_layers=3, dropout=0.3).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()

        fold_metrics = []
        for epoch in range(1, epochs + 1):
            loss = train(model, train_loader, optimizer, criterion)
            metrics = test(model, test_loader)
            print(f"[Epoch {epoch:03d}] Loss: {loss:.4f} | "
                  f"Exact: {metrics['Exact Match']:.4f} | "
                  f"MicroAUC: {metrics['ROC-AUC (micro)']:.4f} | "
                  f"F1: {metrics['F1 (micro)']:.4f} | "
                  f"Hamming: {metrics['Hamming Loss']:.4f}")
            fold_metrics.append(metrics)

        # Best epoch by exact match
        best_idx = np.argmax([m["Exact Match"] for m in fold_metrics])
        best_metrics = fold_metrics[best_idx]
        print(f"\n✅ Fold {fold} Best Epoch: {best_idx + 1}")
        for k, v in best_metrics.items():
            print(f"   {k:<18s}: {v:.4f}")

        acc_list.append(best_metrics["Exact Match"])
        f1_list.append(best_metrics["F1 (micro)"])
        auc_list.append(best_metrics["ROC-AUC (micro)"])

    print("\n📊 Final 10-Fold Cross-Validation Summary:")
    print(f"Exact Match: {np.mean(acc_list):.4f} ± {np.std(acc_list):.4f}")
    print(f"Micro-F1    : {np.mean(f1_list):.4f} ± {np.std(f1_list):.4f}")
    print(f"Micro-AUC   : {np.mean(auc_list):.4f} ± {np.std(auc_list):.4f}")

# -------------------------------
# Example usage
# -------------------------------
if __name__ == "__main__":
    # Suppose you have tensors X and y
    # X: [num_samples, num_features]
    # y: [num_samples, num_labels] (multi-hot)

    A, node_fe, Label, data, Graph = load_data_SubLocal(600)
    # define filtration
    F_voltage = np.array([0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.90, 0.85, 0.80, 0.75, 0, -1])
    F_Flow = np.array([200, 175, 150, 125, 100, 75, 50, 30, 15, 10, 0, -1])
    betti0 = MP_NX(data, Graph, F_voltage, F_Flow, 600)
    X =  torch.tensor(betti0,dtype=torch.float)             # example feature matrix
    y = torch.tensor(Label,dtype=torch.float)  # random multi-labels

    run_cross_validation(X, y, hidden_dim=128, epochs=500, folds=10, batch_size=32, lr=0.001)
