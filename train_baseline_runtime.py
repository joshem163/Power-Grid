import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv, global_mean_pool
from torch_geometric.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import time
import psutil
import os
from tqdm import tqdm
from models import GNN, GPSModel, GraphTransformer, Graphormer
from load_data import process_data, load_graph_data, print_stat

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ------------------------ TRAIN ------------------------
def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
        correct += (out.argmax(dim=1) == data.y).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)

# ------------------------ TEST ------------------------
def test(model, loader):
    model.eval()
    correct = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)
            all_preds.append(pred.cpu())
            all_labels.append(data.y.cpu())
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    acc = (all_preds == all_labels).sum().item() / len(all_labels)
    return acc, all_preds.numpy(), all_labels.numpy()

def run_train_val_test(data_list, model_name, hidden_dim, epochs, batch_size, lr):
    # --- Split 80/10/10 ---
    np.random.seed(42)
    idx = np.random.permutation(len(data_list))
    n_total = len(idx)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train+n_val]
    test_idx = idx[n_train+n_val:]

    train_data = [data_list[i] for i in train_idx]
    val_data = [data_list[i] for i in val_idx]
    test_data = [data_list[i] for i in test_idx]

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    # --- Model Selection ---
    in_dim = data_list[0].x.size(1)
    if model_name in ['GCN', 'SAGE', 'GAT', 'GIN']:
        model = GNN(model_name, in_channels=in_dim, hidden_channels=hidden_dim, num_classes=2).to(device)
    elif model_name == 'UniMP':
        model = GraphTransformer(in_channels=in_dim, hidden_channels=hidden_dim, num_classes=2).to(device)
    elif model_name == 'graphormer':
        model = Graphormer(in_channels=in_dim, hidden_channels=hidden_dim, num_classes=2).to(device)
    elif model_name == 'GPS':
        model = GPSModel(in_channels=in_dim, hidden_channels=hidden_dim, num_classes=2).to(device)
    else:
        raise ValueError(f"Model '{model_name}' not defined")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc = 0
    best_state = None

    # --- Memory profiling setup ---
    process = psutil.Process(os.getpid())
    start_cpu_mem = process.memory_info().rss / 1e6  # MB
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        start_gpu_mem = torch.cuda.memory_allocated(device) / 1e6

    # --- Training ---
    print(f"\n🚀 Starting training for {epochs} epochs...")
    train_start = time.time()

    for epoch in tqdm(range(1, epochs + 1)):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion)
        val_acc, _, _ = test(model, val_loader)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()

    train_end = time.time()
    train_time = train_end - train_start

    # --- Measure memory usage ---
    end_cpu_mem = process.memory_info().rss / 1e6
    cpu_mem_used = end_cpu_mem - start_cpu_mem
    gpu_mem_peak = torch.cuda.max_memory_allocated(device) / 1e6 if device.type == 'cuda' else 0

    print(f"✅ Training completed in {train_time:.2f}s | CPU Δ: {cpu_mem_used:.2f} MB | GPU peak: {gpu_mem_peak:.2f} MB")

    # --- Testing ---
    print("\n🧪 Evaluating on test set...")
    model.load_state_dict(best_state)
    test_start = time.time()
    test_acc, y_pred, y_true = test(model, test_loader)
    test_end = time.time()
    test_time = test_end - test_start

    print("🧮 Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')

    print(f"✅ Test Accuracy: {test_acc:.4f}")
    print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
    print(f"🕒 Test Time: {test_time:.2f}s | GPU peak: {gpu_mem_peak:.2f} MB")

    return {
        "train_time": train_time,
        "test_time": test_time,
        "test_acc": test_acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "cpu_mem": cpu_mem_used,
        "gpu_mem": gpu_mem_peak
    }


# ------------------------ RUNNER ------------------------
# def run_train_val_test(data_list, model_name, hidden_dim, epochs, batch_size, lr):
#     # --- Split 80/10/10 ---
#     np.random.seed(42)
#     idx = np.random.permutation(len(data_list))
#     n_total = len(idx)
#     n_train = int(0.8 * n_total)
#     n_val = int(0.1 * n_total)
#     train_idx = idx[:n_train]
#     val_idx = idx[n_train:n_train+n_val]
#     test_idx = idx[n_train+n_val:]
#
#     train_data = [data_list[i] for i in train_idx]
#     val_data = [data_list[i] for i in val_idx]
#     test_data = [data_list[i] for i in test_idx]
#
#     train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(val_data, batch_size=batch_size)
#     test_loader = DataLoader(test_data, batch_size=batch_size)
#
#     # --- Model Selection ---
#     in_dim = data_list[0].x.size(1)
#     if model_name in ['GCN', 'SAGE', 'GAT', 'GIN']:
#         model = GNN(model_name, in_channels=in_dim, hidden_channels=hidden_dim, num_classes=2).to(device)
#     elif model_name == 'UniMP':
#         model = GraphTransformer(in_channels=in_dim, hidden_channels=hidden_dim, num_classes=2).to(device)
#     elif model_name == 'graphormer':
#         model = Graphormer(in_channels=in_dim, hidden_channels=hidden_dim, num_classes=2).to(device)
#     elif model_name == 'GPS':
#         model = GPSModel(in_channels=in_dim, hidden_channels=hidden_dim, num_classes=2).to(device)
#     else:
#         raise ValueError(f"Model '{model_name}' not defined")
#
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     criterion = torch.nn.CrossEntropyLoss()
#
#     best_val_acc = 0
#     best_state = None
#
#     # --- Training ---
#     print(f"\n🚀 Starting training for {epochs} epochs...")
#     train_start = time.time()
#
#     for epoch in tqdm(range(1, epochs + 1)):
#         train_loss, train_acc = train(model, train_loader, optimizer, criterion)
#         val_acc, _, _ = test(model, val_loader)
#
#         if val_acc > best_val_acc:
#             best_val_acc = val_acc
#             best_state = model.state_dict()
#
#         #print(f"[Epoch {epoch:03d}] Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
#
#     train_end = time.time()
#     print(f"✅ Training completed in {(train_end - train_start):.2f} seconds.")
#
#     # --- Testing ---
#     print("\n🧪 Evaluating on test set...")
#     model.load_state_dict(best_state)
#     test_start = time.time()
#     test_acc, y_pred, y_true = test(model, test_loader)
#     test_end = time.time()
#
#     print("🧮 Confusion Matrix:")
#     print(confusion_matrix(y_true, y_pred))
#     precision = precision_score(y_true, y_pred, average='binary')
#     recall = recall_score(y_true, y_pred, average='binary')
#     f1 = f1_score(y_true, y_pred, average='binary')
#
#     print(f"✅ Test Accuracy: {test_acc:.4f}")
#     print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
#     print(f"🕒 Test Time: {(test_end - test_start):.2f} seconds")
#
#     return {
#         "train_time": train_end - train_start,
#         "test_time": test_end - test_start,
#         "test_acc": test_acc,
#         "precision": precision,
#         "recall": recall,
#         "f1": f1
#     }

# ------------------------ MAIN ------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GNN Trainer (80/10/10 Split)")
    parser.add_argument('--model', type=str, choices=['GCN','SAGE','GAT','GIN','UniMP','graphormer','GPS'], default='GAT')
    parser.add_argument('--dataset_name', type=str, choices=['bus37','bus123'], default='SF')
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    args, _ = parser.parse_known_args()

    A, node_fe, Class,pdata,G = process_data(args.dataset_name, 20)
    data_list = load_graph_data(A, node_fe, Class)

    print(f"\n================== Running Model: {args.model} ==================")
    results = run_train_val_test(
        data_list,
        model_name=args.model,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )

    print("\n📊 Summary:")
    print(f"Train Time: {results['train_time']:.2f}s | Test Time: {1000 * results['test_time']:.2f}ms")
    print(f"CPU ΔMem: {results['cpu_mem']:.2f} MB | GPU Peak: {results['gpu_mem']:.2f} MB")
    print(
        f"Test Acc: {results['test_acc']:.4f}, Precision: {results['precision']:.4f}, Recall: {results['recall']:.4f}, F1: {results['f1']:.4f}")
