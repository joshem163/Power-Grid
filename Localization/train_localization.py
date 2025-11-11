import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, hamming_loss
import numpy as np
from tqdm import tqdm
import argparse
import sys
# Argument parsing
from models import GNN,GPSModel,GraphTransformer,Graphormer
from load_data import process_data,load_graph_data, print_stat,load_data_SubLocal

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
# Training function
def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

# Test function
def test(model, loader):
    model.eval()
    all_probs, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            probs = torch.sigmoid(out).cpu()
            preds = (probs > 0.4).float()
            all_probs.append(probs)
            all_preds.append(preds)
            all_labels.append(data.y.cpu())

    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.cat(all_preds).numpy()
    y_prob = torch.cat(all_probs).numpy()

    # --- Metrics ---
    exact_match = accuracy_score(y_true, y_pred)
    roc_auc_micro = roc_auc_score(y_true, y_prob, average='micro')
    roc_auc_macro = roc_auc_score(y_true, y_prob, average='macro')
    f1_micro = f1_score(y_true, y_pred, average='micro')
    hamming = hamming_loss(y_true, y_pred)

    metrics = {
        "Exact Match": exact_match,
        "ROC-AUC (micro)": roc_auc_micro,
        "ROC-AUC (macro)": roc_auc_macro,
        "F1 (micro)": f1_micro,
        "Hamming Loss": hamming,
    }
    return metrics

# Cross-validation runner
def run_cross_validation(data_list, model_name, hidden_dim, epochs, folds, batch_size, lr):
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    acc_list=[]
    f1_list=[]
    auc_list=[]

    for fold, (train_idx, test_idx) in enumerate(kf.split(data_list), 1):
        print(f'\n🔁 Fold {fold}')
        train_data = [data_list[i] for i in train_idx]
        test_data = [data_list[i] for i in test_idx]

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size)
        num_labels = 6

        if model_name in ['GCN', 'SAGE', 'GAT', 'GIN']:
            model = GNN(model_name, in_channels=data_list[0].x.size(1),
                        hidden_channels=hidden_dim, num_classes=num_labels).to(device)
        elif model_name == 'UniMP':
            model = GraphTransformer(in_channels=data_list[0].x.size(1),
                                     hidden_channels=hidden_dim, num_classes=num_labels).to(device)
        elif model_name == 'graphormer':
            model = Graphormer(in_channels=data_list[0].x.size(1),
                               hidden_channels=hidden_dim, num_classes=num_labels).to(device)
        elif model_name == 'GPS':
            model = GPSModel(in_channels=data_list[0].x.size(1),
                             hidden_channels=hidden_dim, num_classes=num_labels).to(device)
        else:
            raise ValueError(f"Model '{model_name}' is not defined.")

        # model = GNN(model_type, in_channels=train_data[0].x.size(1),
        #             hidden_channels=hidden_dim, num_classes=2).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.BCEWithLogitsLoss()

        fold_metrics = []
        for epoch in range(1, epochs + 1):
            loss = train(model, train_loader, optimizer, criterion)
            metrics = test(model, test_loader)

            print(f"[Epoch {epoch}] Loss: {loss:.4f} | "
                  f"Exact Match: {metrics['Exact Match']:.4f} | "
                  f"Micro AUC: {metrics['ROC-AUC (micro)']:.4f} | "
                  f"Macro AUC: {metrics['ROC-AUC (macro)']:.4f} | "
                  f"Micro F1: {metrics['F1 (micro)']:.4f} | "
                  f"Hamming: {metrics['Hamming Loss']:.4f}")
            fold_metrics.append(metrics)
        #Find best epoch based on Exact Match
        best_idx = np.argmax([m["Exact Match"] for m in fold_metrics])
        best_metrics = fold_metrics[best_idx]

        print(f"\n✅ Fold {fold} Summary:")
        print(f"   Best Epoch: {best_idx + 1}")
        for k in best_metrics:
            print(f"   {k:<18s}: best = {best_metrics[k]:.4f}")

        # === Store the best metrics for final 10-fold aggregation ===
        acc_list.append(best_metrics["Exact Match"])
        f1_list.append(best_metrics["F1 (micro)"])
        auc_list.append(best_metrics["ROC-AUC (micro)"])

    print("\n📊 Final 10-Fold Cross-Validation Summary:")
    print(f"Exact Match: {np.mean(acc_list):.4f} ± {np.std(acc_list):.4f}")
    print(f"Micro-F1    : {np.mean(f1_list):.4f} ± {np.std(f1_list):.4f}")
    print(f"Micro-AUC   : {np.mean(auc_list):.4f} ± {np.std(auc_list):.4f}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GNN Model Trainer with Cross Validation")
    parser.add_argument('--model', type=str, choices=['GCN', 'SAGE', 'GAT', 'GIN', 'all'], default='all',
                        help='Type of GNN model to use or "all" to run all models')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden layer dimension')
    parser.add_argument('--epochs', type=int, default=2, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--folds', type=int, default=10, help='Number of folds for cross-validation')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    args, _ = parser.parse_known_args()  # Jupyter/IPython-safe

    # Load your data
    A,node_fe,Class,POdata,G=load_data_SubLocal(600)
    print(Class)
    data_list = load_graph_data(A, node_fe, Class)  # Replace with your actual data

    models_to_run = ['GCN','SAGE', 'GAT', 'GIN','UniMP', 'graphormer','GPS'] if args.model == 'all' else [args.model]
    #models_to_run = ['GCN' ] if args.model == 'all' else [args.model]

    for modelName in models_to_run:
        print(f"\n================== Running Model: {modelName} ==================")
        run_cross_validation(
            data_list,
            model_name=modelName,
            hidden_dim=args.hidden_dim,
            epochs=args.epochs,
            folds=args.folds,
            batch_size=args.batch_size,
            lr=args.lr
        )


