import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from tqdm import tqdm
import argparse
import sys
# Argument parsing
from models import GNN,GPSModel,GraphTransformer,Graphormer
from load_data import process_data,load_graph_data, print_stat

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
# Training function
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

# Test function
def test(model, loader):
    model.eval()
    correct = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)
            all_preds.append(pred.cpu())
            all_labels.append(data.y.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    accuracy = (all_preds == all_labels).sum().item() / len(all_labels)
    return accuracy, all_preds.numpy(), all_labels.numpy()

# Cross-validation runner
def run_cross_validation(data_list, model_name, hidden_dim, epochs, folds, batch_size, lr):
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    acc_list = []
    pre_list=[]
    rec_list=[]
    f1_list=[]

    for fold, (train_idx, test_idx) in enumerate(kf.split(data_list), 1):
        print(f'\n🔁 Fold {fold}')
        train_data = [data_list[i] for i in train_idx]
        test_data = [data_list[i] for i in test_idx]

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size)

        if model_name in ['GCN', 'SAGE', 'GAT', 'GIN']:
            model = GNN(model_name, in_channels=data_list[0].x.size(1),
                        hidden_channels=hidden_dim, num_classes=2).to(device)
        elif model_name == 'UniMP':
            model = GraphTransformer(in_channels=data_list[0].x.size(1),
                                     hidden_channels=hidden_dim, num_classes=2).to(device)
        elif model_name == 'graphormer':
            model = Graphormer(in_channels=data_list[0].x.size(1),
                               hidden_channels=hidden_dim, num_classes=2).to(device)
        elif model_name == 'GPS':
            model = GPSModel(in_channels=data_list[0].x.size(1),
                             hidden_channels=hidden_dim, num_classes=2).to(device)
        else:
            raise ValueError(f"Model '{model_name}' is not defined.")

        # model = GNN(model_type, in_channels=train_data[0].x.size(1),
        #             hidden_channels=hidden_dim, num_classes=2).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()

        train_accuracies = []
        test_accuracies = []
        test_pre = []
        test_rec = []
        test_f1 = []

        for epoch in range(1, epochs + 1):
            loss, train_acc = train(model, train_loader, optimizer, criterion)
            test_acc, y_pred, y_true = test(model, test_loader)
            print("\n🧮 Confusion Matrix:")
            print(confusion_matrix(y_true, y_pred))

            precision = precision_score(y_true, y_pred, average='binary')
            recall = recall_score(y_true, y_pred, average='binary')
            f1 = f1_score(y_true, y_pred, average='binary')

            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)
            test_pre.append(precision)
            test_rec.append(recall)
            test_f1.append(f1)

            print(
                f"[Epoch {epoch}] Test Acc: {test_acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
        accuracy = print_stat(train_accuracies, test_accuracies)
        preci = print_stat(train_accuracies, test_pre)
        reca = print_stat(train_accuracies, test_rec)
        f1sc = print_stat(train_accuracies, test_f1)
        acc_list.append(accuracy[1])
        pre_list.append(preci[1])
        rec_list.append(reca[1])
        f1_list.append(f1sc[1])
        # pre_list.append(np.max(test_pre))
        # rec_list.append(np.max(test_rec))
        # f1_list.append(np.max(test_f1))

        print(f'✅ Test Accuracy Fold {fold}: {test_acc:.4f}')

    print("\n📊 Final 10-Fold Cross Validation Accuracy:")
    print(
        f"ACC: {np.mean(acc_list):.4f} ± {np.std(acc_list):.4f},pre: {np.mean(pre_list):.4f} ± {np.std(pre_list):.4f},rec: {np.mean(rec_list):.4f} ± {np.std(rec_list):.4f},f1: {np.mean(f1_list):.4f} ± {np.std(f1_list):.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GNN Model Trainer with Cross Validation")
    parser.add_argument('--model', type=str, choices=['GCN', 'SAGE', 'GAT', 'GIN', 'all'], default='all',
                        help='Type of GNN model to use or "all" to run all models')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden layer dimension')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--folds', type=int, default=10, help='Number of folds for cross-validation')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    args, _ = parser.parse_known_args()  # Jupyter/IPython-safe

    # Load your data
    A,node_fe,Class=process_data(600)
    print(Class)
    data_list = load_graph_data(A, node_fe, Class)  # Replace with your actual data

    models_to_run = ['GCN','SAGE', 'GAT', 'GIN','UniMP', 'graphormer'] if args.model == 'all' else [args.model]
    #models_to_run = [ 'UniMP', 'graphormer', 'GPS'] if args.model == 'all' else [args.model]

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


