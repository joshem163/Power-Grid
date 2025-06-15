import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
from sklearn.model_selection import KFold
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
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)
            correct += (pred == data.y).sum().item()
    return correct / len(loader.dataset)

# Cross-validation runner
def run_cross_validation(data_list, model_name, hidden_dim, epochs, folds, batch_size, lr):
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    acc_list = []

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

        for epoch in tqdm(range(1, epochs + 1), desc=f"Training Epochs (Fold {fold})"):
            loss, train_acc = train(model, train_loader, optimizer, criterion)
            test_acc = test(model, test_loader)
            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)
            print(test_acc)
        accuracy = print_stat(train_accuracies, test_accuracies)
        acc_list.append(accuracy[0])
        print(f'✅ Test Accuracy Fold {fold}: {test_acc:.4f}')

    print("\n📊 Final 10-Fold Cross Validation Accuracy:")
    print(f"Mean: {np.mean(acc_list):.4f} ± {np.std(acc_list):.4f}")

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
    A,node_fe,Class=process_data(300)
    print(Class)
    data_list = load_graph_data(A, node_fe, Class)  # Replace with your actual data

    #models_to_run = ['GCN', 'SAGE', 'GAT', 'GIN'] if args.model == 'all' else [args.model]
    models_to_run = [ 'UniMP', 'graphormer', 'GPS'] if args.model == 'all' else [args.model]

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