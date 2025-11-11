import torch
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv, SAGEConv, GATConv, GINConv, TAGConv, ChebConv, ARMAConv,
    TransformerConv, GPSConv, global_mean_pool
)
from torch.nn import Linear, Embedding
from torch_geometric.nn import GATv2Conv
import sys
# Argument parsing

# Custom GNN Model
class GNN(torch.nn.Module):
    def __init__(self, model_type, in_channels, hidden_channels, num_classes):
        super().__init__()
        self.model_type = model_type

        if model_type == 'GCN':
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
        elif model_type == 'SAGE':
            self.conv1 = SAGEConv(in_channels, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        elif model_type == 'GAT':
            self.conv1 = GATConv(in_channels, hidden_channels, heads=2, concat=False)
            self.conv2 = GATConv(hidden_channels, hidden_channels, heads=2, concat=False)
        elif model_type == 'GIN':
            nn1 = torch.nn.Sequential(
                torch.nn.Linear(in_channels, hidden_channels), torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels, hidden_channels))
            nn2 = torch.nn.Sequential(
                torch.nn.Linear(hidden_channels, hidden_channels), torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels, hidden_channels))
            self.conv1 = GINConv(nn1)
            self.conv2 = GINConv(nn2)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        self.lin = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        return self.lin(x)



# Standard GNNs
class GNN(torch.nn.Module):
    def __init__(self, model_type, in_channels, hidden_channels, num_classes):
        super().__init__()
        self.model_type = model_type

        if model_type == 'GCN':
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
        elif model_type == 'SAGE':
            self.conv1 = SAGEConv(in_channels, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        elif model_type == 'GAT':
            self.conv1 = GATConv(in_channels, hidden_channels, heads=2, concat=False)
            self.conv2 = GATConv(hidden_channels, hidden_channels, heads=2, concat=False)
        elif model_type == 'GIN':
            nn1 = torch.nn.Sequential(
                torch.nn.Linear(in_channels, hidden_channels), torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels, hidden_channels))
            nn2 = torch.nn.Sequential(
                torch.nn.Linear(hidden_channels, hidden_channels), torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels, hidden_channels))
            self.conv1 = GINConv(nn1)
            self.conv2 = GINConv(nn2)
        elif model_type == 'TAG':
            self.conv1 = TAGConv(in_channels, hidden_channels)
            self.conv2 = TAGConv(hidden_channels, hidden_channels)
        elif model_type == 'Cheb':
            self.conv1 = ChebConv(in_channels, hidden_channels, K=3)
            self.conv2 = ChebConv(hidden_channels, hidden_channels, K=3)
        elif model_type == 'ARMA':
            self.conv1 = ARMAConv(in_channels, hidden_channels)
            self.conv2 = ARMAConv(hidden_channels, hidden_channels)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        self.lin = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        return self.lin(x)


# Transformer-based GNNs
class GraphTransformer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, heads=4):
        super().__init__()
        self.conv1 = TransformerConv(in_channels, hidden_channels, heads=heads, concat=False)
        self.conv2 = TransformerConv(hidden_channels, hidden_channels, heads=heads, concat=False)
        self.lin = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        return self.lin(x)


class GPSModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super().__init__()
        self.input_proj = torch.nn.Linear(in_channels, hidden_channels)

        self.conv1 = GPSConv(
            channels=hidden_channels,
            conv=GCNConv(hidden_channels, hidden_channels),
            heads=2
        )
        self.conv2 = GPSConv(
            channels=hidden_channels,
            conv=GCNConv(hidden_channels, hidden_channels),
            heads=2
        )
        self.lin = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.input_proj(x)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        return self.lin(x)



class Graphormer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, num_layers=2, heads=4, max_degree=10):
        super().__init__()
        self.input_proj = Linear(in_channels, hidden_channels)

        # Structural encodings (e.g., node degree encoding as Graphormer does)
        self.degree_emb = Embedding(max_degree + 1, hidden_channels)

        self.layers = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                GATv2Conv(hidden_channels, hidden_channels // heads, heads=heads, concat=True)
            )

        self.norms = torch.nn.ModuleList([torch.nn.LayerNorm(hidden_channels) for _ in range(num_layers)])

        self.classifier = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch, deg=None):
        x = self.input_proj(x)

        if deg is not None:
            deg = deg.clamp(max=self.degree_emb.num_embeddings - 1)
            x = x + self.degree_emb(deg)

        for conv, norm in zip(self.layers, self.norms):
            residual = x
            x = conv(x, edge_index)
            x = F.relu(x)
            x = norm(x + residual)

        x = global_mean_pool(x, batch)
        return self.classifier(x)
