import networkx as nx
import pickle
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix
import numpy as np
import scipy.sparse as sp
import os

def load_data_outage(N_scenarios):
    G=nx.read_gml("34busEx.gml")
    A = nx.to_numpy_array(G)
    N = list(G.nodes)
    with open('LineFailures_34bus.pkl', 'rb') as f:
        POData = pickle.load(f)
    # N_scenarios=len(POData)
    node_fe=[]
    for i in range(N_scenarios):
        values = np.array([
    POData[0]['BusVoltages'].get(node, np.array([-1, -1, -1])) for node in N
])
        node_fe.append(values)
    Class=[]
    for i in range(N_scenarios):
        Class.append(POData[i]["Outage"])
    i=0
    while i < len(Class):
        if Class[i] == 'No':
            Class[i] = 0
        if Class[i] == 'Yes':
            Class[i] = 1
        i += 1
    return A, node_fe, Class


def process_data(N_scenarios):
    G=nx.read_gml("8500busEx.gml")
    #G = nx.read_gml("SFOP1UEx.gml")
    A = nx.to_numpy_array(G)
    with open('PartialObservdata.pkl', 'rb') as f:
        POData = pickle.load(f)
    node_fe=[]
    for i in range(N_scenarios):
        values = np.array(list(POData[i]['Bus Voltages'].values()))
        values = np.nan_to_num(values, nan=-1)
        node_fe.append(values)
    Class=[]
    for i in range(N_scenarios):
        Class.append(POData[i]["NetOutage"])
    i=0
    while i < len(Class):
        if Class[i] == 'No':
            Class[i] = 0
        if Class[i] == 'Yes':
            Class[i] = 1
        i += 1
    return A, node_fe, Class


def load_graph_data(adj_matrix, node_features_list, graph_labels):
    """
    Args:
        adj_matrix: numpy array or scipy sparse matrix of shape [N, N] (shared across all graphs)
        node_features_list: list of NumPy arrays or torch tensors of shape [N, F]
        graph_labels: list or array of graph-level labels
    Returns:
        List of torch_geometric.data.Data objects
    """
    if isinstance(adj_matrix, np.ndarray):
        adj_matrix = sp.coo_matrix(adj_matrix)

    edge_index, edge_weight = from_scipy_sparse_matrix(adj_matrix)

    data_list = []
    for i in range(len(node_features_list)):
        x = torch.tensor(node_features_list[i], dtype=torch.float)
        y = torch.tensor([graph_labels[i]], dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, y=y)
        data_list.append(data)

    return data_list


def print_stat(train_acc, test_acc):
    argmax = np.argmax(train_acc)
    best_result = test_acc[argmax]
    train_ac = np.max(train_acc)
    test_ac = np.max(test_acc)
    #print(f'Train accuracy = {train_ac:.4f}%,Test Accuracy = {test_ac:.4f}%\n')
    return test_ac, best_result
path='/Users/joshem/PhD Research/Power Distribution Network/MP-Grid/Experiment/data'
def load_data_outage(bus_name,N_scenarios):
    base_path = os.path.join(path,bus_name)
    graph_path = os.path.join(base_path, f"{bus_name}Ex.gml")
    pickle_path = os.path.join(base_path, 'PartialObservdata.pkl')
    #G=nx.read_gml("8500busEx.gml")
    G = nx.read_gml(graph_path)
    A = nx.to_numpy_array(G)
    N = list(G.nodes)
    E= list(G.edges)
    with open(pickle_path, 'rb') as f:
        POData = pickle.load(f)
    Class=[]
    for i in range(N_scenarios):
        Class.append(POData[i]["NetOutage"])
    i=0
    while i < len(Class):
        if Class[i] == 'No':
            Class[i] = 0
        if Class[i] == 'Yes':
            Class[i] = 1
        i += 1
    return POData, Class,G
