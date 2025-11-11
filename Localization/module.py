import os
import networkx as nx
import numpy as np
from tqdm import tqdm
import pandas as pd
import math
import pyflagser
def Average(lst):
    return sum(lst) / len(lst)
def MP_feature(POData,Graph,F_voltage,F_Flow,N_Senario):
    N = list(Graph.nodes)
    E= list(Graph.edges)
    A = nx.to_numpy_array(Graph)
    list_b0 = []
    list_b1 = []
    for k in range(N_Senario):
        print("\rProcessing file {} ({}%)".format(k, 100 * k // (N_Senario - 1)), end='', flush=True)
        AverageVoltage = []
        Voltage = POData[k]["Bus Voltages"]
        for x, y in Voltage.items():
            AverageVoltage.append(Average(list(y)))
        AverageVoltage = [-1 if math.isnan(x) else x for x in AverageVoltage]
        # print(AverageVoltage)
        BranchFlow = []
        Bflow = POData[k]["BranchFlow"]
        for x, y in Bflow.items():
            BranchFlow.append(y)
        # print(BranchFlow)
        BranchFlow = [-1 if math.isnan(x) else x for x in BranchFlow]
        # print(BranchFlow)
        b0_points = []
        b1_points = []
        for p in range(len(F_voltage)):
            for q in range(len(F_Flow)):
                n_active = np.where(np.array(AverageVoltage) > F_voltage[p])[0].tolist()
                indices = np.where(np.array(BranchFlow) > F_Flow[q])[0].tolist()
                for s in indices:
                    n_active.append(int(N.index(E[s][0])))
                    n_active.append(int(N.index(E[s][1])))
                Active_node = np.unique(n_active)
                # print(Active_node)
                b = A[Active_node, :][:, Active_node]
                my_flag = pyflagser.flagser_unweighted(b, min_dimension=0, max_dimension=2, directed=False, coeff=2,
                                                       approximation=None)
                x = my_flag["betti"]
                # print(x[0])
                # x_points.append(unique_list[k])
                b0_points.append(x[0])
                b1_points.append(x[1])
                # print(b)
                # print(x_points)
                n_active.clear()
        # print(b0_points)
        list_b0.append(b0_points)
        list_b1.append(b1_points)
    return list_b0
def MP_NX(POData,Graph,F_voltage,F_Flow,N_Senario):
    N = list(Graph.nodes)
    E= list(Graph.edges)
    A = nx.to_numpy_array(Graph)
    list_b0 = []
    list_b1 = []
    for k in tqdm(range(N_Senario),'Processing MP features'):
        #print("\rProcessing file {} ({}%)".format(k, 100 * k // (N_Senario - 1)), end='', flush=True)
        AverageVoltage = []
        Voltage = POData[k]["BusVoltages"]#Bus Voltages
        for x, y in Voltage.items():
            AverageVoltage.append(Average(list(y)))
        AverageVoltage = [-1 if math.isnan(x) else x for x in AverageVoltage]
        # print(AverageVoltage)
        BranchFlow = []
        Bflow = POData[k]["BranchFlows"]#BranchFlow
        for x, y in Bflow.items():
            BranchFlow.append(y)
        # print(BranchFlow)
        BranchFlow = [-1 if math.isnan(x) else x for x in BranchFlow]
        # print(BranchFlow)
        b0_points = []
        for p in range(len(F_voltage)):
            Active_node_v = np.where(np.array(AverageVoltage) > F_voltage[p])[0]
            for q in range(len(F_Flow)):
                if Active_node_v.size == 0:
                    b0_points.append(0)
                    continue
                G = nx.Graph()
                G.add_nodes_from(Active_node_v)
                # Find edges where branch flow exceeds threshold F_Flow[q]
                indices = np.where(np.array(BranchFlow) > F_Flow[q])[0]
                edges_to_add = [(int(N.index(E[s][0])), int(N.index(E[s][1]))) for s in indices]
                edges_to_add = [(a, b) for a, b in edges_to_add if a in Active_node_v and b in Active_node_v]
                G.add_edges_from(edges_to_add)
                b0_points.append(nx.number_connected_components(G))

        # print(b0_points)
        list_b0.append(b0_points)

    return list_b0
def SP_Voltage(POData,Graph,F_voltage,N_Senario):
    N = list(Graph.nodes)
    E= list(Graph.edges)
    A = nx.to_numpy_array(Graph)
    list_b0 = []
    for k in tqdm(range(N_Senario),'Processing SP features'):
        AverageVoltage = []
        Voltage = POData[k]["BusVoltages"]#Bus Voltages
        for x, y in Voltage.items():
            AverageVoltage.append(Average(list(y)))
        AverageVoltage = [-1 if math.isnan(x) else x for x in AverageVoltage]
        b0_points = []
        for p in range(len(F_voltage)):
            n_active = np.where(np.array(AverageVoltage) > F_voltage[p])[0].tolist()
            Active_node = np.unique(n_active)
            if Active_node.size == 0:
                b0_points.append(0)
                continue
            b = A[Active_node, :][:, Active_node]
            my_flag = pyflagser.flagser_unweighted(b, min_dimension=0, max_dimension=2, directed=False, coeff=2,
                                                       approximation=None)
            x = my_flag["betti"]
            b0_points.append(x[0])
            n_active.clear()
        # print(b0_points)
        list_b0.append(b0_points)
    return list_b0
def SP_Bflow(POData,Graph,F_Flow,N_Senario):
    N = list(Graph.nodes)
    E= list(Graph.edges)
    A = nx.to_numpy_array(Graph)
    list_b0 = []
    list_b1 = []
    for k in tqdm(range(N_Senario),'Processing SP features'):
        BranchFlow = []
        Bflow = POData[k]["BranchFlow"]
        for x, y in Bflow.items():
            BranchFlow.append(y)
        # print(BranchFlow)
        BranchFlow = [-1 if math.isnan(x) else x for x in BranchFlow]
        # print(BranchFlow)
        b0_points = []
        for q in range(len(F_Flow)):
            indices = np.where(np.array(BranchFlow) >= F_Flow[q])[0]
            n_active=[]
            for s in indices:
                n_active.append(int(N.index(E[s][0])))
                n_active.append(int(N.index(E[s][1])))
            Active_node = np.unique(n_active)
            if Active_node.size == 0:
                b0_points.append(0)
                continue
            b = A[Active_node, :][:, Active_node]
            my_flag = pyflagser.flagser_unweighted(b, min_dimension=0, max_dimension=2, directed=False, coeff=2,
                                                       approximation=None)
            x = my_flag["betti"]

            b0_points.append(x[0])
            n_active.clear()
        # print(b0_points)
        list_b0.append(b0_points)
    return list_b0