import os
import networkx as nx
import numpy as np
import pandas as pd
import math
import pyflagser
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
G=nx.read_gml("/home/mxu200014/sanfran/SFOP1UEx.gml")
A = nx.to_numpy_array(G)
N=list(G.nodes)
E=list(G.edges)
import pickle
with open('/home/mxu200014/sanfran/PartialObservdata.pkl', 'rb') as f:
    POData = pickle.load(f)
N_Senario=len(POData)
F_voltage=[0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.90, 0.85, 0.80, 0.75, 0, -1]
F_Flow=[200,175,150,125,100,75,50,30,15,10,0,-1]
p=len(F_voltage)
q=len(F_Flow)
Class=[]
for i in range(300):
    Class.append(POData[i]["NetOutage"])
#print(Class)
i=0
l=0
m=0
while i < len(Class):
    if Class[i] == 'No':
        l=l+1
        Class[i] = 0
    if Class[i] == 'Yes':
        m=m+1
        Class[i] = 1
    i += 1
AverageVoltage=[]
def Average(lst):
    return sum(lst) / len(lst)

import time
start = time.time()
list_b0=[]
list_b1=[]
N_Senario=300
for k in range(N_Senario):
    #print("\rProcessing file {} ({}%)".format(k, 100*k//(N_Senario-1)), end='', flush=True)
    AverageVoltage=[]
    Voltage=POData[k]["Bus Voltages"]
    for x, y in Voltage.items():
        AverageVoltage.append(Average(list(y)))
    AverageVoltage = [-1 if math.isnan(x) else x for x in AverageVoltage]
    #print(AverageVoltage)
    BranchFlow=[]
    Bflow=POData[k]["BranchFlow"]
    for x, y in Bflow.items():
        BranchFlow.append(y)
    #print(BranchFlow)
    BranchFlow = [-1 if math.isnan(x) else x for x in BranchFlow]
    #print(BranchFlow)
    b0_points=[]
    b1_points=[]
    for p in range(len(F_voltage)):
        for q in range(len(F_Flow)):
            n_active = np.where(np.array(AverageVoltage) > F_voltage[p])[0].tolist()
            indices = np.where(np.array(BranchFlow) > F_Flow[q])[0].tolist()
            for s in indices:
                n_active.append(int(N.index(E[s][0])))
                n_active.append(int(N.index(E[s][1])))
            Active_node=np.unique(n_active)
            if (len(Active_node)==0):
                b=np.array([[0,0],[0,0]])
            else:
                b=A[Active_node,:][:,Active_node]
            #print(Active_node)
            #b=A[Active_node,:][:,Active_node]
            my_flag=pyflagser.flagser_unweighted(b, min_dimension=0, max_dimension=2, directed=False, coeff=2, approximation=None)
            x = my_flag["betti"]
            #print(x[0])
            #x_points.append(unique_list[k])
            b0_points.append(x[0])
            b1_points.append(x[1])
            #print(b)
            #print(x_points)
            n_active.clear()
    #print(b0_points)
    list_b0.append(b0_points)
    list_b1.append(b1_points)
end = time.time()
print("The time of execution of above program is :", (end-start), "s")

x =np.array(list_b0)
x1 =np.array(list_b1)
n=len(list_b0[0])
p=len(F_voltage)
q=len(F_Flow)
Feture=[]
for i in range(p*q):
    Feture.append("{}".format(i))
#Feture=list(range(9,17))
data = pd.DataFrame(x, columns =Feture)
data.insert(loc=p*q,column='Class',value=Class)
data.head(10)


compression_opts = dict(method='zip',archive_name='Feature_SanMulti.csv')
data.to_csv('Feature_SanMulti.zip', index=True,compression=compression_opts)

from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
print("XGBoost using only Branch Flow")

feature=[]
for i in range(q):
    feature.append("{}".format(i))


X=data[feature].astype("float") # Features
y=data['Class'].astype("category")  # Labels
#y=data["Class"].astype("category")
#print(X)

bst = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1,enable_categorical=True)
scores = cross_val_score(bst, X, y, cv=10)
print("%0.3f accuracy with a standard deviation of %0.2f" % (scores.mean()*100, scores.std()))