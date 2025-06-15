from load_data import load_data_outage
from module import *
import argparse

parser = argparse.ArgumentParser(description="Mp-Grid for Outage Detection")
parser.add_argument('--bus', type=str, default='bus37')
parser.add_argument('--model', type=str, default='SP_Bflow')#SP_voltage,SP_Bflow
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--cvs', type=int, default=2)
parser.add_argument('--scenarios', type=int, default=600)
args = parser.parse_args()
#Load dataset
data,Label,Graph=load_data_outage(args.bus,args.scenarios)
#define filtration
F_voltage=np.array([0.99,0.98,0.97,0.96,0.95,0.94,0.93,0.90,0.85,0.80,0.75,0,-1])
F_Flow=np.array([200,175,150,125,100,75,50,30,15,10,0,-1])
#Extract Topological features
if args.model=="MP":
    betti0 = MP_feature(data, Graph, F_voltage, F_Flow, args.scenarios)
elif args.model=="SP_voltage":
    betti0 = SP_Voltage(data, Graph, F_voltage, args.scenarios)
elif args.model=="SP_Bflow":
    betti0 = SP_Bflow(data, Graph, F_Flow, args.scenarios)
else:
    print("Model error: model is not defined")



from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
print("XGBoost")


X=betti0# Features
y=Label  # Labels

bst = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1)
scores = cross_val_score(bst, X, y, cv=10)
print("%0.3f accuracy with a standard deviation of %0.4f" % (scores.mean()*100, scores.std()))