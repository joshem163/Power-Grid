from load_data import load_data_outage
from module import *
import argparse
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="Mp-Grid for Outage Detection")
parser.add_argument('--bus', type=str, default='bus123')
parser.add_argument('--model', type=str, default='MP')#SP_voltage,SP_Bflow
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
from sklearn.model_selection import GridSearchCV

X = betti0  # Features
y = Label   # Labels

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300,500],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# Initialize the classifier
xgb = XGBClassifier(eval_metric='logloss')

# Set up GridSearchCV
grid_search = GridSearchCV(estimator=xgb,
                           param_grid=param_grid,
                           scoring='accuracy',
                           cv=10,
                           verbose=1,
                           n_jobs=-1)

# Fit the model
grid_search.fit(X, y)
# Mean accuracy and standard deviation from the best model
mean_acc = grid_search.cv_results_['mean_test_score'][grid_search.best_index_]
std_acc = grid_search.cv_results_['std_test_score'][grid_search.best_index_]

print(f"Best Accuracy: {mean_acc:.4f}% ± {std_acc:.4f}")
print("Best Parameters:", grid_search.best_params_)
