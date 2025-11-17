from load_data import load_data_outage
from module import *
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="Mp-Grid for Outage Detection")
parser.add_argument('--bus', type=str, default='bus37')
parser.add_argument('--model', type=str, default='SP_voltage')#SP_voltage,SP_Bflow
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--cvs', type=int, default=2)
parser.add_argument('--scenarios', type=int, default=600)
args = parser.parse_args()
#Load dataset
data,Label,Graph=load_data_outage(args.bus,args.scenarios)
#define filtration
F_voltage=np.array([0.99,0.98,0.97,0.96,0.95,0.94,0.93,0.90,0.85,0.80,0.75,0.5,0.25,0,-1])
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

X = np.array(betti0)  # Features
y = np.array(Label)   # Labels
# Convert to DataFrame for easy plotting
df = pd.DataFrame(X, columns=[f"Feature_{i+1}" for i in range(X.shape[1])])
df["Label"] = y

# Convert to DataFrame for easier handling
df = pd.DataFrame(X, columns=[f"F{i+1}" for i in range(X.shape[1])])
df["label"] = y
#
# plt.figure(figsize=(8, 5))
# plt.rcParams.update({'font.size': 13})
#
# # Define colors for the two classes
# color_map = {0: "tab:blue", 1: "tab:orange"}
#
# # Plot each sample's Betti curve, colored by its class
# for i in range(10):
#     lbl = y[i]
#     plt.plot(range(1, X.shape[1] + 1),
#              X[i, :],
#              color=color_map[lbl],
#              alpha=0.4, linewidth=1.5)
#
# # Style and labels
# plt.xlabel("Filtration Step", fontsize=14)
# plt.ylabel(r"Betti Number ($\beta_0$)", fontsize=14)
# plt.title(r"Betti Curves Colored by Class", fontsize=15)
#
# # Legend
# from matplotlib.lines import Line2D
# custom_lines = [
#     Line2D([0], [0], color="tab:blue", lw=2),
#     Line2D([0], [0], color="tab:orange", lw=2)
# ]
# plt.legend(custom_lines, ["Normal", "Outage"],
#            title="Class", fontsize=12, title_fontsize=13, loc="upper right")
#
# plt.tight_layout()
# plt.show()
#
# Melt data: each feature as a point along the x-axis
df_melted = df.melt(id_vars="label", var_name="Feature", value_name="Value")

# Encode feature names numerically for regression
df_melted["Feature_idx"] = df_melted["Feature"].str.extract("(\d+)").astype(int)

# Plot setup
plt.figure(figsize=(8, 5))
plt.rcParams.update({'font.size': 13})  # increase overall font size

# Disable Seaborn's automatic legend
sns.scatterplot(
    data=df_melted,
    x="Feature_idx",
    y="Value",
    hue="label",
    alpha=0.5,
    s=40,
    palette={0: "tab:blue", 1: "tab:orange"},
    legend=False      # 👈 turn off default legend
)

# Fit and plot regression line for each class
for lbl, color in zip([0, 1], ["tab:blue", "tab:orange"]):
    sub = df_melted[df_melted["label"] == lbl]
    X_fit = sub["Feature_idx"].values.reshape(-1, 1)
    y_fit = sub["Value"].values
    model = LinearRegression().fit(X_fit, y_fit)
    x_range = np.linspace(df_melted["Feature_idx"].min(), df_melted["Feature_idx"].max(), 100).reshape(-1, 1)
    y_pred = model.predict(x_range)
    plt.plot(x_range, y_pred, color=color, linewidth=2.5, label=f"{'Normal' if lbl==0 else 'Outage'}")

# Style adjustments
plt.xlabel("Filtration thresholds", fontsize=16)
plt.ylabel(r"Feature Value ($\beta_0$)", fontsize=16)
#plt.title("Feature-wise Values with Regression Lines per Class", fontsize=15)

# ✅ Manual legend with correct colors
plt.legend(
    title="Class",
    fontsize=15,
    title_fontsize=13,
    loc="lower right",
    frameon=True
)

plt.tight_layout()
output_path = "bettiVSoutage.pdf"   # or "clustering_results.pdf"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()

#
# # Convert to DataFrame for easy plotting
# df = pd.DataFrame(X, columns=[f"Feature_{i+1}" for i in range(X.shape[1])])
# df["Label"] = y
#
# # Convert to DataFrame for easier handling
# df = pd.DataFrame(X, columns=[f"F{i+1}" for i in range(X.shape[1])])
# df["label"] = y
#
# # Melt data: each feature as a point along the x-axis
# df_melted = df.melt(id_vars="label", var_name="Feature", value_name="Value")
#
# # Encode feature names numerically for regression
# df_melted["Feature_idx"] = df_melted["Feature"].str.extract("(\d+)").astype(int)
#
# # Plot setup
# plt.figure(figsize=(8, 5))
# sns.scatterplot(data=df_melted, x="Feature_idx", y="Value", hue="label", alpha=0.5)
#
# # Fit and plot regression line for each class
# for lbl, color in zip([0, 1], ["tab:blue", "tab:orange"]):
#     sub = df_melted[df_melted["label"] == lbl]
#     X_fit = sub["Feature_idx"].values.reshape(-1, 1)
#     y_fit = sub["Value"].values
#     model = LinearRegression().fit(X_fit, y_fit)
#     x_range = np.linspace(df_melted["Feature_idx"].min(), df_melted["Feature_idx"].max(), 100).reshape(-1, 1)
#     y_pred = model.predict(x_range)
#     plt.plot(x_range, y_pred, color=color, linewidth=2, label=f"Class {lbl} regression")
#
# # Style adjustments
# plt.xlabel("Feature Index")
# plt.ylabel("Feature Value")
# plt.title("Feature-wise Values with Regression Lines per Class")
# plt.legend(title="Class")
# plt.tight_layout()
# plt.show()
