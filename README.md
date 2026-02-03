![MultiG](https://github.com/joshem163/Power-Grid/assets/133717791/0331ee4e-c6db-4bb6-8a75-8b5e4830c9ae)

# MP-Grid (Power-Grid): Detecting Power Grid Outages with Topological Machine Learning

Official implementation of **MP-Grid**, a topology-informed learning pipeline for **power distribution outage detection and localization** using **multiparameter persistent homology** (MPH) and lightweight classifiers.

> Paper: **MP-Grid: Detecting Power Grid Outages with Topological Machine Learning**  
> Authors: Md Joshem Uddin, Damilola R. Olojede, Roshni Anna Jacob, Baris Coskunuzer, Jie Zhang  
> Code + datasets + trained parameters: https://github.com/joshem163/Power-Grid

---

## Overview

Power distribution outages (from extreme weather, equipment faults, or cyber-physical events) alter both the **state** and **topology** of the grid. MP-Grid captures these changes by computing **multiparameter persistent homology summaries** using:
- **Bus voltages** as a node filtration function, and
- **Branch flows/currents** as an edge filtration function,

then vectorizing topological summaries (e.g., Betti-0 signatures over a 2D threshold grid) into a **fixed-length feature vector** for classification (default: **XGBoost**).

---

## What’s in this repository

At a high level, the repository contains:
- **MP-Grid training/inference** scripts (e.g., `train_MPgrid.py`, runtime scripts),
- **Baseline models** training scripts (e.g., `train_baseline.py`, metric/runtime scripts),
- Data utilities (e.g., `load_data.py`),
- Model utilities (e.g., `models.py`, `module.py`),
- Notebooks for experiments/visualization,
- Task-specific folders such as `Localization/` and topological feature utilities.

---

## Datasets

The paper evaluates MP-Grid on IEEE test feeders and realistic synthetic networks, including:
- **IEEE 37-bus**
- **IEEE 123-bus**
- **342-node LVN**
- **IEEE 8500-bus**
- **NREL synthetic San Francisco Bay Area network (SMART-DS)**

Data are generated via power-flow simulation (e.g., OpenDSS) with outage scenarios and (optionally) partial observability.

---

## Requirements

The paper reports experiments in **Python 3.11.4** and uses a lightweight stack for MP-Grid:
- `numpy`, `pandas`
- `networkx`
- `pyflagser` (for topological computations, if used in your scripts)
- `scikit-learn`
- `xgboost`

Baselines may additionally require:
- `torch`, `torch-geometric` (for GNN baselines),
- `matplotlib` / `seaborn` (optional plotting).

---
 
# Runing the Experiments
Run the appropriate training script (e.g., train_MPgrid.py) with the correct data path to reproduce the results.

