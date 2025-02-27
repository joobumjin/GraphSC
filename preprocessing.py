import argparse
from tqdm import tqdm

import math
import os
from pathlib import Path

import pandas as pd
import numpy as np
import pickle
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

def check_for_nan(dataset):
    for i, data in enumerate(dataset):
        if torch.isnan(data.x).any():
            print(f"NaN found in features at index {i}")
        if torch.isnan(data.y).any():
            print(f"NaN found in target at index {i}")

def load_dataset_from_pickle(pickle_file):
    with open(pickle_file, 'rb') as f:
        dataset = pickle.load(f)
    if isinstance(dataset, list) and all(isinstance(d, Data) for d in dataset):
        return dataset
    else:
        raise ValueError("The loaded dataset is not a list of Data objects).")
    
def print_stats(train_dataset, val_dataset, test_dataset, num_features, num_targets, print_detailed = False):
    print("###################################")
    print()
    print(f'Number of training graphs: {len(train_dataset)}')
    print(f'Number of validation graphs: {len(val_dataset)}')
    print(f'Number of test graphs: {len(test_dataset)}')
    print(f'Number of features: {num_features}')
    print(f'Number of targets: {num_targets}')
    print("###################################")

    if print_detailed:
        data = train_dataset[0]
        print()
        print(data)
        print('=============================================================')
        print(f'Number of nodes: {data.num_nodes}')
        print(f'Number of edges: {data.num_edges}')
        print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
        print(f'Has isolated nodes: {data.has_isolated_nodes()}')
        print(f'Has self-loops: {data.has_self_loops()}')
        print(f'Is undirected: {data.is_undirected()}')
        print('=============================================================')
        print()
        print(f'Number of training graphs: {len(train_dataset)}')
        print(f'Number of test graphs: {len(val_dataset)}')
        print('=============================================================')
        print()
        # for step, data in enumerate(train_loader):
        #     print(f'Step {step + 1}:')
        #     print('=======')
        #     print(f'Number of graphs in the current batch: {data.num_graphs}')
        #     print(data)
        #     print()
        print()

def get_loaders(data_dirs, target, batch_size, print_data_stats = True, print_detailed = False):
    train_pickle_file = data_dirs[f"Train_{target}"]
    val_pickle_file = data_dirs[f"Valid_{target}"]
    test_pickle_file = data_dirs[f"Test_{target}"]

    train_dataset = load_dataset_from_pickle(train_pickle_file)
    val_dataset = load_dataset_from_pickle(val_pickle_file)
    test_dataset = load_dataset_from_pickle(test_pickle_file)

    check_for_nan(train_dataset)
    check_for_nan(val_dataset)
    check_for_nan(test_dataset)

    num_features = train_dataset[0].x.shape[1]  # Number of features per node
    num_targets = train_dataset[0].y.shape[0]

    detail_list = [num_features, num_targets]
    
    if print_data_stats: print_stats(train_dataset, val_dataset, test_dataset, num_features, num_targets, print_detailed)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, detail_list