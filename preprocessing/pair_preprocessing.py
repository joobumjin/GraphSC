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

class PairData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index1':
            return self.x1.size(0)
        if key == 'edge_index2':
            return self.x2.size(0)
        if key == 'edge_weights1':
            return self.x1.size(0)
        if key == 'edge_weights2':
            return self.x2.size(0)
        return super().__inc__(key, value, *args, **kwargs)

def check_for_nan(dataset):
    for i, data in enumerate(dataset):
        if torch.isnan(data.x1).any() or torch.isnan(data.x2).any() :
            print(f"NaN found in features at index {i}")
        if torch.isnan(data.y).any():
            print(f"NaN found in target at index {i}")

def load_dataset_from_pickle(pickle_file):
    with open(pickle_file, 'rb') as f:
        dataset = pickle.load(f)
    if isinstance(dataset, list) and all(isinstance(d, PairData) for d in dataset):
        return dataset
    else:
        raise ValueError("The loaded dataset is not a list of Data objects).")

def get_loaders(data_dirs, target, batch_size):
    train_pickle_file = data_dirs[f"Train_{target}"]
    val_pickle_file = data_dirs[f"Valid_{target}"]
    test_pickle_file = data_dirs[f"Test_{target}"]

    train_dataset = load_dataset_from_pickle(train_pickle_file)
    val_dataset = load_dataset_from_pickle(val_pickle_file)
    test_dataset = load_dataset_from_pickle(test_pickle_file)

    check_for_nan(train_dataset)
    check_for_nan(val_dataset)
    check_for_nan(test_dataset)

    num_features = train_dataset[0].x1.shape[1]  # Number of features per node
    num_targets = train_dataset[0].y.shape[0]

    detail_list = [num_features, num_targets]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, follow_batch=["x1", "x2"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, follow_batch=["x1", "x2"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, follow_batch=["x1", "x2"], shuffle=False)

    return train_loader, val_loader, test_loader, detail_list