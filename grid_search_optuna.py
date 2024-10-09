import argparse
from tqdm import tqdm

import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import torch
from torch.nn import MSELoss
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import optuna

from GNN.src.gnn_modular import Modular_GCN
import GNN.src.gnn_multiple as GCNs
from GNN.src import test_acc


def parse_args(args=None):
    """ 
    Perform command-line argument parsing (other otherwise parse arguments with defaults). 
    To parse in an interative context (i.e. in notebook), add required arguments.
    These will go into args and will generate a list that can be passed in.
    For example: 
        parse_args('--type', 'rnn', ...)
    """
    parser = argparse.ArgumentParser(description="Specify Hyperparameters for Grid Searching the hyperparameters of the GNN", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data',           required=True,              help='File path to the assignment data file.')
    parser.add_argument('--pred',           required=True,              choices=['TER', 'VEGF', 'Both'],    help='Type of Value being Predicted from QBAMs')
    parser.add_argument('--log_path',       default='',                 help='where the optuna study logs will stored')
    parser.add_argument('--batch_size',     type=int,   default=20,     help='Model\'s batch size.')

    if args is None: 
        return parser.parse_args()      ## For calling through command line
    return parser.parse_args(args)      ## For calling through notebook.


def train(model, train_loader, optimizer, criterion):
    model.train()
    for data in train_loader:
        data = data.to(model.device)  # Move data to the same device as the model
        out = model(data)
        loss = criterion(out, data.y.reshape(-1, model.output_dim))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def test(model, loader, criterion, print_met=False):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for data in tqdm(loader, desc="Testing", leave=False):
            data = data.to(model.device)
            out = model(data)
            loss = criterion(out, data.y.reshape(-1, model.output_dim))
            total_loss += loss.item()

            if print_met:
                print(f"Predicted: {out}, True: {data.y.reshape(-1, model.output_dim)}, RMSE: {math.sqrt(loss.item())}")

    avg_loss = total_loss / len(loader.dataset)
    return math.sqrt(avg_loss)

def train_model(train_loader, val_loader, test_loader, model, learning_rate, num_epochs, output_filepath = None, img_path = None, convergence_epsilon = 0.05, gamma=0.95):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using", device)
    model = model.to(device)
    model.device = device
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
    criterion = MSELoss()

    train_losses = []
    val_losses = []
    test_losses = []

    for epoch in tqdm(range(1, num_epochs + 1), desc="Training Epochs"):
        train(model, train_loader, optimizer, criterion)
        train_rmse = test(model, train_loader, criterion, False)
        val_rmse = test(model, val_loader, criterion, False)
        test_rmse = test(model, test_loader, criterion, False)

        train_losses.append(train_rmse)
        val_losses.append(val_rmse)
        test_losses.append(test_rmse)

        if len(train_losses) > 4:
            last_3 = np.array(train_losses)[:-3]
            prev = np.array(train_losses)[-1:-4]
            avg_loss_diff = np.mean(np.abs(last_3 - prev))
            if avg_loss_diff < convergence_epsilon:
                print(f"Stopping early on epoch {epoch} with average changes in loss {avg_loss_diff}")
                break

        if epoch % 3 == 0:
            scheduler.step()
            if epoch % 20 == 0:
                print(f'\nEpoch: {epoch:03d}, Train RMSE: {train_rmse:.4f}, Val RMSE: {val_rmse:.4f}\n')

    train_losses = np.array(train_losses)
    val_losses = np.array(val_losses)
    test_losses = np.array(test_losses)

    if output_filepath:
        torch.save(model.state_dict(), output_filepath)
        print("Saved the model to:", output_filepath)

    if img_path:
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training RMSE')
        plt.plot(val_losses, label='Validation RMSE')
        plt.plot(test_losses, label='Test RMSE')
        plt.xlabel('Epoch')
        plt.ylabel('RMSE')
        plt.title('Training and Validation RMSE')
        plt.legend()
        plt.savefig(img_path)
        plt.close()
        print(f"Saved graph to {img_path}")

    return train_losses[-1], val_losses[-1]


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


def objective(trial, target, model_constructors, num_features, num_targets, train_dataset, val_dataset, test_dataset, args):
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset)

    num_epochs = 100

    #Tuning
    num_gcn = trial.suggest_int("num_gcn", 2, 5)
    num_dense = trial.suggest_int("num_dense", 2, 5)
    arch_string = f"G{num_gcn}_D{num_dense}"
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-3, step=0.00025)

    model_class = model_constructors[arch_string]
    model = model_class(num_features, num_targets)

    train_loss, val_loss = train_model(train_loader, val_loader, test_loader, model, learning_rate, num_epochs)

    test_loss = test_acc.test_model(test_loader, model, task=target)

    return test_loss


def build_data(target, data_dirs):
    train_pickle_file = data_dirs[f"Train_{target}"]
    val_pickle_file = data_dirs[f"Valid_{target}"]
    test_pickle_file = data_dirs[f"Test_{target}"]

    train_dataset = load_dataset_from_pickle(train_pickle_file)
    val_dataset = load_dataset_from_pickle(val_pickle_file)
    test_dataset = load_dataset_from_pickle(test_pickle_file)

    check_for_nan(train_dataset)
    check_for_nan(test_dataset)

    return {"train_dataset": train_dataset, "val_dataset": val_dataset, "test_dataset": test_dataset}


def main(args):
    # arg_dict = {"target": args.pred, "batch_size": args.batch_size, }
    target = args.pred

    data_dirs = {}
    for data_type in ['TER', 'VEGF', 'Both']:
        data_dirs[f"Train_{data_type}"] = f"{args.data}/{data_type}/Train_{data_type}.pkl"
        data_dirs[f"Valid_{data_type}"] = f"{args.data}/{data_type}/Valid_{data_type}.pkl"
        data_dirs[f"Test_{data_type}"] = f"{args.data}/{data_type}/Test_{data_type}.pkl"

    model_constructors = {
        "G2_D2": GCNs.GCN_G2_D2,
        "G2_D3": GCNs.GCN_G2_D3,
        "G2_D4": GCNs.GCN_G2_D4,
        "G2_D5": GCNs.GCN_G2_D5,
        "G3_D2": GCNs.GCN_G3_D2,
        "G3_D3": GCNs.GCN_G3_D3,
        "G3_D4": GCNs.GCN_G3_D4,
        "G3_D5": GCNs.GCN_G3_D5,
        "G4_D2": GCNs.GCN_G4_D2,
        "G4_D3": GCNs.GCN_G4_D3,
        "G4_D4": GCNs.GCN_G4_D4,
        "G4_D5": GCNs.GCN_G4_D5,
        "G5_D2": GCNs.GCN_G5_D2,
        "G5_D3": GCNs.GCN_G5_D3,
        "G5_D4": GCNs.GCN_G5_D4,
        "G5_D5": GCNs.GCN_G5_D5
    }
    
    datasets = build_data(target, data_dirs)

    num_features = datasets["train_dataset"][0].x.shape[1]  # Number of features per node
    num_targets = datasets["train_dataset"][0].y.shape[0]

    data_details = {"num_features": num_features, "num_targets": num_targets}

    Path(f'{args.log_path}').mkdir(parents=True, exist_ok=True)
    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(f"{args.log_path}/optuna_journal_storage.log")
    )

    study = optuna.create_study(storage = storage)
    study.optimize(lambda trial: objective(trial, target, model_constructors, **data_details, **datasets), n_trials=100, n_jobs=-1)

    print(f"Best value: {study.best_value} (params: {study.best_params})")


## END UTILITY METHODS
##############################################################################

if __name__ == '__main__':
    main(parse_args())