import argparse
import math
import os
from pathlib import Path
import datetime

from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import torch
from torch.nn import MSELoss
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import optuna

from preprocessing import get_loaders
from train_test import train, test
import GNN.src.gnn_multiple as GCNs
from GNN.src.gnn_modular import Modular_GCN
from GNN.src import test_acc


def parse_args(args=None):
    """ 
    Perform command-line argument parsing (other otherwise parse arguments with defaults). 
    To parse in an interative context (i.e. in notebook), add required arguments.
    These will go into args and will generate a list that can be passed in.
    For example: 
        parse_args('--type', 'rnn', ...)
    """
    parser = argparse.ArgumentParser(description="Specify Hyperparameters to Optimize for the GNN", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data',           required=True,                                          help='File path to the assignment data file.')
    parser.add_argument('--pred',           required=True,  choices=['TER', 'VEGF', 'Both'],        help='Type of Value being Predicted from QBAMs')
    parser.add_argument('--log_path',       default='',                                             help='where the optuna study logs will stored')
    parser.add_argument('--batch_size',     type=int,       default=20,                             help='Model\'s batch size.')
    parser.add_argument('--normed',         required=False, action='store_true',                    help='Whether or not to use normalized label values')

    if args is None: 
        return parser.parse_args()      ## For calling through command line
    return parser.parse_args(args)      ## For calling through notebook.

def train_model(train_loader, val_loader, test_loader, model, learning_rate, num_epochs, output_filepath = None, img_path = None, convergence_epsilon = 0.5, gamma=0.95):
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
        scheduler.step()

        train_rmse = test(model, train_loader, criterion)
        val_rmse = test(model, val_loader, criterion)
        test_rmse = test(model, test_loader, criterion)

        train_losses.append(train_rmse)
        val_losses.append(val_rmse)
        test_losses.append(test_rmse)

        if len(train_losses) > 4:
            last_3 = np.array(train_losses)[:-4:-1]
            prev = np.array(train_losses)[-2:-5:-1]
            avg_loss_diff = np.mean(np.abs(last_3 - prev))
            if avg_loss_diff < convergence_epsilon:
                print(f"Stopping early on epoch {epoch} with average changes in loss {avg_loss_diff}")
                break

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

def objective(trial, target, model_constructors, data_details, train_loader, val_loader, test_loader):
    num_epochs = 100

    #Tuning
    num_gcn = trial.suggest_int("num_gcn", 2, 5)
    num_dense = trial.suggest_int("num_dense", 2, 5)
    hidden_size = trial.suggest_int("hidden_size", 32, 512, step=16)
    arch_string = f"G{num_gcn}_D{num_dense}"
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-3, step=5e-5)

    model_class = model_constructors[arch_string]
    model = model_class(*data_details, hidden_channels = hidden_size)

    train_loss, val_loss = train_model(train_loader, val_loader, test_loader, model, learning_rate, num_epochs)

    test_loss = test_acc.test_model(test_loader, model, task=target)

    return test_loss

# def objective(trial, target, model_class, data_details, train_loader, val_loader, test_loader):
#     num_epochs = 100

#     #Tuning
#     num_gcn = trial.suggest_int("num_gcn", 2, 5)
#     num_dense = trial.suggest_int("num_dense", 2, 5)
#     arch_string = f"G{num_gcn}_D{num_dense}"
#     learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-3, step=5e-5)

#     model = model_class(*data_details, num_dense, num_gcn)

#     train_loss, val_loss = train_model(train_loader, val_loader, test_loader, model, learning_rate, num_epochs)

#     test_loss = test_acc.test_model(test_loader, model, task=target)

#     return test_loss


def main(args):
    # arg_dict = {"target": args.pred, "batch_size": args.batch_size, }
    target = args.pred
    print(f"Optuna Searching {target}")

    norm_string = "_normalized" if args.normed else ""

    data_dirs = {}
    for data_type in ['TER', 'VEGF', 'Both', 'Donor']:
        data_dirs[f"Train_{data_type}"] = f"{args.data}/{data_type}/Train_{data_type}{norm_string}.pkl"
        data_dirs[f"Valid_{data_type}"] = f"{args.data}/{data_type}/Valid_{data_type}{norm_string}.pkl"
        data_dirs[f"Test_{data_type}"] = f"{args.data}/{data_type}/Test_{data_type}{norm_string}.pkl"

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
    
    train_loader, val_loader, test_loader, data_details = get_loaders(data_dirs, target, args.batch_size)


    Path(f'{args.log_path}').mkdir(parents=True, exist_ok=True)
    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(f"{args.log_path}/optuna_journal_storage_modular.log")
    )

    time_string = datetime.datetime.now().strftime('%d-%b-%Y-%H%M')

    study = optuna.create_study(study_name=f"{time_string}_optimize_{args.pred}{norm_string}_modular",storage = storage, direction="minimize")
    study.set_metric_names(["RMSE"])
    study.optimize(lambda trial: objective(trial, target, model_constructors, data_details, train_loader, val_loader, test_loader), n_trials=100)
    # study.optimize(lambda trial: objective(trial, target, Modular_GCN, data_details, train_loader, val_loader, test_loader), n_trials=100)

    print(f"Best value: {study.best_value} (params: {study.best_params})")


## END UTILITY METHODS
##############################################################################

if __name__ == '__main__':
    main(parse_args())