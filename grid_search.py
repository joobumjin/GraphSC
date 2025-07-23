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

from preprocessing import get_loaders
from train_test import train, test
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
    parser.add_argument('--chkpt_path',     default='',                 help='where the model checkpoint is')
    parser.add_argument('--img_path',       default='',                 help='where the model saves loss graphs')
    parser.add_argument('--results_path',   default='',                 help='where the model saves text files with test predictions')
    parser.add_argument('--batch_size',     type=int,   default=20,     help='Model\'s batch size.')

    if args is None: 
        return parser.parse_args()      ## For calling through command line
    return parser.parse_args(args)      ## For calling through notebook.

def plot_losses(loss_dict, img_path):
    plt.figure(figsize=(10, 6))
    for loss_label, loss_list in loss_dict:
        plt.plot(np.array(loss_list), label=f"{loss_label}")
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.title('Training, Validation, and Testing RMSE')
    plt.legend()
    plt.savefig(img_path)
    plt.close()
    print(f"Saved graph to {img_path}")

def export_stats_excel(dir, target, train_data, val_data, test_data):
    df_filepath = f"{dir}/{target}_stats.xlsx"
    df_file = Path(df_filepath)

    kwargs = {"mode": "a", "if_sheet_exists": "new"} if df_file.exists() and target != "TER" else {}

    with pd.ExcelWriter(df_filepath, engine="openpyxl", **kwargs) as writer:
        start_row = 1
        for data, split in zip([train_data, val_data, test_data], ["Train", "Val", "Test"]):
            pd.DataFrame([split]).to_excel(writer, sheet_name=target, startrow=start_row-1, startcol=0, header=False, index=False)
            df = pd.DataFrame(data)
            df.to_excel(writer, sheet_name=target, startrow=start_row, startcol=0, index=False)
            start_row += len(data["Learning Rate"]) + 5

    print(f"Wrote performance summary to {df_filepath}")

def train_model(train_loader, val_loader, test_loader, model, output_filepath, img_path, learning_rate, num_epochs, convergence_epsilon = 0.5, gamma=0.95):
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

    torch.save(model.state_dict(), output_filepath)
    print("Saved the model to:", output_filepath)

    if img_path:
        loss_dict = {"Training Loss": train_losses, "Validation Loss": val_losses, "Testing Loss": test_losses}
        plot_losses(loss_dict, img_path)

    return train_losses[-1], val_losses[-1]



def main(args):
    data_dirs = {}
    for data_type in ['TER', 'VEGF', 'Both']:
        data_dirs[f"Train_{data_type}"] = f"{args.data}/{data_type}/Train_{data_type}.pkl"
        data_dirs[f"Valid_{data_type}"] = f"{args.data}/{data_type}/Valid_{data_type}.pkl"
        data_dirs[f"Test_{data_type}"] = f"{args.data}/{data_type}/Test_{data_type}.pkl"

    model_constructors = {
        "G2_D2": GCNs.GCN_G2_D2, "G2_D3": GCNs.GCN_G2_D3, "G2_D4": GCNs.GCN_G2_D4, "G2_D5": GCNs.GCN_G2_D5,
        "G3_D2": GCNs.GCN_G3_D2, "G3_D3": GCNs.GCN_G3_D3, "G3_D4": GCNs.GCN_G3_D4, "G3_D5": GCNs.GCN_G3_D5,
        "G4_D2": GCNs.GCN_G4_D2, "G4_D3": GCNs.GCN_G4_D3, "G4_D4": GCNs.GCN_G4_D4, "G4_D5": GCNs.GCN_G4_D5,
        "G5_D2": GCNs.GCN_G5_D2, "G5_D3": GCNs.GCN_G5_D3, "G5_D4": GCNs.GCN_G5_D4, "G5_D5": GCNs.GCN_G5_D5
    }

    target = args.pred

    train_loader, val_loader, test_loader, data_details = get_loaders(data_dirs, target, args.batch_size)

    lr_epoch = [(0.0005, 150), (0.00075, 150), (0.001, 50), (0.0025, 50), (0.005, 50)]

    train_data = {"Learning Rate": [lr for lr, _ in lr_epoch]}
    val_data = {"Learning Rate": [lr for lr, _ in lr_epoch]}
    test_data = {"Learning Rate": [lr for lr, _ in lr_epoch]}

    #Tuning Modular
    for num_gcn in ([2, 3, 4, 5]):
        for num_dense in ([2, 3, 4, 5]):
            arch_string = f"G{num_gcn}_D{num_dense}"
            train_losses, val_losses, test_losses = [], [], []
            for (learning_rate, num_epochs) in lr_epoch:
                print("___________________________________")
                print()
                print("Learning Rate:", learning_rate)
                print("Epochs:", num_epochs )
                print(f"Num GCN Layers {num_gcn}")
                print(f"Num Dense Layers {num_dense}")
                print("___________________________________")

                hyper_param_dir = f"{args.pred}/lr{learning_rate}_e{num_epochs}" 

                Path(f'{args.chkpt_path}/{hyper_param_dir}').mkdir(parents=True, exist_ok=True)
                Path(f'{args.img_path}/{hyper_param_dir}').mkdir(parents=True, exist_ok=True)
                Path(f'{args.results_path}/{hyper_param_dir}').mkdir(parents=True, exist_ok=True)
                
                output_filepath = f'{args.chkpt_path}/{hyper_param_dir}/g{num_gcn}_d{num_dense}_Abs_model.pth'
                img_path = f"{args.img_path}/{hyper_param_dir}/g{num_gcn}_d{num_dense}_RMSE_Loss_Graph.jpg"
                results_file = f'{args.results_path}/{hyper_param_dir}/g{num_gcn}_d{num_dense}_sample_preds.txt'
                plotted_preds = f'{args.results_path}/{hyper_param_dir}/g{num_gcn}_d{num_dense}_preds_graph'

                model_class = model_constructors[arch_string]
                model = model_class(*data_details)

                train_loss, val_loss = train_model(train_loader, val_loader, test_loader, model, output_filepath, img_path, learning_rate, num_epochs)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                
                test_loss = test_acc.test_model(test_loader, model, task=args.pred, write_to_file=results_file, vis_preds=plotted_preds)
                test_losses.append(test_loss)

            train_data[arch_string] = train_losses
            val_data[arch_string] = val_losses
            test_data[arch_string] = test_losses

    #Create performance summaries
    export_stats_excel(args.results_path, target, train_data, val_data, test_data)

## END UTILITY METHODS
##############################################################################

if __name__ == '__main__':
    main(parse_args())