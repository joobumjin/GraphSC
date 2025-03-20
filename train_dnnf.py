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
import optuna

from img_preprocessing import get_image_loaders, HealthyData
from train_test import train_multidata, test, MetricPrinter
from GNN.src.dnn_f import DNN_F
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
    parser.add_argument('--graph_path',     required=False,                                         help='Where to store the training graphs')
    parser.add_argument('--batch_size',     type=int,       default=20,                             help='Model\'s batch size.')
    parser.add_argument('--normed',         required=False, action='store_true',                    help='Whether or not to use normalized label values')
    parser.add_argument('--extra_data',     required=True,  default=None,                           help='File path to the assignment data file.')

    if args is None: 
        return parser.parse_args()      ## For calling through command line
    return parser.parse_args(args)      ## For calling through notebook.

def train_model(train_loaders, val_loader, test_loader, model, learning_rate, num_epochs, output_filepath = None, img_path = None, convergence_epsilon = 0.5, gamma=0.95):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using", device)
    model = model.to(device)
    model.device = device
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
    criterion = MSELoss()

    train_losses = []
    val_losses = []
    # test_losses = []

    for epoch in tqdm(range(1, num_epochs + 1), desc="Training Epochs"):
        train_rmse = train_multidata(model, train_loaders, optimizer, criterion)
        scheduler.step()

        val_rmse = test(model, val_loader, criterion)
        # test_rmse = test(model, test_loader, criterion)

        train_losses.append(train_rmse)
        val_losses.append(val_rmse)
        # test_losses.append(test_rmse)

        if len(train_losses) > 4:
            last_3 = np.array(train_losses)[:-4:-1]
            prev = np.array(train_losses)[-2:-5:-1]
            avg_loss_diff = np.mean(np.abs(last_3 - prev))
            if avg_loss_diff < convergence_epsilon:
                print(f"Stopping early on epoch {epoch} with average changes in loss {avg_loss_diff}")
                break

        if epoch % 20 == 0:
            # print(f'\rEpoch: {epoch:03d}, Train RMSE: {train_rmse:.4f}', end='')
            print(f'Epoch: {epoch:03d}, Train RMSE: {train_rmse:.4f}', end='\n')

    train_losses = np.array(train_losses)
    val_losses = np.array(val_losses)
    # test_losses = np.array(test_losses)

    if output_filepath:
        torch.save(model.state_dict(), output_filepath)
        print("Saved the model to:", output_filepath)

    if img_path:
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training RMSE')
        plt.plot(val_losses, label='Validation RMSE')
        # plt.plot(test_losses, label='Test RMSE')
        plt.xlabel('Epoch')
        plt.ylabel('RMSE')
        plt.title('Training and Validation RMSE')
        plt.legend()
        plt.savefig(img_path)
        plt.close()
        print(f"Saved graph to {img_path}")

    return train_losses[-1]

def main(args):
    target = args.pred
    print(f"Training DNN F on {target}")

    # norm_string = "_normalized" if args.normed else ""

    data_base_dir = f"{args.data}/full_imgs"
    # data_dirs = {"train": "train_samples.csv", "valid":"valid_samples.csv", "test": "test_samples.csv"}
    data_dirs = {"train": ["train_TER_imgs_0.pkl", "train_TER_imgs_1.pkl", "train_TER_imgs_2.pkl"], 
                 "valid": "valid_TER_imgs_0.pkl", 
                 "test":  "test_TER_imgs_0.pkl"}
 
    train_loaders, valid_loader, test_loader = get_image_loaders(data_base_dir, data_dirs, target, args.batch_size)

    out_dim = 2 if target=="Both" else 1

    model = DNN_F(out_dim)

    train_loss = train_model(train_loaders, valid_loader, test_loader, model, learning_rate=1e-3, num_epochs=200, img_path = args.graph_path)
    test_loss = test_acc.test_model(test_loader, model, task=target)

    print(f"Final Test Loss: {test_loss}")


## END UTILITY METHODS
##############################################################################

if __name__ == '__main__':
    main(parse_args())