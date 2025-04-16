import argparse
from pathlib import Path
import datetime

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import MSELoss
import optuna

from preprocessing import get_loaders
from train_test import train, train_multidata, test, test_multidata, SSLELoss, StandardInlinePrint
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
    parser = argparse.ArgumentParser(description="Specify Hyperparameters to Optimize for the GNN", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data',           required=True,                                          help='File path to the assignment data file.')
    parser.add_argument('--pred',           required=True,  choices=['TER', 'VEGF', 'Both'],        help='Type of Value being Predicted from QBAMs')
    parser.add_argument('--log_path',       default='',                                             help='where the optuna study logs will stored')
    parser.add_argument('--batch_size',     type=int,       default=20,                             help='Model\'s batch size.')

    if args is None: 
        return parser.parse_args()      ## For calling through command line
    return parser.parse_args(args)      ## For calling through notebook.

def train_model(train_loaders, val_loaders, model, learning_rate, num_epochs, output_filepath = None, img_path = None, convergence_epsilon = None, gamma=0.95, weight_decay = None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using", device)

    model = model.to(device)
    model.device = device
    opt_args = {name: arg for (arg, name) in zip([learning_rate, weight_decay], ["lr", "weight_decay"]) if arg is not None}
    optimizer = torch.optim.Adam(model.parameters(), **opt_args)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
    log_train = True
    train_criterion = SSLELoss() if log_train else MSELoss(reduction='sum')
    test_criterion = MSELoss(reduction='sum')
    if len(train_loaders) > 1: train_fn = train_multidata
    else: 
        train_fn = train
        train_loaders = train_loaders[0]

    if len(val_loaders) > 1: test_fn = test_multidata
    else: 
        test_fn = test
        val_loaders = val_loaders[0]

    train_losses = []
    val_losses = []

    epoch_tqdm = tqdm(range(1, num_epochs + 1), desc="Training Epochs", postfix={"Train RMSE": 0.0, "Valid RMSE": 0.0})
    
    for epoch in epoch_tqdm:
        printer = StandardInlinePrint() if epoch == num_epochs else None
        train_rmse = train_fn(model, train_loaders, optimizer, train_criterion, metric_printer=printer)
        scheduler.step()

        val_rmse = test_fn(model, val_loaders, test_criterion, metric_printer=printer)

        train_losses.append(train_rmse)
        val_losses.append(val_rmse)

        epoch_tqdm.set_postfix({"Train RMSE": train_rmse, "Valid RMSE": val_rmse})

        # if convergence_epsilon is not None:
        #     if len(train_losses) > 4:
        #         last_3 = np.array(train_losses)[:-4:-1]
        #         prev = np.array(train_losses)[-2:-5:-1]
        #         avg_loss_diff = np.mean(np.abs(last_3 - prev))
        #         if avg_loss_diff < convergence_epsilon:
        #             print(f"Stopping early on epoch {epoch} with average changes in loss {avg_loss_diff}")
        #             break

    epoch_tqdm.close()

    train_losses = np.array(train_losses)
    val_losses = np.array(val_losses)

    if output_filepath:
        torch.save(model.state_dict(), output_filepath)
        print("Saved the model to:", output_filepath)

    if img_path:
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training RMSE')
        plt.plot(val_losses, label='Validation RMSE')
        plt.xlabel('Epoch')
        plt.ylabel('RMSE')
        plt.title('Training and Validation RMSE')
        plt.legend()
        plt.savefig(img_path)
        plt.close()
        print(f"Saved graph to {img_path}")

    return train_losses[-1], val_losses[-1]

def objective(trial, target, model_constructors, data_details, train_loaders, val_loaders, test_loaders, data_path = None):
    num_epochs = 300

    #Tuning
    num_gcn = trial.suggest_int("num_gcn", 3, 5)
    num_dense = trial.suggest_int("num_dense", 3, 5)
    hidden_size = trial.suggest_int("hidden_size", 1, 200, step=16)
    dense_hidden = trial.suggest_int("dense_hidden", 1, 600, step=32)
    arch_string = f"G{num_gcn}_D{num_dense}"
    learning_rate = trial.suggest_float("learning_rate", 1e-7, 5e-3, step=5e-7)
    lr_decay = trial.suggest_float("learning_rate_decay", 0.5, 1.0, step=.05)
    weight_decay = trial.suggest_float("l2_penalty", 0, 1e-2, step=5e-5)
    dropout_rate = trial.suggest_float("dropout", 0, 0.5, step=0.1)

    model_class = model_constructors[arch_string]
    model = model_class(*data_details, hidden_channels = hidden_size, dense_hidden = dense_hidden, dropout_p=dropout_rate)

    print(f"{num_gcn} GCN Layers | {hidden_size} units\n{num_dense} Dense Layers | {dense_hidden}\nDropout Rate: {dropout_rate}\nLearning Rate: {learning_rate} with Decay {lr_decay} and Weight Decay: {weight_decay}")

    _, _ = train_model(train_loaders, val_loaders, model, learning_rate, num_epochs, img_path=f"{data_path}/Train_graphs/{arch_string}_h{hidden_size}_d{dense_hidden}_lr{learning_rate}_decay{lr_decay}.jpeg", gamma=lr_decay, weight_decay=weight_decay)

    test_loss = test_acc.test_model(test_loaders, model, task=target, test_multiple=True)

    return test_loss


def main(args):
    target = args.pred
    print(f"Optuna Searching {target}")

    data_dirs = {}
    for data_type in ['TER', 'VEGF', 'Both', 'Donor']:
        data_dirs[f"Train_{data_type}"] = f"{args.data}/{data_type}/Train_{data_type}.pkl"
        data_dirs[f"Valid_{data_type}"] = f"{args.data}/{data_type}/Valid_{data_type}.pkl"
        data_dirs[f"Test_{data_type}"] = f"{args.data}/{data_type}/Test_{data_type}.pkl"

    model_constructors = GCNs.get_model_constructors()

    train_loader, val_loader, test_loader, data_details = get_loaders(data_dirs, target, args.batch_size)
    train_loaders = [train_loader]
    val_loaders = [val_loader]
    test_loaders = [test_loader]


    Path(f'{args.log_path}').mkdir(parents=True, exist_ok=True)
    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(f"{args.log_path}/optuna_journal_storage_diff_rmse.log")
    )

    time_string = datetime.datetime.now().strftime('%d-%b-%Y-%H%M')

    study = optuna.create_study(study_name=f"{time_string}_optimize_{args.pred}",storage = storage, direction="minimize")
    study.set_metric_names(["RMSE"])
    study.optimize(lambda trial: objective(trial, target, model_constructors, data_details, train_loaders, val_loaders, test_loaders, data_path = args.data))

    print(f"Best value: {study.best_value} (params: {study.best_params})")


## END UTILITY METHODS
##############################################################################

if __name__ == '__main__':
    main(parse_args())