import argparse
from pathlib import Path
import datetime
import wandb

from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from torch.nn import BCEWithLogitsLoss

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from preprocessing import get_loaders
from train_test import Accuracy, train, train_multidata, test, test_multidata
# import GNN.src.gnn_multiple as GCNs
from GNN.src.gnn_modular import Modular_GNN
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
    parser.add_argument('--pred',           required=True,  choices=['Donor'],                      help='Type of Value being Predicted from QBAMs')
    # parser.add_argument('--study_name',     required=True,                                          help='Name of the log to which the Optuna study shall be saved')
    # parser.add_argument('--log_path',       default='',                                             help='where the optuna study logs will stored')
    parser.add_argument('--batch_size',     type=int,       default=20,                             help='Model\'s batch size.')

    if args is None: 
        return parser.parse_args()      ## For calling through command line
    return parser.parse_args(args)      ## For calling through notebook.

def train_model(train_loaders, val_loaders, model, learning_rate, num_epochs, output_filepath = None, img_path = None, gamma=0.95, weight_decay = None, wandb_run = None):
    #
    #setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using", device)

    model = model.to(device)
    model.device = device
    opt_args = {name: arg for (arg, name) in zip([learning_rate, weight_decay], ["lr", "weight_decay"]) if arg is not None}
    optimizer = torch.optim.Adam(model.parameters(), **opt_args)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
    crit_string = "BCE"
    train_criterion = BCEWithLogitsLoss(reduction='sum')
    train_crits = {
        "Acc": Accuracy()
    }
    test_crits = {
        "BCE": BCEWithLogitsLoss(reduction='sum'),
        "Acc": Accuracy()
    }
    
    train_losses = []
    train_metrics = {crit: [] for crit in train_crits}
    val_metrics = {crit: [] for crit in test_crits}

    #
    #data
    if len(train_loaders) > 1: train_fn = train_multidata
    else: 
        train_fn = train
        train_loaders = train_loaders[0]

    if len(val_loaders) > 1: test_fn = test_multidata
    else: 
        test_fn = test
        val_loaders = val_loaders[0]

    #
    #run
    epoch_tqdm = tqdm(range(1, num_epochs + 1), desc="Training Epochs", postfix={f"Train {crit_string}": 0.0, "Train Acc": 0.0, f"Valid {crit_string}": 0.0, f"Valid Acc": 0.0})
    
    for _ in epoch_tqdm:
        #train
        train_loss = train_fn(model, train_loaders, optimizer, train_criterion)
        scheduler.step()

        train_losses.append(train_loss)
        postfix = {f"Train {crit_string}": train_loss}

        #eval
        for crit_dict, metric_dict, loader, split in zip([train_crits, test_crits], [train_metrics, val_metrics], [train_loaders, val_loaders], ["Train", "Valid"]):
            for crit, crit_obj in crit_dict.items():
                metric = test_fn(model, loader, crit_obj)
                metric_dict[crit].append(metric)
                postfix[f"{split} {crit}"] = metric

        if wandb_run: wandb_run.log(postfix)
        epoch_tqdm.set_postfix(postfix)

    epoch_tqdm.close()

    #
    #output formating
    train_losses = np.array(train_losses)
    train_metrics = {crit: np.array(history) for crit, history in train_metrics.items()}
    val_metrics = {crit: np.array(history) for crit, history in val_metrics.items()}

    #
    #model saving
    if output_filepath:
        torch.save(model.state_dict(), output_filepath)
        print("Saved the model to:", output_filepath)

    #
    #plotting
    # losses
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('BCE')
    p1 = ax1.plot(train_losses, label='Training BCE', color='tab:blue')
    p2 = ax1.plot(val_metrics["BCE"], label='Validation BCE', color="tab:orange")
    # ax1.legend()
    ax1.tick_params(axis='y')
    #accs
    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy')
    p3 = ax2.plot(train_metrics["Acc"], label='Train Accuracy', color="tab:green")
    p4 = ax2.plot(val_metrics["Acc"], label='Validation Accuracy', color="tab:red")
    # ax2.legend()

    ax1.legend(handles=p1+p2+p3+p4, loc='best')

    #outoutting
    fig.tight_layout()  
    plt.title('Training and Validation BCE/Acc')
    plt.legend()

    if img_path:
        plt.savefig(img_path)
        print(f"Saved graph to {img_path}")

    if wandb_run: wandb_run.log({"chart": plt})

    plt.close()

    return train_losses[-1], val_metrics["BCE"][-1]

def eval(test_loaders, model, wandb_run = None):
    #
    #setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using", device)

    model = model.to(device)
    model.device = device

    test_crits = {
        "BCE": BCEWithLogitsLoss(reduction='sum'),
        "Acc": Accuracy()
    }

    #
    #data
    if len(test_loaders) > 1: test_fn = test_multidata
    else: 
        test_fn = test
        test_loaders = test_loaders[0]

    #
    #evaluate
    for crit_dict, loader, split in zip([test_crits], [test_loaders], ["Test"]):
        for crit, crit_obj in crit_dict.items():
            if wandb_run: 
                wandb_run.summary[f"{split} {crit}"] = test_fn(model, loader, crit_obj)

# def objective(trial, target, model_constructors, data_details, train_loaders, val_loaders, test_loaders, data_path = None):
#     num_epochs = 300

#     #Tuning
#     num_gcn = trial.suggest_int("num_gcn", 4, 5)
#     num_dense = trial.suggest_int("num_dense", 4, 5)
#     hidden_size = 144 # trial.suggest_int("hidden_size", 64, 200, step=16)
#     dense_hidden = trial.suggest_int("dense_hidden", 128, 512, step=32)
#     arch_string = f"G{num_gcn}_D{num_dense}"
#     learning_rate = trial.suggest_float("learning_rate", 0.001, 0.005, step=0.001)
#     lr_decay = trial.suggest_float("learning_rate_decay", 0.5, 1.0, step=.1)
#     weight_decay = 0.005 #trial.suggest_float("l2_penalty", 0, 1e-2, step=5e-5)
#     dropout_rate = trial.suggest_float("dropout", 0.2, 0.7, step=0.1)

#     model_class = model_constructors[arch_string]
#     model = model_class(*data_details, hidden_channels = hidden_size, dense_hidden = dense_hidden, dropout_p=dropout_rate)

#     print(f"{num_gcn} GCN Layers | {hidden_size} units\n{num_dense} Dense Layers | {dense_hidden}\nDropout Rate: {dropout_rate}\nLearning Rate: {learning_rate} with Decay {lr_decay} and Weight Decay: {weight_decay}")

#     _, _ = train_model(train_loaders, val_loaders, model, learning_rate, num_epochs, img_path=f"{data_path}/Train_graphs/{arch_string}_h{hidden_size}_d{dense_hidden}_lr{learning_rate}_decay{lr_decay}.jpeg", gamma=lr_decay, weight_decay=weight_decay)

#     print(f"Validation Stats")
#     _ = test_acc.test_model(val_loaders, model, task=target, test_multiple=False)
#     print(f"Test Stats")
#     test_loss = test_acc.test_model(test_loaders, model, task=target, test_multiple=False)

#     return test_loss

def main(args):
    sns.set_theme()

    #
    # get data
    #
    target = args.pred
    print(f"Training {target}")

    data_dirs = {}
    for data_type in ['TER', 'VEGF', 'Both', 'Donor']:
        data_dirs[f"Train_{data_type}"] = f"{args.data}/{data_type}/train_singular_donors.pkl"
        data_dirs[f"Valid_{data_type}"] = f"{args.data}/{data_type}/valid_singular_donors.pkl"
        data_dirs[f"Test_{data_type}"] = f"{args.data}/{data_type}/test_singular_donors.pkl"

    Path(f"{args.data}/{target}/Train_graphs").mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, data_details = get_loaders(data_dirs, target, args.batch_size)
    train_loaders = [train_loader]
    val_loaders = [val_loader]
    test_loaders = [test_loader]

    #
    #hyper params
    model_args = {
        "num_node_features": data_details[0], 
        "output_dim": data_details[1],  
        "num_gcn": 3, #max 2-3
        "num_dense": 4, 
        "hidden_channels": 256, 
        "dense_hidden": 512, 
        "dropout_p": 0.25
    }

    config={
        "architecture": "GATv2 Modular",
        "dataset": "Donor, Singular Graph",
        "epochs": 150,
        "learning_rate": 1e-8,
        "lr_decay": 0,
        "weight_decay": 0.005,
    }

    config = {**model_args, **config}

    #
    #build models
    model = Modular_GNN(**model_args)
    #actually build the weights to get grad tracking
    dummy_batch = next(iter(val_loader))
    _ = model(dummy_batch)

    print(f"###############################################################################\n"
            f"{model_args["num_gcn"]} GCN Layers\t| {model_args["hidden_channels"]} units\n"
            f"{model_args["num_dense"]} Dense Layers\t| {model_args["dense_hidden"]} units\n"
            f"Dropout Rate: {model_args["dropout_p"]}\n"
            f"Learning Rate: {config["learning_rate"]} with Decay {config["lr_decay"]} and Weight Decay: {config["weight_decay"]}\n"
            f"###############################################################################")


    #
    #run
    with wandb.init(
        entity="bumjin_joo-brown-university", 
        project="qbam-donor", 
        name=f"Modular Singular, LR{config["learning_rate"]}", 
        config=config) as run:
        # run.watch(model)

        _, _ = train_model(train_loaders, 
                        val_loaders, 
                        model, 
                        config["learning_rate"], 
                        config["epochs"], 
                        gamma=config["lr_decay"], 
                        weight_decay=config["weight_decay"],
                        wandb_run = run)
        
        eval(test_loaders, model, wandb_run = run)


## END UTILITY METHODS
##############################################################################

if __name__ == '__main__':
    main(parse_args())