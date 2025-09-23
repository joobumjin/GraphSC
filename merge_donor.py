import argparse
import datetime
import wandb
import optuna

from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from torch.nn import BCEWithLogitsLoss

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GraphConv, GCNConv, GATConv, GATv2Conv
from preprocessing.preprocessing import get_loaders
from utils.train_test import Accuracy, train, train_multidata, test, test_multidata
from GNN.src.gnn_modular import Modular_GNN
from GNN.src.gnn_merge import GNN_Merge

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
    parser.add_argument('--batch_size',     type=int,       default=20,                             help='Model\'s batch size.')

    if args is None: 
        return parser.parse_args()      ## For calling through command line
    return parser.parse_args(args)      ## For calling through notebook.

def train_model(train_loaders, val_loaders, model, opt_args, num_epochs, output_filepath = None, gamma=0.95, wandb_run = None, trial = None, pruning = False):
    #setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using", device)

    model = model.to(device)
    model.device = device

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

    #data
    if len(train_loaders) > 1: train_fn = train_multidata
    else: 
        train_fn = train
        train_loaders = train_loaders[0]

    if len(val_loaders) > 1: test_fn = test_multidata
    else: 
        test_fn = test
        val_loaders = val_loaders[0]

    #run
    epoch_tqdm = tqdm(range(1, num_epochs + 1), desc="Training Epochs", postfix={f"Train {crit_string}": 0.0, "Train Acc": 0.0, f"Valid {crit_string}": 0.0, f"Valid Acc": 0.0})
    
    for epoch in epoch_tqdm:
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

        if epoch % 30 == 0 and trial and pruning: 
            trial.report(postfix["Valid Acc"], epoch)
            if trial.should_prune(): return postfix["Train BCE"], postfix["Valid BCE"], True

    epoch_tqdm.close()

    #output formating
    train_losses = np.array(train_losses)
    train_metrics = {crit: np.array(history) for crit, history in train_metrics.items()}
    val_metrics = {crit: np.array(history) for crit, history in val_metrics.items()}

    #model saving
    if output_filepath:
        torch.save(model.state_dict(), output_filepath)
        print("Saved the model to:", output_filepath)

    #plotting
    # losses
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('BCE')
    p1 = ax1.plot(train_losses, label='Training BCE', color='tab:blue')
    p2 = ax1.plot(val_metrics["BCE"], label='Validation BCE', color="tab:orange")
    ax1.tick_params(axis='y')
    #accs
    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy')
    p3 = ax2.plot(train_metrics["Acc"], label='Train Accuracy', color="tab:green")
    p4 = ax2.plot(val_metrics["Acc"], label='Validation Accuracy', color="tab:red")

    ax1.legend(handles=p1+p2+p3+p4, loc='best')

    #outoutting
    fig.tight_layout()  
    plt.title('Training and Validation BCE/Acc')
    plt.legend()

    if wandb_run: wandb_run.log({"chart": plt})

    plt.close()

    return train_losses[-1], val_metrics["BCE"][-1], False

def eval(test_loaders, model, wandb_run = None):
    #setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using", device)

    model = model.to(device)
    model.device = device

    acc = 0
    test_crits = {
        "BCE": BCEWithLogitsLoss(reduction='sum'),
        "Acc": Accuracy()
    }

    #data
    if len(test_loaders) > 1: test_fn = test_multidata
    else: 
        test_fn = test
        test_loaders = test_loaders[0]

    #evaluate
    for crit_dict, loader, split in zip([test_crits], [test_loaders], ["Test"]):
        for crit, crit_obj in crit_dict.items():
            metric_calc = test_fn(model, loader, crit_obj)
            if wandb_run: wandb_run.summary[f"{split} {crit}"] = metric_calc
            if crit == "Acc": acc = metric_calc

    return acc

##############################################################################

def objective(trial, data_details, train_loaders, val_loaders, test_loaders, layer_dict):
    #
    #hyper params
    layer = trial.suggest_categorical("layer type", layer_dict.keys())

    model_args = {
        "num_node_features": data_details[0], 
        "output_dim": data_details[1],  
        "num_gcn": trial.suggest_int("num_gcn", 2, 3),
        "num_dense": trial.suggest_int("num_dense", 3, 5), 
        "hidden_channels": trial.suggest_int("hidden_size", 64, 256, step=64), 
        "dense_hidden": trial.suggest_int("hidden_size", 64, 256, step=64), 
        "dropout_p": trial.suggest_float("dropout", 0.1, 0.7, step=0.1),
        "gnn_layer": layer_dict[layer]
    }

    opt_args = {
        "lr": trial.suggest_float("learning_rate", 0.0001, 0.005, step=0.0001),
        "weight_decay": trial.suggest_float("l2_penalty", 0, 1e-2, step=5e-5),
    }

    config={
        "graph layer": f"{layer}",
        "epochs": 75,
        "lr_decay": trial.suggest_float("learning_rate_decay", 0.7, 1.0, step=.1),
    }

    config = {**model_args, **opt_args, **config}
    print(f"###############################################################################\n"
            f"{model_args["num_gcn"]} {layer} Layers\t| {model_args["hidden_channels"]} units\n"
            f"{model_args["num_dense"]} Dense Layers\t| {model_args["dense_hidden"]} units\n"
            f"Dropout Rate: {model_args["dropout_p"]}\n"
            f"Learning Rate: {config["lr"]} with Decay {config["lr_decay"]} and Weight Decay: {config["weight_decay"]}\n"
            f"###############################################################################")

    #
    #build models
    gnn1 = Modular_GNN(**model_args)
    gnn2 = Modular_GNN(**model_args)
    model = GNN_Merge(gnn1, gnn2)
    #actually build the weights to get grad tracking
    dummy_batch = next(iter(val_loaders[0]))
    _ = model(dummy_batch)


    #
    #run
    run = wandb.init(
        entity="bumjin_joo-brown-university", 
        project="qbam-donor-optuna", 
        name=f"Singular {layer}, LR{config["learning_rate"]:.5f}", 
        config=config
    )

    _, _, should_prune = train_model(train_loaders, 
                    val_loaders, 
                    model, 
                    opt_args = opt_args,
                    num_epochs=config["epochs"], 
                    gamma=config["lr_decay"], 
                    wandb_run = run,
                    trial = trial,
                    pruning = True)
    
    if should_prune:
        run.summary["state"] = "pruned"
        wandb.finish(quiet=True)
        raise optuna.TrialPruned()

    test_acc = eval(test_loaders, model, wandb_run = run)

    run.summary["final accuracy"] = test_acc
    run.summary["state"] = "completed"
    wandb.finish(quiet=True)

    return test_acc

##############################################################################

def main(args):
    sns.set_theme()

    #
    # get data
    target = args.pred
    print(f"Training {target}")

    data_dirs = {}
    for data_type in ['TER', 'VEGF', 'Both', 'Donor']:
        data_dirs[f"Train_{data_type}"] = f"{args.data}/{data_type}/train_singular_donors.pkl"
        data_dirs[f"Valid_{data_type}"] = f"{args.data}/{data_type}/valid_singular_donors.pkl"
        data_dirs[f"Test_{data_type}"] = f"{args.data}/{data_type}/test_singular_donors.pkl"

    layer_dict = {"Graph": GraphConv, "GCN": GCNConv, "GAT": GATConv, "GATv2": GATv2Conv}

    train_loader, val_loader, test_loader, data_details = get_loaders(data_dirs, target, args.batch_size)
    train_loaders = [train_loader]
    val_loaders = [val_loader]
    test_loaders = [test_loader]

    time_string = datetime.datetime.now().strftime('%d-%b-%Y-%H%M')
    study = optuna.create_study(study_name=f"{time_string}_optimize_{args.pred}", direction="maximize") #direction=["maximize,minimize"]
    study.set_metric_names(["Test Acc"]) #["Test Acc", "Test BCE"]

    study.optimize(lambda trial: objective(trial, data_details, train_loaders, val_loaders, test_loaders, layer_dict), n_trials=200)

    print(f"Best value: {study.best_value} (params: {study.best_params})")


## END UTILITY METHODS
##############################################################################

if __name__ == '__main__':
    main(parse_args())