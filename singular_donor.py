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
from utils import Accuracy, train_model, eval_model, get_test_criteria
from models import Modular_GNN, get_layer_dict

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

def graph_train_stats(train_losses, train_metrics, val_metrics, wandb_run):
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

##############################################################################

def objective(trial, data_details, train_loaders, val_loaders, test_loaders, layer_dict, args):
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
        "network": "singular",
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
    model = Modular_GNN(**model_args)
    #actually build the weights to get grad tracking
    dummy_batch = next(iter(val_loaders[0]))
    _ = model(dummy_batch)


    #
    #run
    run = wandb.init(
        entity="bumjin_joo-brown-university", 
        project="qbam-donor", 
        name=f"Singular {layer}, LR{config["lr"]:.5f}", 
        config=config
    )

    _, _, _, should_prune = train_model(train_loaders, 
                                        val_loaders, 
                                        model, 
                                        opt_args = opt_args,
                                        num_epochs=config["epochs"], 
                                        crit_string = "BCE", 
                                        train_criterion = BCEWithLogitsLoss(reduction="sum"), 
                                        train_crits = {"Acc": Accuracy()}, 
                                        test_crits = get_test_criteria(args.pred),   
                                        gamma=config["lr_decay"], 
                                        wandb_run = run, 
                                        trial = trial, 
                                        pruning = True if not args.multi_opt else False,
                                        graph_fn = graph_train_stats)
    
    if should_prune:
        run.summary["state"] = "pruned"
        wandb.finish(quiet=True)
        raise optuna.TrialPruned()

    test_values = eval_model(test_loaders, 
                            model,
                            test_crits = get_test_criteria(args.pred),  
                            wandb_run = run, 
                            multi=args.multi_opt)

    run.summary["state"] = "completed"
    wandb.finish(quiet=True)

    if args.multi_opt:
        return test_values["Test Acc", "Test BCE"]
    
    return test_values["Test Acc"]

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

    layer_dict = get_layer_dict()

    train_loader, val_loader, test_loader, data_details = get_loaders(data_dirs, target, args.batch_size)
    train_loaders = [train_loader]
    val_loaders = [val_loader]
    test_loaders = [test_loader]


    time_string = datetime.datetime.now().strftime('%d-%b-%Y-%H%M')
    if args.multi_opt:
        crits = get_test_criteria(args.pred)
        study = optuna.create_study(study_name=f"{time_string}_optimize_{target}", directions=["maximize", "minimize"])
        study.set_metric_names(crits.keys())
    else:
        study = optuna.create_study(study_name=f"{time_string}_optimize_{target}", direction="maximize")
        study.set_metric_names(["Test Acc"])

    study.optimize(lambda trial: objective(trial, data_details, train_loaders, val_loaders, test_loaders, layer_dict, args), n_trials=200)

    print(f"Best value: {study.best_value} (params: {study.best_params})")


## END UTILITY METHODS
##############################################################################

if __name__ == '__main__':
    main(parse_args())