import argparse
from pathlib import Path
import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns

from preprocessing import get_loaders
from utils import SSLELoss, RMSELoss, train_model, eval_model, get_test_criteria
from models import Modular_GNN


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
    parser.add_argument('--pre_pred',       required=True,  choices=['TER', 'VEGF', 'Both'],        help='Type of Value being Predicted from QBAMs')
    parser.add_argument('--trans_pred',     required=True,  choices=['TER', 'VEGF', 'Both'],        help='Type of Value being Predicted from QBAMs')
    parser.add_argument('--log_path',       default='',                                             help='where the optuna study logs will stored')
    parser.add_argument('--batch_size',     type=int,       default=20,                             help='Model\'s batch size.')

    if args is None: 
        return parser.parse_args()      ## For calling through command line
    return parser.parse_args(args)      ## For calling through notebook.

def graph_train_stats(train_losses, train_metrics, val_metrics, wandb_run):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('RMSE')
    p1 = ax1.plot(train_losses, label='Training RMSE', color='tab:blue')
    p2 = ax1.plot(val_metrics["RMSE"], label='Validation RMSE', color="tab:orange")
    ax1.tick_params(axis='y')

    ax1.legend(handles=p1+p2, loc='best')

    #outoutting
    fig.tight_layout()  
    plt.title('Training and Validation RMSE')
    plt.legend()

    if wandb_run: wandb_run.log({"chart": plt})

    plt.close()

##############################################################################

def optimize(target, model, opt_args, config, train_loaders, val_loaders, test_loaders, args, model_params = None):
    #
    #run
    _, _, _, _ = train_model(train_loaders, 
                             val_loaders, 
                             model, 
                             opt_args = opt_args,
                             num_epochs=config["epochs"], 
                             crit_string = "RMSE", 
                             train_criterion = RMSELoss(reduction="sum"), 
                             train_crits = {}, 
                             test_crits = get_test_criteria(target),  
                             gamma=config["lr_decay"], 
                             wandb_run = None, 
                             trial = None, 
                             pruning = False,
                             graph_fn = graph_train_stats,
                             model_params = model_params)

    test_values = eval_model(test_loaders, 
                             model,
                             test_crits = get_test_criteria(target),  
                             wandb_run = None, 
                             multi=args.multi_opt)

    return test_values["Test RMSE"]

##############################################################################

def main(args):
    sns.set_theme()
    
    ## build data
    pretrain_target = args.pre_pred
    print(f"Pretraining {pretrain_target}")

    data_dirs = {}
    for data_type in ['TER', 'VEGF', 'Both']:
        data_dirs[f"Train_{data_type}"] = f"{args.data}/{data_type}/Train_{data_type}.pkl"
        data_dirs[f"Valid_{data_type}"] = f"{args.data}/{data_type}/Valid_{data_type}.pkl"
        data_dirs[f"Test_{data_type}"] = f"{args.data}/{data_type}/Test_{data_type}.pkl"

    Path(f"{args.data}/pretrain_{pretrain_target}/Train_graphs").mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, data_details = get_loaders(data_dirs, pretrain_target, args.batch_size)
    train_loaders = [train_loader]
    val_loaders = [val_loader]
    test_loaders = [test_loader]

    #build model
    model_args = {
        "num_node_features": data_details[0], 
        "output_dim": data_details[1],  
        "num_gcn": 4,
        "num_dense": 4,
        "hidden_channels": 128, 
        "dense_hidden": 128, 
        "dropout_p": 0.4,
    }
    
    config={
        "graph layer": "GATv2",
        "lr_decay": 0.8,
        "epochs": 200
    }
    model = Modular_GNN(**model_args)

    print(f"###############################################################################\n"
            f"{model_args["num_gcn"]} GATv2 Layers\t| {model_args["hidden_channels"]} units\n"
            f"{model_args["num_dense"]} Dense Layers\t| {model_args["dense_hidden"]} units\n"
            f"Dropout Rate: {model_args["dropout_p"]}\n"
            f"Learning Rate: {config["lr"]} with Decay {config["lr_decay"]} and Weight Decay: {config["weight_decay"]}\n"
            f"###############################################################################")

    #
    #build models
    opt_args = {
        "lr": 0.004,
        "weight_decay": 0.005
    }

    #pretraining
    test_pretrain_loss = optimize(pretrain_target, model, opt_args, config, train_loaders, val_loaders, test_loaders, args)

    #next step
    transfer_target = args.trans_pred
    print(f"Finetuning {transfer_target}")

    #build data
    Path(f"{args.data}/transfer_{transfer_target}/Train_graphs").mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, data_details = get_loaders(data_dirs, transfer_target, args.batch_size)
    train_loaders = [train_loader]
    val_loaders = [val_loader]
    test_loaders = [test_loader]

    #freeze entire model and remove last dense layer
    for param in model.parameters(): param.requires_grad = False
    model.guillotine_last()

    #perform swap on last layer to new linear probe
    model.output_dim = data_details[1]
    new_dense = torch.nn.Linear(model_args["dense_hidden"], data_details[1])
    for param in new_dense.parameters(): param.requires_grad = True
    model.extend_dense([new_dense])

    #finetuning
    config["epochs"] = 20
    test_transfer_loss = optimize(transfer_target, model, opt_args, config, train_loaders, val_loaders, test_loaders, args, new_dense.parameters())

    with open(f'{pretrain_target}to{transfer_target}.log', 'a') as out_log:
        out_log.write(f"Pretraining {pretrain_target}: Final Test Loss {test_pretrain_loss}")
        out_log.write(f"Finetuning {transfer_target}: Final Test Loss: {test_transfer_loss}")
        out_log.write(f"----------------------------------\n")


## END UTILITY METHODS
##############################################################################

if __name__ == '__main__':
    main(parse_args())