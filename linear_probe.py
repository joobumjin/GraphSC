import argparse
import os
import json

from PIL import Image
import torch
import pickle
import numpy as np
from torch.utils.data import DataLoader
from torch.nn import Linear, Sequential, LeakyReLU, BatchNorm1d
from torch.nn.init import trunc_normal_
from utils import SSLELoss, RMSELoss, train_model, eval_model, get_test_criteria
import matplotlib.pyplot as plt
import seaborn as sns

graph_dir = ""

def parse_args(args=None):
    """ 
    Perform command-line argument parsing (other otherwise parse arguments with defaults). 
    To parse in an interative context (i.e. in notebook), add required arguments.
    These will go into args and will generate a list that can be passed in.
    For example: 
        parse_args('--type', 'rnn', ...)
    """
    parser = argparse.ArgumentParser(description="Specify Hyperparameters to Optimize for the GNN", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data',           required=True,                                                  help='File path to the assignment data file.')
    parser.add_argument('--pred',                                   default="TER",                          help='Type of Value being Predicted from QBAMs')
    parser.add_argument('--dataset',                                default="Healthy",                      help='Name of Dataset.')
    parser.add_argument('--batch_size',     type=int,               default=32,                             help='Model\'s batch size.')
    parser.add_argument('--model',          type=str,               default="biomedclip",                   help='Model whose embeddings are being linear probed')
    
    if args is None: 
        return parser.parse_args()      ## For calling through command line
    return parser.parse_args(args)      ## For calling through notebook.

class LinearProbe(torch.nn.Module):
    def __init__(self, input_dim, out_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = out_dim

        layers = [BatchNorm1d(self.input_dim),
                  Linear(self.input_dim, 128),
                  LeakyReLU(),
                  Linear(128, 32),
                  LeakyReLU(),
                  Linear(32, self.output_dim)]
        
        for layer in layers: 
            if hasattr(layer, "weight"): trunc_normal_(layer.weight, std=0.01)

        self.linear = Sequential(*layers)

    def forward(self, data):
        x = self.linear(data.x)

        return x

class Data():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def to(self, device):
       self.x = self.x.to(device)
       self.y = self.y.to(device)

       return self
    
def collate(data):
    """
    In our cases, we want to collate a list of Data instances
    """
    #better to prealloc numpy?
    images = torch.stack([sample.x[0] for sample in data])
    labels = torch.stack([sample.y[0] for sample in data])

    return Data(images, labels)

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

    os.makedirs(graph_dir, exist_ok=True)
    plt.savefig(f"{graph_dir}/train_graph.png")

    plt.close()

def main(args):
    sns.set_theme()
    embed_fp = f"{args.data}/{args.model}_embeds"
    loaders = []
    pred = "TER"

    for split in ["Train", "Val", "Test"]:
        with open(f"{embed_fp}/{split}.pkl", 'rb') as f:
            embeds = pickle.load(f)

            print(f"Constructing Dataloaders")
            loaders.append([DataLoader(embeds, batch_size = args.batch_size, shuffle = True, collate_fn=lambda data: collate(data))])

    in_dims = embeds[0].x.shape[1]
    out_dims = embeds[0].y.shape[1]
        
    train_loaders, val_loaders, test_loaders = loaders
    print(f"Data loaded")

    opt_args = {
        "lr":1.5e-4,
        "weight_decay": 1e-1,
        "betas": (0.9, 0.95)
    }

    config={
        "epochs": 300,
        # "lr_decay": .95,
    }

    global graph_dir
    graph_dir = f"{args.data}/{args.model}_probe"
        
    model_args = {
        "input_dim": in_dims,
        "out_dim": out_dims,
    }

    sched_args = {
        "warmup_epochs": 10
    }

    model = LinearProbe(**model_args)
    # print(linear_probe(encoding).shape)
    _, _, _, _ = train_model(train_loaders, 
                            val_loaders, 
                            model, 
                            opt_args = opt_args,
                            num_epochs=config["epochs"], 
                            crit_string = "RMSE", 
                            train_criterion = RMSELoss(reduction="sum"), 
                            train_crits = {}, 
                            test_crits = get_test_criteria(pred),  
                            scheduler_args = sched_args, #gamma=config["lr_decay"], 
                            wandb_run = None, 
                            trial = None, 
                            pruning = False,
                            graph_fn = graph_train_stats)

    test_values = eval_model(test_loaders, 
                            model,
                            test_crits = get_test_criteria(pred),  
                            wandb_run = None)
    
    print(f"Test Performance: {test_values}")

##############################################################################

if __name__ == '__main__':
    main(parse_args)