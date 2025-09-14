import argparse
from pathlib import Path
import datetime
import wandb

from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from torch.nn import BCELoss

from pair_preprocessing import PairData, get_loaders
from train_test import Accuracy, train, train_multidata, test, test_multidata, StandardInlinePrint
import GNN.src.gnn_multiple as GCNs
from GNN.src import test_acc
from GNN.src.gnn_merge import GCN_Merge

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
    #
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using", device)

    model = model.to(device)
    model.device = device
    opt_args = {name: arg for (arg, name) in zip([learning_rate, weight_decay], ["lr", "weight_decay"]) if arg is not None}
    optimizer = torch.optim.Adam(model.parameters(), **opt_args)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
    crit_string = "BCE"
    train_criterion = BCELoss(reduction='sum')
    train_crits = {
        "Acc": Accuracy()
    }
    test_crits = {
        "BCE": BCELoss(reduction='sum'),
        "Acc": Accuracy()
    }
    
    train_losses = []
    train_metrics = {crit: [] for crit in train_crits}
    val_metrics = {crit: [] for crit in test_crits}

    #
    #data
    #
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
    #
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
    #
    train_losses = np.array(train_losses)
    train_metrics = {crit: np.array(history) for crit, history in train_metrics.items()}
    val_metrics = {crit: np.array(history) for crit, history in val_metrics.items()}

    #
    #model saving
    #
    if output_filepath:
        torch.save(model.state_dict(), output_filepath)
        print("Saved the model to:", output_filepath)

    #
    #plotting
    # 
    #losses
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('BCE')
    ax1.plot(train_losses, label='Training BCE', color='tab:blue')
    ax1.plot(val_metrics["BCE"], label='Validation BCE', color="tab:orange")
    ax1.legend()
    ax1.tick_params(axis='y')
    #accs
    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy')
    ax2.plot(train_metrics["Acc"], label='Train Accuracy', color="tab:green")
    ax2.plot(val_metrics["Acc"], label='Validation Accuracy', color="tab:red")
    ax2.legend()
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

def main(args):
    sns.set_theme()

    #
    # get data
    #
    target = args.pred
    print(f"Training {target}")

    data_dirs = {}
    for data_type in ['TER', 'VEGF', 'Both', 'Donor']:
        data_dirs[f"Train_{data_type}"] = f"{args.data}/{data_type}/train_pairwise_donors.pkl"
        data_dirs[f"Valid_{data_type}"] = f"{args.data}/{data_type}/valid_pairwise_donors.pkl"
        data_dirs[f"Test_{data_type}"] = f"{args.data}/{data_type}/test_pairwise_donors.pkl"

    Path(f"{args.data}/{target}/Train_graphs").mkdir(parents=True, exist_ok=True)

    model_constructors = GCNs.get_model_constructors()

    train_loader, val_loader, test_loader, data_details = get_loaders(data_dirs, target, args.batch_size)
    train_loaders = [train_loader]
    val_loaders = [val_loader]
    test_loaders = [test_loader]

    #
    #hyper params
    #
    num_epochs = 100
    num_gcn = 4
    num_dense = 5
    hidden_size = 144
    dense_hidden = 512
    arch_string = f"G{num_gcn}_D{num_dense}"
    learning_rate = 0.001
    lr_decay = 0.5
    weight_decay = 0.005
    dropout_rate = 0.5

    config={
        "architecture": "GATv2-Donor-Merge",
        "dataset": "Donor",
        "epochs": num_epochs,
        "num_gcn": num_gcn,
        "num_dense": num_dense,
        "hidden_size": hidden_size,
        "dense_hidden": dense_hidden,
        "learning_rate": learning_rate,
        "lr_decay": lr_decay,
        "weight_decay": weight_decay,
        "dropout_rate": dropout_rate,
    }

    #
    #build models
    #
    model_class = model_constructors[arch_string]
    model1 = model_class(*data_details, hidden_channels = hidden_size, dense_hidden = dense_hidden, dropout_p=dropout_rate)
    model2 = model_class(*data_details, hidden_channels = hidden_size, dense_hidden = dense_hidden, dropout_p=dropout_rate)
    gnn_merge = GCN_Merge(model1, model2)

    print(f"#########################################################################################\
            {num_gcn} GCN Layers | {hidden_size} units\n\
            {num_dense} Dense Layers | {dense_hidden}\n\
            Dropout Rate: {dropout_rate}\n\
            Learning Rate: {learning_rate} with Decay {lr_decay} and Weight Decay: {weight_decay}\n#########################################################################################")

    loss_graph_path = f"{args.data}/Train_graphs/Merge.jpeg"

    #
    #run
    #
    with wandb.init(entity="bumjin_joo-brown-university", project="qbam-donor", name="Test-Merge", config=config) as run:
        run.watch(model1)
        run.watch(model2)
        run.watch(gnn_merge)

        _, _ = train_model(train_loaders, 
                        val_loaders, 
                        gnn_merge, 
                        learning_rate, 
                        num_epochs, 
                        img_path=loss_graph_path, 
                        gamma=lr_decay, 
                        weight_decay=weight_decay,
                        wandb_run = run)

    print(f"Validation Stats")
    _ = test_acc.test_model(val_loaders, gnn_merge, task=target, test_multiple=False)
    print(f"Test Stats")
    test_loss = test_acc.test_model(test_loaders, gnn_merge, task=target, test_multiple=False)


## END UTILITY METHODS
##############################################################################

if __name__ == '__main__':
    main(parse_args())