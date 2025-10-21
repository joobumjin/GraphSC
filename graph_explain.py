import argparse
import datetime

from tqdm import tqdm
import pandas
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from preprocessing.preprocessing import get_loaders, get_feature_labels
from utils import SSLELoss, RMSELoss, train_model, eval_model, visualize_feature_importance, get_test_criteria, load_model
from torch_geometric.nn import GraphConv, GCNConv, GATConv, GATv2Conv, TransformerConv
from torch_geometric.explain import Explainer, GNNExplainer
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
    parser.add_argument('--data',           required=True,                                                  help='File path to the assignment data file.')
    parser.add_argument('--pred',           required=True,          choices=['TER', 'VEGF', 'Both'],        help='Type of Value being Predicted from QBAMs')
    parser.add_argument('--dataset',                                                                        help='Name of Dataset.')
    parser.add_argument('--batch_size',     type=int,               default=20,                             help='Model\'s batch size.')
    parser.add_argument('--multi_opt',      action="store_true",                                            help='Whether or not to optimize against mutliple objectives')
    parser.add_argument('--model_path',     type=str,                                                       help='Optional path to saved model weights')

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

def optimize(data_details, train_loaders, val_loaders, test_loaders, args):
    #
    #hyper params
    layer = GATConv

    model_args = {
        "num_node_features": data_details[0], 
        "output_dim": data_details[1],  
        "num_gnn": 3,
        "num_dense": 4, 
        "hidden_channels": 256, 
        "dense_hidden": 256, 
        "dropout_p": 0.7,
        "gnn_layer": layer
    }

    opt_args = {
        "lr": 0.0015,
        "weight_decay": 0.0095,
    }

    config={
        "epochs": 75,
        "lr_decay": 0.9,
    }

    config = {**model_args, **opt_args, **config}
    print(f"###############################################################################\n"
            f"{model_args["num_gnn"]} {layer} Layers\t| {model_args["hidden_channels"]} units\n"
            f"{model_args["num_dense"]} Dense Layers\t| {model_args["dense_hidden"]} units\n"
            f"Dropout Rate: {model_args["dropout_p"]}\n"
            f"Learning Rate: {config["lr"]} with Decay {config["lr_decay"]} and Weight Decay: {config["weight_decay"]}\n"
            f"###############################################################################")

    #
    #build models
    model = Modular_GNN(**model_args)

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
                            test_crits = get_test_criteria(args.pred),  
                            gamma=config["lr_decay"], 
                            wandb_run = None, 
                            trial = None, 
                            pruning = True if not args.multi_opt else False,
                            graph_fn = None)

    return model

##############################################################################

def main(args):
    sns.set_theme()

    #
    # get data
    target = args.pred
    print(f"Training {target}")

    data_dirs = {}
    for data_type in ['TER', 'VEGF', 'Both']:
        data_dirs[f"Train_{data_type}"] = f"{args.data}/{data_type}/Train_{data_type}.pkl"
        data_dirs[f"Valid_{data_type}"] = f"{args.data}/{data_type}/Valid_{data_type}.pkl"
        data_dirs[f"Test_{data_type}"] = f"{args.data}/{data_type}/Test_{data_type}.pkl"

    train_loader, val_loader, test_loader, data_details = get_loaders(data_dirs, target, args.batch_size)
    train_loaders = [train_loader]
    val_loaders = [val_loader]
    test_loaders = [test_loader]

    #train model
    if args.model_path is None:
        model = optimize(data_details, train_loaders, val_loaders, test_loaders, args)
    else:
        model, _, _ = load_model(Modular_GNN, args.model_path)

        model.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(model.device)

    #explain model
    explainer = Explainer(
        model=model,
        algorithm= GNNExplainer(epochs=200), #AttentionExplainer()
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='regression',
            task_level='graph',
            return_type='raw'
        ),
    )

    data = next(iter(test_loader))
    data = data.to(model.device)

    explanation = explainer(data.x, data.edge_index, batch_index = data.batch)
    print(explanation.edge_mask)
    print(explanation.node_mask)

    _ = visualize_feature_importance(explanation, f"{args.data}/explanations/feature_importance.png", feat_labels = get_feature_labels())

## END UTILITY METHODS
##############################################################################

if __name__ == '__main__':
    main(parse_args())