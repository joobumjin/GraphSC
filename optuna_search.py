import argparse
import datetime

from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import optuna
import wandb

from preprocessing.preprocessing import get_loaders, get_feature_labels
from utils import SSLELoss, RMSELoss, train_model, eval_model, save_model, visualize_feature_importance, get_test_criteria
from torch_geometric.nn import GraphConv, GCNConv, GATConv, GATv2Conv, TransformerConv
from models import Modular_GNN, get_layer_dict

from torch_geometric.explain import Explainer, GNNExplainer


best_rmse = None

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
    parser.add_argument('--save_path',      type=str,                                                       help='Optional path to save model weights to')

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

def get_explanation(model, data, run):
    data = data.to(model.device)
    
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

    explanation = explainer(data.x, data.edge_index, batch_index = data.batch)

    visualize_feature_importance(explanation, feat_labels = get_feature_labels(), run=run)

##############################################################################

def objective(trial, data_dirs, layer_dict, args):
    #
    #hyper params
    layer = trial.suggest_categorical("layer type", layer_dict.keys())
    feat_norm = trial.suggest_categorical("feature normalization", ['', '_01', '_z'])

    train_loaders, val_loaders, test_loaders, data_details = get_loaders(data_dirs, f"{args.pred}{feat_norm}", args.batch_size)
    train_loaders = [train_loaders]
    val_loaders = [val_loaders]
    test_loaders = [test_loaders]

    model_args = {
        "num_node_features": data_details[0], 
        "output_dim": data_details[1],  
        "num_gnn": trial.suggest_int("num_gnn", 2, 3),
        "num_dense": trial.suggest_int("num_dense", 3, 5), 
        "hidden_channels": trial.suggest_int("hidden_size", 64, 256, step=64), 
        "dense_hidden": trial.suggest_int("hidden_size", 64, 256, step=64), 
        "dropout_p": trial.suggest_float("dropout", 0.1, 0.7, step=0.1),
        "gnn_layer": layer_dict[layer]
    }

    opt_args = {
        "lr": trial.suggest_float("learning_rate", 0.0001, 0.005, step=0.0001),
        "weight_decay": trial.suggest_float("l2_penalty", 0, 1e-2, step=5e-5),
        # "betas": (0.9, 0.95)
    }

    sched_args = {
        "warmup_epochs": 10
    }

    config={
        "graph layer": f"{layer}",
        "epochs": 50,
        # "lr_decay": trial.suggest_float("learning_rate_decay", 0.7, 1.0, step=.1),
        "target": args.pred,
        "feature_normalization": feat_norm
    }

    config = {**model_args, **opt_args, **sched_args, **config}
    print(f"###############################################################################\n"
            f"{model_args["num_gnn"]} {layer} Layers\t| {model_args["hidden_channels"]} units\n"
            f"{model_args["num_dense"]} Dense Layers\t| {model_args["dense_hidden"]} units\n"
            f"Dropout Rate: {model_args["dropout_p"]}\n"
            f"Learning Rate: {config["lr"]} with Half Cosine Decay and Weight Decay: {config["weight_decay"]}\n"
            f"###############################################################################")

    #
    #build models
    model = Modular_GNN(**model_args)

    #
    #run
    run = wandb.init(
        entity="bumjin_joo-brown-university", 
        project=f"qbam-DATA-Graph-{args.pred}-{args.dataset}{"-Multi" if args.multi_opt else ""}", 
        name=f"{layer}, LR{config["lr"]:.5f}, {feat_norm}", 
        config=config
    )
    
    _, _, _, should_prune = train_model(train_loaders, 
                                     val_loaders, 
                                     model, 
                                     opt_args = opt_args,
                                     num_epochs=config["epochs"], 
                                     crit_string = "RMSE", 
                                     train_criterion = RMSELoss(reduction="sum"), 
                                     train_crits = {}, 
                                     test_crits = get_test_criteria(args.pred),  
                                     scheduler_args = sched_args, #  gamma=config["lr_decay"], 
                                     wandb_run = run, 
                                     trial = trial, 
                                     pruning = True if not args.multi_opt else False,
                                     graph_fn = graph_train_stats)
    
    if should_prune:
        run.summary["state"] = "pruned"
        wandb.finish()
        raise optuna.TrialPruned()

    test_values = eval_model(test_loaders, 
                             model,
                             test_crits = get_test_criteria(args.pred),  
                             wandb_run = run)

    global best_rmse
    if args.save_path and (best_rmse is None or test_values["Test RMSE"] < best_rmse):
        save_model(model, model_args, config, f"{args.save_path}/{layer}_{args.pred}_RMSE{test_values["Test RMSE"]}")
        best_rmse = test_values["Test RMSE"]

        sample_batch = next(iter(test_loaders[0]))
        get_explanation(model, sample_batch, run)        

    run.summary["state"] = "completed"
    wandb.finish()

    if args.multi_opt:
        return [test_values[key] for key in test_values]

    return test_values["Test RMSE"]

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

        data_dirs[f"Train_{data_type}_01"] = f"{args.data}/{data_type}_featnorm/Train_{data_type}_01.pkl"
        data_dirs[f"Valid_{data_type}_01"] = f"{args.data}/{data_type}_featnorm/Valid_{data_type}_01.pkl"
        data_dirs[f"Test_{data_type}_01"] = f"{args.data}/{data_type}_featnorm/Test_{data_type}_01.pkl"

        data_dirs[f"Train_{data_type}_z"] = f"{args.data}/{data_type}_featnorm/Train_{data_type}_z.pkl"
        data_dirs[f"Valid_{data_type}_z"] = f"{args.data}/{data_type}_featnorm/Valid_{data_type}_z.pkl"
        data_dirs[f"Test_{data_type}_z"] = f"{args.data}/{data_type}_featnorm/Test_{data_type}_z.pkl"

    layer_dict = get_layer_dict()

    #
    # optuna optimization
    time_string = datetime.datetime.now().strftime('%d-%b-%Y-%H%M')
    if args.multi_opt:
        crits = get_test_criteria(target)
        study = optuna.create_study(study_name=f"{time_string}_optimize_{target}", directions=["minimize" for _ in crits])
        study.set_metric_names(list(crits.keys()))
    else:
        study = optuna.create_study(study_name=f"{time_string}_optimize_{target}", direction="minimize")
        study.set_metric_names(["RMSE"])

    study.optimize(lambda trial: objective(trial, data_dirs, layer_dict, args), n_trials=200)

    print(f"Best value: {study.best_value} (params: {study.best_params})")


## END UTILITY METHODS
##############################################################################

if __name__ == '__main__':
    main(parse_args())