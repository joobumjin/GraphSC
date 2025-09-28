import argparse
import time
import datetime

from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import wandb
import optuna

from preprocessing.img_preprocessing import get_image_loaders, HealthyData
from utils.train_test import train, train_multidata, train_multidata_timed, test, test_multidata, SSLELoss, RMSELoss
from GNN.src.dnn_f import DNN_F

def parse_args(args=None):
    """ 
    Perform command-line argument parsing (other otherwise parse arguments with defaults). 
    To parse in an interative context (i.e. in notebook), add required arguments.
    These will go into args and will generate a list that can be passed in.
    For example: 
        parse_args('--type', 'rnn', ...)
    """
    parser = argparse.ArgumentParser(description="Specify Hyperparameters to Optimize for the CNN", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data',           required=True,                                          help='File path to the assignment data file.')
    parser.add_argument('--pred',           required=True,  choices=['TER', 'VEGF', 'Both'],        help='Type of Value being Predicted from QBAMs')
    parser.add_argument('--batch_size',     type=int,       default=20,                             help='Model\'s batch size.')
    parser.add_argument('--multi_opt',      action="store_true",                                            help='Whether or not to optimize against mutliple objectives')
    parser.add_argument('--normed',         required=False, action='store_true',                    help='Whether or not to use normalized label values')
    parser.add_argument('--extra_data',     required=False, default=None,                           help='File path to the assignment data file.')

    if args is None: 
        return parser.parse_args()      ## For calling through command line
    return parser.parse_args(args)      ## For calling through notebook.


def get_test_criteria(task = None):
    test_crits = {
        "RMSE": RMSELoss(reduction='sum'),
    }
    
    if task and task == 'VEGF': test_crits["VEGF_RMSERatio"] = RMSELoss(reduction='sum', ratio=True)
    elif task and task == 'Both': 
        test_crits["TER_RMSE"] = RMSELoss(reduction='sum', start_ind=2, end_ind=3)
        test_crits["VEGF_RMSE"] = RMSELoss(reduction='sum', start_ind=0, end_ind=2)
        test_crits["VEGF_RMSERatio"] = RMSELoss(reduction='sum', ratio=True, start_ind=0, end_ind=2)

    return test_crits

##############################################################################

def train_model(train_loaders, val_loaders, model, opt_args, num_epochs, output_filepath = None, gamma=0.95, wandb_run = None, trial = None, pruning = False, task = None):
    #setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    timed = True
    print("Using", device)

    model = model.to(device)
    model.device = device

    optimizer = torch.optim.Adam(model.parameters(), **opt_args)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
    crit_string = "RMSE"
    train_criterion = RMSELoss(reduction='sum')
    train_crits = {}
    test_crits = get_test_criteria(task)
    
    train_losses = []
    train_metrics = {crit: [] for crit in train_crits}
    val_metrics = {crit: [] for crit in test_crits}

    #data
    if len(train_loaders) > 1: train_fn = train_multidata_timed if timed else train_multidata
    else: 
        train_fn = train
        train_loaders = train_loaders[0]

    if len(val_loaders) > 1: test_fn = test_multidata
    else: 
        test_fn = test
        val_loaders = val_loaders[0]

    #run
    epoch_tqdm = tqdm(range(1, num_epochs + 1), desc="Training Epochs", postfix={f"Train {crit_string}": 0.0, f"Valid {crit_string}": 0.0})

    start = time.time()
    
    for epoch in epoch_tqdm:
        #train
        train_loss, avg_data_time, avg_pred_time = train_fn(model, train_loaders, optimizer, train_criterion)
        scheduler.step()

        train_losses.append(train_loss)
        postfix = {f"Train {crit_string}": train_loss}

        #eval
        for crit_dict, metric_dict, loader, split in zip([train_crits, test_crits], [train_metrics, val_metrics], [train_loaders, val_loaders], ["Train", "Valid"]):
            for crit, crit_obj in crit_dict.items():
                metric = test_fn(model, loader, crit_obj)
                metric_dict[crit].append(metric)
                postfix[f"{split} {crit}"] = metric

        postfix[f"Epoch Time"] = time.time() - start
        postfix[f"Batching Time"] = avg_data_time
        postfix[f"Prediction Time"] = avg_pred_time

        if wandb_run: wandb_run.log(postfix)
        epoch_tqdm.set_postfix(postfix)

        if epoch % 15 == 0 and trial and pruning: 
            trial.report(postfix["Valid RMSE"], epoch)
            if trial.should_prune(): return postfix["Train RMSE"], postfix["Valid RMSE"], True

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

    return train_losses[-1], val_metrics["RMSE"][-1], False

def eval(test_loaders, model, wandb_run = None, task = None, multi = False):
    #setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using", device)

    model = model.to(device)
    model.device = device

    test_crits = get_test_criteria(task)

    #data
    if len(test_loaders) > 1: test_fn = test_multidata
    else: 
        test_fn = test
        test_loaders = test_loaders[0]

    metrics = {}

    #evaluate
    for crit_dict, loader, split in zip([test_crits], [test_loaders], ["Test"]):
        for crit, crit_obj in crit_dict.items():
            metric_calc = test_fn(model, loader, crit_obj)
            metrics[f"{split} {crit}"] = metric_calc
            if wandb_run: wandb_run.summary[f"{split} {crit}"] = metric_calc

    if multi: return tuple([metrics[name] for name in metrics.keys()])
    
    return metrics["Test RMSE"]

##############################################################################

def objective(trial, data_details, train_loaders, val_loaders, test_loaders, args):
    #
    #hyper params
    opt_args = {
        "lr": trial.suggest_float("learning_rate", 0.0001, 0.005, step=0.0001),
        "weight_decay": trial.suggest_float("l2_penalty", 0, 1e-2, step=5e-5),
    }

    config={
        "epochs": 60,
        "lr_decay": trial.suggest_float("learning_rate_decay", 0.7, 1.0, step=.1),
    }

    config = {**opt_args, **config}
    print(f"###############################################################################\n"
            f"Learning Rate: {config['lr']} with Decay {config['lr_decay']} and Weight Decay: {config['weight_decay']}\n"
            f"###############################################################################")

    #
    #build models
    model = DNN_F(*data_details)

    #
    #run
    run = wandb.init(
        entity="bumjin_joo-brown-university", 
        project=f"qbam-DNN-F-{args.pred}{"-Multi" if args.multi_opt else ""}", 
        name=f"DNN-F, LR{config["lr"]:.5f}", 
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
                    pruning = True if not args.multi_opt else False,
                    task = args.pred)
    
    if should_prune:
        run.summary["state"] = "pruned"
        wandb.finish()
        raise optuna.TrialPruned()

    test_values = eval(test_loaders, model, wandb_run = run, task=args.pred, multi=args.multi_opt)

    run.summary["state"] = "completed"
    wandb.finish()

    return test_values

##############################################################################

def main(args):
    sns.set_theme()

    target = args.pred
    print(f"Training DNN F on {target} with batch size {args.batch_size}")

    # norm_string = "_normalized" if args.normed else ""

    data_base_dir = f"{args.data}/full_imgs"
    data_dirs = {"train": [f"{data_base_dir}/train_TER_imgs_0.pkl", 
                           f"{data_base_dir}/train_TER_imgs_1.pkl", 
                           f"{data_base_dir}/train_TER_imgs_2.pkl"], 
                 "valid": f"{data_base_dir}/valid_TER_imgs_0.pkl", 
                 "test":  f"{data_base_dir}/test_TER_imgs_0.pkl"}
 
    print(f"Loading Data")
    train_loaders, val_loaders, test_loaders, out_dim = get_image_loaders(data_dirs, target, args.batch_size)

    time_string = datetime.datetime.now().strftime('%d-%b-%Y-%H%M')
    if args.multi_opt:
        crits = get_test_criteria(target)
        study = optuna.create_study(study_name=f"{time_string}_optimize_{target}", directions=["minimize" for _ in crits])
        study.set_metric_names(list(crits.keys()))
    else:
        study = optuna.create_study(study_name=f"{time_string}_optimize_{target}", direction="minimize")
        study.set_metric_names(["RMSE"])

    study.optimize(lambda trial: objective(trial, [out_dim], train_loaders, val_loaders, test_loaders, args), n_trials=150)

    print(f"Best value: {study.best_value} (params: {study.best_params})")



## END UTILITY METHODS
##############################################################################

if __name__ == '__main__':
    main(parse_args())