import time
from typing import Optional, Dict, Callable, Iterator
import copy

from tqdm import tqdm
import numpy as np
import pandas as pd

import torch
from torch.nn.parameter import Parameter
import wandb
import optuna
import torch.utils.data as torch_data
import torch_geometric.loader as torch_geom

from utils.train_test import train, test
from utils.lr_sched import HalfCosDecay
from utils.early_stop import EarlyStopper

"""
Converts list of floats into np arrays and returns all in same form as input
"""
def format_outputs(train_losses: list[float], metric_histories: list[Dict[str, list[float]]]):
    train_losses = np.array(train_losses)
    metric_histories = [{crit: np.array(history) for crit, history in hist_dict.items()} for hist_dict in metric_histories]

    return train_losses, *metric_histories

"""
Runs entire training regime for a model
For each epoch, trains on train set and evaluates on train and validation set
params:
    train_loaders: Training data loaders in a list
    val_loaders: Validation dataloaders in a list
    model: Model to be trained
    opt_args: Arguments to pass to optimizer in Dictionary:
        keys:
            "lr": learning rate of optimizer
            "weight_decay": l2 penalty term
    num_epochs: Number of epochs to be trained
    crit_string: String describing the training loss function
    train_criterion: nn.Module to calculate the training loss
    train_crits: Other remaining evaluation criterion for the train set
    test_crits: Criterion on which the model will be evaluated with the test set
    scheduler_args: Arguments to pass to the half cosine decay learning rate scheduler
        warmup_epochs: number of epochs to warm up to full start learning rate, default 10
        max_epochs: number of training epochs, default 100
        min_lr: minimum lr to decay to, default 0
        start_lr: lr to start at after full warmup, default 1e-3
    wandb_run: Optional W&B run to record things to
    trial: Optional Optuna trial for optimization
    pruning: Whether or not to use Optuna Pruning
    graph_fn: The function to be used to graph model performance over time
    timed: Whether or not to use timed training functions
    model_params: Optional specified parameters to be trained
    return_best: Optional boolean, whether or not a copy of the best performing model should be returned
    eval_key: Optional string on which best model performance should be determined
    eval_maximize: Optional boolean on whether larger is better for eval key performance
returns:
    train_losses: np array of train loss per epoch
    train_metrics: dictionary of all extra training metrics per epoch
            {string of criterion name : np array of criterion per epoch} 
    val_metrics: dictionary of all validation metrics per epoch
            {string of criterion name : np array of criterion per epoch} 
    pruned: whether or not optuna decided to prune 
            (always false if pruning disabled)
"""
def train_model(train_loaders: torch_data.DataLoader | torch_geom.DataLoader, 
                val_loaders: torch_data.DataLoader | torch_geom.DataLoader, 
                model: torch.nn.Module, opt_args: Dict[str, float], 
                num_epochs: int, 
                crit_string: str, train_criterion: torch.nn.Module, 
                train_crits: Dict[str, torch.nn.Module], 
                test_crits: Dict[str, torch.nn.Module], 
                scheduler_args: Optional[Dict[str, int|float]] = {}, 
                wandb_run: Optional[wandb.Run] = None, 
                trial: Optional[optuna.Trial] = None, pruning: Optional[bool] = False, 
                graph_fn: Optional[Callable[..., None]] = None, 
                timed: Optional[bool]=False, 
                model_params: Optional[Iterator[Parameter]] = None,
                return_best: Optional[bool] = False,
                eval_key: Optional[str] = "RMSE",
                eval_maximize: Optional[bool] = False):
    #setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Training using", device)

    model = model.to(device)
    model.device = device

    pruned = False

    best_model = []
    if return_best:
        best_val = None
        eval_key = f"Valid {eval_key}"
        compare = lambda x, y: x > y if eval_maximize else lambda x, y: x < y

    params = model_params if model_params is not None else model.parameters()
    optimizer = torch.optim.AdamW(params, **opt_args)
    scheduler_args["max_epochs"] = num_epochs
    scheduler_args["start_lr"] = opt_args["lr"]
    scheduler = HalfCosDecay(**scheduler_args)

    stopper = EarlyStopper(patience = 3, direction = "minimize")
    
    train_losses = []
    train_hist = {crit: [] for crit in train_crits}
    val_hist   = {crit: [] for crit in test_crits}

    #run
    epoch_tqdm = tqdm(range(1, num_epochs + 1), desc="Training Epochs", postfix={})

    start = time.time()
    
    for epoch in epoch_tqdm:
        #train
        train_loss = train(model, train_loaders, optimizer, train_criterion, scheduler, epoch=epoch)
        if timed: (train_loss, avg_data_time, avg_pred_time) = train_loss

        train_losses.append(train_loss)
        postfix = {f"Train {crit_string}": train_loss}

        #eval
        for crit_dict, metric_hist, loader, split in zip([train_crits, test_crits], [train_hist, val_hist], [train_loaders, val_loaders], ["Train", "Valid"]):
            for crit, crit_obj in crit_dict.items():
                metric = test(model, loader, crit_obj)
                metric_hist[crit].append(metric)
                postfix[f"{split} {crit}"] = metric

        if timed:
            postfix[f"Epoch Time"] = time.time() - start
            postfix[f"Batching Time"] = avg_data_time
            postfix[f"Prediction Time"] = avg_pred_time

        postfix["lr"] = optimizer.param_groups[0]["lr"]

        #log
        if wandb_run: wandb_run.log(postfix)
        epoch_tqdm.set_postfix(postfix)

        if trial and pruning: trial.report(postfix[f"Valid {crit_string}"], epoch)

        if return_best and (best_val is None or compare(postfix[eval_key], best_val)):
            best_val = postfix[eval_key]
            best_model = [copy.deepcopy(model)]

        if stopper.check_stop(train_loss): break

        elif epoch % 10 == 0 and trial and pruning and trial.should_prune(): 
            print("Pruned by Optuna")
            pruned = True
            break

    epoch_tqdm.close()

    #output formating
    train_losses, train_hist, val_hist = format_outputs(train_losses, [train_hist, val_hist])

    #plotting
    if graph_fn is not None: graph_fn(train_losses, train_hist, val_hist, wandb_run)

    return train_losses, train_hist, val_hist, pruned, *best_model


"""
Runs evaluation step for a model
For each epoch, evaluates on given loaders
params:
    test_loaders: Data loaders in a list upon which the model should be tested
    model: Model to be tested
    test_crits: Criterion on which the model will be evaluated with the test set
    wandb_run: Optional W&B run to record things to
returns:
    metrics: dictionary of all metrics
            {string of criterion name : float value of criterion per epoch} 
"""
def eval_model(test_loaders: torch_data.DataLoader | torch_geom.DataLoader, 
               model: torch.nn.Module, 
               test_crits: Dict[str, torch.nn.Module],  
               wandb_run: wandb.Run = None):
    #setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Evaluating using", device)

    model = model.to(device)
    model.device = device

    metric_hists = {}

    #evaluate
    for crit_dict, loader, split in zip([test_crits], [test_loaders], ["Test"]):
        for crit, crit_obj in crit_dict.items():
            metric_calc = test(model, loader, crit_obj)
            metric_hists[f"{split} {crit}"] = metric_calc
            if wandb_run: wandb_run.summary[f"{split} {crit}"] = metric_calc

    return metric_hists

def gather_preds(all_loaders: Dict[str, list[torch_data.DataLoader | torch_geom.DataLoader]], 
                 model: torch.nn.Module, 
                 scatter_fn,
                 target: str = "TER",
                 wandb_run: wandb.Run = None,
                 save_path: Optional[str] = None):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Evaluating using", device)

    model = model.to(device)
    model.device = device
    model.eval()

    ters = pd.DataFrame(columns=["Predicted", "Ground Truth", "Split"])
    vegfs = pd.DataFrame(columns=["Predicted", "Ground Truth", "Split"])

    for split, loaders in all_loaders.items():
        for loader in loaders:
            for data in loader:
                data = data.to(model.device)  # Move data to the same device as the model
                out = model(data).detach()
                ter = out[:, 0] if target in ["TER", "Both"] else None
                gt_ter = data.y[:, 0] if target in ["TER", "Both"] else None
                vegf, gt_vegf = None, None
                if target == "VEGF":
                    vegf, gt_vegf = out[:, 0] / out[:, 1], data.y[:, 0] / data.y[:, 1]
                elif target == "Both":
                    vegf, gt_vegf = out[:, 1] / out[:, 2], data.y[:, 1] / data.y[:, 2]
                
                if target in ["TER", "Both"]:
                    new_entries = pd.DataFrame({"Predicted": ter, "Ground Truth": gt_ter, "Split": [split for _ in range(len(data.x))]})
                    ters = pd.concat((ters, new_entries)) if len(ters) > 0 else new_entries
                if target in ["VEGF", "Both"]:
                    new_entries = pd.DataFrame({"Predicted": vegf, "Ground Truth": gt_vegf, "Split": [split for _ in range(len(data.x))]})
                    vegfs = pd.concat((vegfs, new_entries)) if len(vegfs) > 0 else new_entries

    scatter_fn(ters, vegfs, wandb_run, save_path)