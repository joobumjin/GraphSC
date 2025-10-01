import time

from tqdm import tqdm
import numpy as np
import torch

from utils.train_test import train, train_multidata, train_multidata_timed, test, test_multidata


def train_model(train_loaders, val_loaders, model, opt_args, num_epochs, crit_string, train_criterion, train_crits, test_crits, output_filepath = None, gamma=0.95, wandb_run = None, trial = None, pruning = False, graph_fn = None, timed=False):
    #setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using", device)

    model = model.to(device)
    model.device = device

    optimizer = torch.optim.Adam(model.parameters(), **opt_args)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
    
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
    epoch_tqdm = tqdm(range(1, num_epochs + 1), desc="Training Epochs", postfix={})

    start = time.time()
    
    for epoch in epoch_tqdm:
        #train
        train_loss = train_fn(model, train_loaders, optimizer, train_criterion)
        if timed: (train_loss, avg_data_time, avg_pred_time) = train_loss
        scheduler.step()

        train_losses.append(train_loss)
        postfix = {f"Train {crit_string}": train_loss}

        #eval
        for crit_dict, metric_dict, loader, split in zip([train_crits, test_crits], [train_metrics, val_metrics], [train_loaders, val_loaders], ["Train", "Valid"]):
            for crit, crit_obj in crit_dict.items():
                metric = test_fn(model, loader, crit_obj)
                metric_dict[crit].append(metric)
                postfix[f"{split} {crit}"] = metric

        if timed:
            postfix[f"Epoch Time"] = time.time() - start
            postfix[f"Batching Time"] = avg_data_time
            postfix[f"Prediction Time"] = avg_pred_time

        if wandb_run: wandb_run.log(postfix)
        epoch_tqdm.set_postfix(postfix)

        if epoch % 15 == 0 and trial and pruning: 
            trial.report(postfix[f"Valid {crit_string}"], epoch)
            
            if trial.should_prune(): 
                train_losses = np.array(train_losses)
                train_metrics = {crit: np.array(history) for crit, history in train_metrics.items()}
                val_metrics = {crit: np.array(history) for crit, history in val_metrics.items()}
                return train_losses, train_metrics, val_metrics, True

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
    if graph_fn is not None: graph_fn(train_losses, train_metrics, val_metrics, wandb_run)

    return train_losses, train_metrics, val_metrics, False

def eval_model(test_loaders, model, test_crits, wandb_run = None, multi = False):
    #setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using", device)

    model = model.to(device)
    model.device = device

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

    return metrics