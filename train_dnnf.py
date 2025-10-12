import argparse
import datetime

import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import optuna

from preprocessing.img_preprocessing import get_image_loaders, ImageData
from utils import SSLELoss, RMSELoss, train_model, eval_model
from models import DNN_F, DNN_F_AMD

def parse_args(args=None):
    """ 
    Perform command-line argument parsing (other otherwise parse arguments with defaults). 
    To parse in an interative context (i.e. in notebook), add required arguments.
    These will go into args and will generate a list that can be passed in.
    For example: 
        parse_args('--type', 'rnn', ...)
    """
    parser = argparse.ArgumentParser(description="Specify Hyperparameters to Optimize for the CNN", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data',           required=True,                                              help='File path to the assignment data file.')
    parser.add_argument('--pred',           required=True,      choices=['TER', 'VEGF', 'Both'],        help='Type of Value being Predicted from QBAMs')
    parser.add_argument('--dataset',        default="Healthy",  choices=['Healthy', 'AMD'],             help='Name of Dataset.')
    parser.add_argument('--batch_size',     type=int,           default=20,                             help='Model\'s batch size.')
    parser.add_argument('--multi_opt',      action="store_true",                                        help='Whether or not to optimize against mutliple objectives')
    parser.add_argument('--normed',         required=False,     action='store_true',                    help='Whether or not to use normalized label values')
    parser.add_argument('--extra_data',     required=False,     default=None,                           help='File path to the assignment data file.')

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

def graph_train_stats(train_losses, train_metrics, val_metrics, wandb_run):
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
    
    return

##############################################################################

def objective(trial, data_details, train_loaders, val_loaders, test_loaders, args):
    #
    #hyper params
    opt_args = {
        "lr": trial.suggest_float("learning_rate", 0.0001, 0.005, step=0.0001),
        "weight_decay": trial.suggest_float("l2_penalty", 0, 1e-2, step=5e-5),
    }

    sched_args = {
        "warmup_epochs": 10
    }

    config={
        "epochs": 60,
        # "lr_decay": trial.suggest_float("learning_rate_decay", 0.7, 1.0, step=.1),
    }

    config = {**opt_args, **sched_args, **config}
    print(f"###############################################################################\n"
            f"Learning Rate: {config['lr']} with Half Cosine Weight Decay and Weight decay {config["weight_decay"]}\n"
            f"###############################################################################")

    #
    #build models
    model_dict = {"Healthy": DNN_F, "AMD": DNN_F_AMD}
    model = model_dict[args.dataset](*data_details)

    #
    #run
    run = wandb.init(
        entity="bumjin_joo-brown-university", 
        project=f"qbam-DNN-F-{args.pred}-{args.dataset}{"-Multi" if args.multi_opt else ""}", 
        name=f"DNN-F, LR{config["lr"]:.5f}", 
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
                                        scheduler_args = sched_args, # gamma=config["lr_decay"], 
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
                             get_test_criteria(args.pred),
                             wandb_run = run)

    run.summary["state"] = "completed"
    wandb.finish()

    if args.multi_opt:
        return [test_values[key] for key in test_values]

    return test_values["Test RMSE"]

##############################################################################

def main(args):
    sns.set_theme()

    target = args.pred
    print(f"Training DNN F on {target} with batch size {args.batch_size}")

    # norm_string = "_normalized" if args.normed else ""

    data_base_dir = f"{args.data}/full_imgs"
    if args.dataset == "Healthy":
        data_dirs = {"train": [f"{data_base_dir}/{args.dataset}/train_TER_imgs_0.pkl", 
                            f"{data_base_dir}/{args.dataset}/train_TER_imgs_1.pkl", 
                            f"{data_base_dir}/{args.dataset}/train_TER_imgs_2.pkl"], 
                    "valid": [f"{data_base_dir}/{args.dataset}/valid_TER_imgs_0.pkl"], 
                    "test":  [f"{data_base_dir}/{args.dataset}/test_TER_imgs_0.pkl"]}
    elif args.dataset == "AMD":
        data_dirs = {"train": [f"{data_base_dir}/{args.dataset}/train_TER.pkl"], 
                    "valid": [f"{data_base_dir}/{args.dataset}/valid_TER.pkl"], 
                    "test":  [f"{data_base_dir}/{args.dataset}/test_TER.pkl"]}
    
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

    study.optimize(lambda trial: objective(trial, [out_dim], train_loaders, val_loaders, test_loaders, args), n_trials=30)

    print(f"Best value: {study.best_value} (params: {study.best_params})")



## END UTILITY METHODS
##############################################################################

if __name__ == '__main__':
    main(parse_args())