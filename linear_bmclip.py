import json

from PIL import Image
import torch
import pickle
import numpy as np
from torch.utils.data import DataLoader
from torch.nn import Linear
from utils import train, train_multidata, test, test_multidata, SSLELoss, RMSELoss, train_model, eval_model
import matplotlib.pyplot as plt
import seaborn as sns


class Data():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def to(self, device):
       self.x = self.x.to(device)
       self.y = self.y.to(device)

       return self
    
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
    
def collate(data):
    """
    In our cases, we want to collate a list of Data instances
    """
    #better to prealloc numpy?
    images = torch.stack([sample.x for sample in data])
    labels = torch.stack([sample.y for sample in data])

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

    plt.savefig(f"/users/bjoo2/data/bjoo2/qbam/data/biomedclip_probe")

    plt.close()

def main():
    sns.set_theme()
    # Download the model and config files
    encoding_out = f"/users/bjoo2/data/bjoo2/qbam/data/biomedclip_embeds"
    loaders = []

    for split in ["Train", "Val", "Test"]:
        embeds = pickle.load(f"{encoding_out}/{split}.pkl")
        batch_size=32

        print(f"Constructing Dataloaders")
        loaders.append([DataLoader(embeds, batch_size = batch_size, collate_fn=lambda data: collate(data))])
        
    train_loaders, val_loaders, test_loaders = loaders
    print(f"Data loaded")

    opt_args = {
        "lr":1e-3,
        "weight_decay": 1e-1,
    }

    config={
        "epochs": 30,
        "lr_decay": .95,
    }
        

    model = torch.nn.Linear(512, 1)
    # print(linear_probe(encoding).shape)
    _, _, _, _ = train_model(train_loaders, 
                            val_loaders, 
                            model, 
                            opt_args = opt_args,
                            num_epochs=config["epochs"], 
                            crit_string = "RMSE", 
                            train_criterion = RMSELoss(reduction="sum"), 
                            train_crits = {}, 
                            test_crits = get_test_criteria(),  
                            gamma=config["lr_decay"], 
                            wandb_run = None, 
                            trial = None, 
                            pruning = False,
                            graph_fn = graph_train_stats)

    test_values = eval_model(test_loaders, 
                            model,
                            test_crits = get_test_criteria(),  
                            wandb_run = None, 
                            multi=False)
    
    print(f"Test Performance: {test_values}")

##############################################################################

if __name__ == '__main__':
    main()