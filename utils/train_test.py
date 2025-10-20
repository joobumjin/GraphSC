import math
import torch
import torch.utils.data as torch_data
import torch_geometric.loader as torch_geom
from abc import ABC, abstractmethod
import time
from torch.nn import BCEWithLogitsLoss
from typing import Dict, Tuple, Optional
from functools import reduce
from utils.lr_sched import HalfCosDecay

"""
Simple Accuracy Metric
"""
class Accuracy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, actual):
        return torch.sum(torch.gt(pred, 0.5)==(actual==1.0))

"""
Sum Squared Log Error
Looks to align pred with the log of the labels
"""    
class SSLELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.root = True
        self.mse = torch.nn.MSELoss(reduction='sum')

    def forward(self, pred, actual):
        return self.mse(pred, torch.log(actual + 1))

"""
Root Mean Squared Error Loss
"""    
class RMSELoss(torch.nn.Module):
    """
    reduction: string passed to MSELoss initializer, 
                reduction operation used on batch of losses
    ratio: used to indicate that we should first calculate 
                the loss of the ratio of _only_ the first two elements
    start_ind: start of the tensors' second dimension from which to calculate loss
    end_ind: end of the tensors' second dimension from which to calculate loss
    """
    def __init__(self, reduction='sum', ratio=False, start_ind=None, end_ind=None):
        super().__init__()
        self.root = True
        self.ratio = ratio
        if start_ind: self.start_ind = start_ind
        if end_ind: self.end_ind = end_ind
        self.mse = torch.nn.MSELoss(reduction=reduction)

    def forward(self, pred, actual):
        #assume that the ratio elements are the first two values
        if self.ratio:
            pred = pred[:, 0:1] / pred[:, 1:2]
            actual = actual[:, 0:1] / actual[:, 1:2]
        return self.mse(pred, actual)
    
"""
Returns a dictionary of testing time criterion
Represented as name : loss object
"""
def get_test_criteria(task: Optional[str] = None) -> Dict[str, torch.nn.Module]:
    if task == "Donor":
        return {"Acc": Accuracy(), "BCE": BCEWithLogitsLoss(reduction='sum')}

    test_crits = {
        "RMSE": RMSELoss(reduction='sum'),
    }
    
    if task and task == 'VEGF': test_crits["VEGF_RMSERatio"] = RMSELoss(reduction='sum', ratio=True)
    elif task and task == 'Both': 
        test_crits["TER_RMSE"] = RMSELoss(reduction='sum', start_ind=2, end_ind=3)
        test_crits["VEGF_RMSE"] = RMSELoss(reduction='sum', start_ind=0, end_ind=2)
        test_crits["VEGF_RMSERatio"] = RMSELoss(reduction='sum', ratio=True, start_ind=0, end_ind=2)

    return test_crits

"""
Given the loaders, calls the correct train function to train for a single epoch
"""
def train(model: torch.nn.Module, 
          loaders: list[torch_data.DataLoader | torch_geom.DataLoader], 
          optimizer: torch.optim.Optimizer, 
          criterion: torch.nn.Module, 
          scheduler: HalfCosDecay, 
          epoch: Optional[int] = 0, 
          timed: Optional[bool] = False):
    if len(loaders) > 1: train_fn = train_multidata_timed if timed else train_multidata
    else: 
        train_fn = train_singledata
        loaders = loaders[0]
    
    return train_fn(model, loaders, optimizer, criterion, scheduler, epoch)

"""
Runs training for one epoch given a single dataloader
"""
def train_singledata(model, loader, optimizer, criterion, scheduler, epoch = 0) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0

    for ind, data in enumerate(loader):
        scheduler.adjust_learning_rate(optimizer, ind / len(loader) + epoch)
        
        optimizer.zero_grad()
        data = data.to(model.device)  # Move data to the same device as the model
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.detach().item()
        if not hasattr(criterion, "ratio") or not criterion.ratio:
            total_samples += torch.numel(data.y)
        else:
            total_samples += (data.y.shape[0] * (data.y.shape[1] - 1))

    metric = total_loss / total_samples

    if hasattr(criterion, "root") and criterion.root: metric = math.sqrt(metric)

    return metric

"""
Runs training for a single epoch when given multiple
training set dataloaders
"""
def train_multidata(model, loaders, optimizer, criterion, scheduler, epoch = 0) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    processed_batches = 0.0
    total_batches = reduce(lambda sum, loader: sum + len(loader), loaders, 0.0)

    for train_loader in loaders:
        for data in train_loader:
            scheduler.adjust_learning_rate(optimizer, processed_batches / total_batches + epoch)
            
            optimizer.zero_grad()
            data = data.to(model.device)  # Move data to the same device as the model
            out = model(data)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()

            total_loss += loss.detach().item()
            processed_batches += 1
            if not hasattr(criterion, "ratio") or not criterion.ratio:
                total_samples += torch.numel(data.y)
            else:
                total_samples += (data.y.shape[0] * (data.y.shape[1] - 1))

    metric = total_loss / total_samples

    if hasattr(criterion, "root") and criterion.root: metric = math.sqrt(metric)

    return metric

"""
Runs training for a single epoch when given multiple
training set dataloaders

Additionally reports the time taken to load the batch,
and then perform and forward and backward step on the batch
"""
def train_multidata_timed(model, loaders, optimizer, criterion, scheduler, epoch = 0) -> Tuple[float, float, float]:
    model.train()
    total_loss = 0.0
    total_samples = 0
    processed_batches = 0.0
    total_batches = reduce(lambda sum, loader: sum + len(loader), loaders, 0.0)

    total_batch_time = 0.0
    total_process_time = 0.0
    for train_loader in loaders:
        data_start_time = time.time()
        for data in train_loader:
            scheduler.adjust_learning_rate(optimizer, processed_batches / total_batches + epoch)
            total_batch_time += time.time() - data_start_time
            
            optimizer.zero_grad()
            data = data.to(model.device)  # Move data to the same device as the model
            process_start_time = time.time()
            out = model(data)
            loss = criterion(out, data.y)
            total_process_time += time.time() - process_start_time
            loss.backward()
            optimizer.step()

            total_loss += loss.detach().item()
            processed_batches += 1
            if not hasattr(criterion, "ratio") or not criterion.ratio:
                total_samples += torch.numel(data.y)
            else:
                total_samples += (data.y.shape[0] * (data.y.shape[1] - 1))
            data_start_time = time.time()

    metric = total_loss / total_samples

    if hasattr(criterion, "root") and criterion.root: metric = math.sqrt(metric)

    return metric, total_batch_time / total_batches, total_process_time / total_batches

"""
Given the dataloaders, calls the correct test function 
Returns calculated criterion value
"""
def test(model: torch.nn.Module, 
         loaders: list[torch_data.DataLoader | torch_geom.DataLoader], 
         criterion: torch.nn.Module):
    if len(loaders) > 1: test_fn = test_multidata
    else: 
        test_fn = test
        loaders = loaders[0]
    
    return test_fn(model, loaders, criterion)

"""
Runs testing with a criterion and a single dataloader
"""
def test_singledata(model, loader, criterion, log_train = False):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(model.device)
            out = model(data)
            labels = data.y
            if log_train: out = torch.exp(out)
            if hasattr(criterion, "start_ind"): 
                out = out[:, criterion.start_ind:criterion.end_ind]
                labels = labels[:, criterion.start_ind:criterion.end_ind]
            loss = criterion(out, labels)
            total_loss += loss.item()
            if not hasattr(criterion, "ratio") or not criterion.ratio:
                total_samples += torch.numel(data.y)
            else:
                total_samples += (data.y.shape[0] * (data.y.shape[1] - 1))

    metric = total_loss / total_samples

    if hasattr(criterion, "root") and criterion.root: metric = math.sqrt(metric)

    return metric

"""
Runs testing on a single criterion
when testing split data is contained in multiple dataloaders
"""
def test_multidata(model, loaders, criterion, log_train = False):
    model.eval()
    total_loss = 0.0
    total_samples = 0.0

    with torch.no_grad(): 
        for loader in loaders:
            for data in loader:
                data = data.to(model.device)
                out = model(data)
                if log_train: out = torch.exp(out)
                loss = criterion(out, data.y)
                total_loss += loss.item()
                if not hasattr(criterion, "ratio") or not criterion.ratio:
                    total_samples += torch.numel(data.y)
                else:
                    total_samples += (data.y.shape[0] * (data.y.shape[1] - 1))
    
    metric = total_loss / total_samples
    
    if hasattr(criterion, "root") and criterion.root: metric = math.sqrt(metric)

    return metric

class MetricPrinter(ABC):
    @abstractmethod
    def __call__(self, preds, labels, loss):
        pass


class StandardInlinePrint(MetricPrinter):
    def __call__(self, preds, labels, loss):
        print(f"Predicted: {preds}, True: {labels}, RMSE: {math.sqrt(loss)}")