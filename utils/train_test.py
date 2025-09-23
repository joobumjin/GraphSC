import math
import torch
from tqdm import tqdm
from abc import ABC, abstractmethod

class Accuracy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, actual):
        return torch.sum(torch.gt(pred, 0.5)==(actual==1.0))
    
class SSLELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.root = True
        self.mse = torch.nn.MSELoss(reduction='sum')

    def forward(self, pred, actual):
        return self.mse(pred, torch.log(actual + 1))
    
class RMSELoss(torch.nn.Module):
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

def train(model, train_loader, optimizer, criterion, metric_printer=None):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for data in train_loader:
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

        if metric_printer is not None:
                metric_printer(out,data.y.reshape(-1, model.output_dim), math.sqrt(loss.item() / len(data.y.reshape(-1, model.output_dim))))

    metric = total_loss / total_samples

    if hasattr(criterion, "root") and criterion.root: metric = math.sqrt(metric)

    return metric

def train_multidata(model, train_loaders, optimizer, criterion):
    model.train()
    total_loss = 0.0
    total_samples = 0
    for train_loader in train_loaders:
        for data in train_loader:
            optimizer.zero_grad()
            data = data.to(model.device)  # Move data to the same device as the model
            out = model(data)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()

            total_loss += loss.detach().item()
            total_samples += torch.numel(data.y)

    metric = total_loss / total_samples

    if hasattr(criterion, "root") and criterion.root: metric = math.sqrt(metric)

    return metric

def test(model, loader, criterion, metric_printer=None, log_train = False):
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

def test_multidata(model, test_loaders, criterion, metric_printer=None, log_train = False):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    for loader in test_loaders: total_samples += len(loader)
    with torch.no_grad(): 
        for loader in test_loaders:
            for data in loader:
                data = data.to(model.device)
                out = model(data)
                if log_train: out = torch.exp(out)
                loss = criterion(out, data.y)
                total_loss += loss.item()
                total_samples += torch.numel(data.y)
    
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