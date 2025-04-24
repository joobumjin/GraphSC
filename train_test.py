import math
import torch
from tqdm import tqdm
from abc import ABC, abstractmethod
import seaborn as sns

sns.set_theme()


class SSLELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss(reduction='sum')
        
    def forward(self, pred, actual):
        return self.mse(pred, torch.log(actual + 1))


def train(model, train_loader, optimizer, criterion, metric_printer=None):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for data in train_loader:
        optimizer.zero_grad()
        data = data.to(model.device)  # Move data to the same device as the model
        out = model(data)
        loss = criterion(out, data.y.reshape(-1, model.output_dim))
        loss.backward()
        optimizer.step()
        total_loss += loss.detach().item()
        total_samples += torch.numel(data.y)

        if metric_printer is not None:
                metric_printer(out,data.y.reshape(-1, model.output_dim), math.sqrt(loss.item() / len(data.y.reshape(-1, model.output_dim))))

    return math.sqrt(total_loss / total_samples)

def train_multidata(model, train_loaders, optimizer, criterion):
    model.train()
    total_loss = 0.0
    total_samples = 0
    for train_loader in train_loaders:
        for data in train_loader:
            optimizer.zero_grad()
            data = data.to(model.device)  # Move data to the same device as the model
            out = model(data)
            loss = criterion(out, data.y.reshape(-1, model.output_dim))
            loss.backward()
            optimizer.step()

            total_loss += loss.detach().item()
            # total_samples += len(data.y.reshape(-1, model.output_dim))
            total_samples += torch.numel(data.y)

    return math.sqrt(total_loss / total_samples)

def test(model, loader, criterion, metric_printer=None, log_train = False):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(model.device)
            out = model(data)
            if log_train: out = torch.exp(out)
            loss = criterion(out, data.y.reshape(-1, model.output_dim))
            total_loss += loss.item()
            total_samples += torch.numel(data.y)
            # total_samples += len(data.y.reshape(-1, model.output_dim))

            if metric_printer is not None:
                metric_printer(out,data.y.reshape(-1, model.output_dim), math.sqrt(loss.item() / len(data.y.reshape(-1, model.output_dim))))

    avg_loss = total_loss / total_samples
    return math.sqrt(avg_loss)

def test_multidata(model, test_loaders, criterion, metric_printer=None, log_train = False):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    for loader in test_loaders: total_samples += len(loader)
    with torch.no_grad(): #, tqdm(total=total_samples, desc="Testing", postfix={"Test RMSE": 0.0}) as pbar:
        for loader in test_loaders:
            for data in loader:
                data = data.to(model.device)
                out = model(data)
                if log_train: out = torch.exp(out)
                loss = criterion(out, data.y.reshape(-1, model.output_dim))
                total_loss += loss.item()
                total_samples += torch.numel(data.y)
                # total_samples += len(data.y.reshape(-1, model.output_dim))

                if metric_printer:
                    metric_printer(out,data.y.reshape(-1, model.output_dim), math.sqrt(loss.item() / len(data.y)))

    if total_samples > 0: avg_loss = total_loss / total_samples
    else:
        print("ERROR: 0 len dataset")
        return total_loss
    return math.sqrt(avg_loss)

class MetricPrinter(ABC):
    @abstractmethod
    def __call__(self, preds, labels, loss):
        pass


class StandardInlinePrint(MetricPrinter):
    def __call__(self, preds, labels, loss):
        print(f"Predicted: {preds}, True: {labels}, RMSE: {math.sqrt(loss)}")