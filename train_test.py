import math
import torch
from tqdm import tqdm
from abc import ABC, abstractmethod

def train(model, train_loader, optimizer, criterion):
    model.train()
    for data in train_loader:
        data = data.to(model.device)  # Move data to the same device as the model
        out = model(data)
        loss = criterion(out, data.y.reshape(-1, model.output_dim))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def test(model, loader, criterion, metric_printer=None):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for data in tqdm(loader, desc="Testing", leave=False):
            data = data.to(model.device)
            out = model(data)
            loss = criterion(out, data.y.reshape(-1, model.output_dim))
            total_loss += loss.item()

            if metric_printer:
                print(metric_printer(out,data.y.reshape(-1, model.output_dim), math.sqrt(loss.item())))

    avg_loss = total_loss / len(loader.dataset)
    return math.sqrt(avg_loss)

class MetricPrinter(ABC):
    @abstractmethod
    def __call__(self, preds, labels, loss):
        pass