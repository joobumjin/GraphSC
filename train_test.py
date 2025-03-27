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

def train_multidata(model, train_loaders, optimizer, criterion):
    model.train()
    total_loss = 0.0
    total_samples = 0
    for train_loader in train_loaders:
        for data in train_loader:
            data = data.to(model.device)  # Move data to the same device as the model
            out = model(data)
            loss = criterion(out, data.y.reshape(-1, model.output_dim))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.detach().item()
            total_samples += 1

    print(f"Training Processed {total_samples}")

    return math.sqrt(total_loss / total_samples)

def test(model, loader, criterion, metric_printer=None):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for data in tqdm(loader, desc="Testing"):
            data = data.to(model.device)
            out = model(data)
            loss = criterion(out, data.y.reshape(-1, model.output_dim))
            total_loss += loss.item()
            total_samples += 1

            if metric_printer:
                print(metric_printer(out,data.y.reshape(-1, model.output_dim), math.sqrt(loss.item())))

    if total_samples > 0: avg_loss = total_loss / total_samples
    else:
        print("ERROR: 0 len dataset")
        return total_loss
    return math.sqrt(avg_loss)

def test_multidata(model, test_loaders, criterion, metric_printer=None):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    for loader in test_loaders: total_samples += len(loader)
    with torch.no_grad(), tqdm(total=total_samples, desc="Testing", postfix={"Test RMSE": 0.0}) as pbar:
        cur_sampled = 0
        for loader in test_loaders:
            for data in loader:
                data = data.to(model.device)
                out = model(data)
                loss = criterion(out, data.y.reshape(-1, model.output_dim))
                total_loss += loss.item()
                cur_sampled += 1

                if metric_printer:
                    print(metric_printer(out,data.y.reshape(-1, model.output_dim), math.sqrt(loss.item())))

                pbar.update(1)
                pbar.set_postfix({"Test RMSE": math.sqrt(total_loss / cur_sampled)})

    if total_samples > 0: avg_loss = total_loss / total_samples
    else:
        print("ERROR: 0 len dataset")
        return total_loss
    return math.sqrt(avg_loss)

class MetricPrinter(ABC):
    @abstractmethod
    def __call__(self, preds, labels, loss):
        pass