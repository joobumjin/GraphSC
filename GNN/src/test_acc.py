import os
import torch
import torch_geometric
import numpy as np
import torch.nn.functional as F
from torch.nn import MSELoss
from tqdm import tqdm
import math
import matplotlib.pyplot as plt

def test(model, loader, criterion, write_to_file, print_met=True):
    total_loss = 0.0
    f = None
    with torch.no_grad():
        if write_to_file: f = open(write_to_file, "x")
        for data in loader:
            data = data.to(model.device)
            out = model(data)
            out = out.view(-1) 
            loss = criterion(out, data.y)
            total_loss += loss.item()

            if print_met:
                print(f"Predicted: {out}, True: {data.y}, RMSE: {math.sqrt(loss.item())}")
            if f:
                f.write(f"Predicted: {out}, True: {data.y}, RMSE: {math.sqrt(loss.item())}")

    avg_loss = total_loss / len(loader.dataset)
    if f: 
        print(f"Wrote Prediciton Outputs to {write_to_file}")
        f.close()
    return math.sqrt(avg_loss)

def test_model(test_loader, model, write_to_file=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.device = device
    criterion = MSELoss()

    model.eval()
    test_rmse = test(model, test_loader, criterion, write_to_file=write_to_file, print_met=False)

    print(f'Test RMSE: {test_rmse:.4f}')
    print()
    

