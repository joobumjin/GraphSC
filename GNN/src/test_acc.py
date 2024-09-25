import os
import torch
import torch_geometric
import numpy as np
import torch.nn.functional as F
from torch.nn import MSELoss
from tqdm import tqdm
import math
import matplotlib.pyplot as plt

def test(model, loader, criterion, write_to_file, vis_preds, task, print_met=True):
    total_loss = 0.0
    f = None
    all_preds = None

    with torch.no_grad():
        if write_to_file: f = open(write_to_file, "w")
        for data in loader:
            data = data.to(model.device)
            out = model(data)
            
            if all_preds is not None: 
                print(out)
                print(all_preds)
                all_preds = np.vstack((all_preds, out.numpy(force=True)))
            else: all_preds = out.numpy(force=True)

            loss = criterion(out, data.y.reshape(-1, model.output_dim))
            total_loss += loss.item()

            if print_met:
                print(f"Predicted: {out}, True: {data.y.reshape(-1, model.output_dim)}, RMSE: {math.sqrt(loss.item())}")
            if f:
                f.write(f"Predicted: {out}, True: {data.y.reshape(-1, model.output_dim)}, RMSE: {math.sqrt(loss.item())}\n")

    if vis_preds:
        label = data.y.reshape(-1, model.output_dim)[0]
        label_x = label[0]
        label_y = 0.0 if (model.output_dim == 1) else label[1]

        x_data = all_preds[:,0]
        y_data = np.zeros_like(x_data) if (model.output_dim == 1) else all_preds[:,1]
            
        plt.plot(x_data, y_data, "b+")
        plt.plot(label_x, label_y, "ro")
        if model.output_dim == 1:
            plt.ylabel(task)
        else:
            plt.xlabel('TER')
            plt.ylabel('VEGF')
        plt.title('Predicted and Ground Truth Values')
        plt.legend()
        plt.savefig(vis_preds)
        plt.close()
        print(f"Saved graph to {vis_preds}")

    avg_loss = total_loss / len(loader.dataset)
    if f: 
        f.write(f"Average Loss: {math.sqrt(avg_loss)}\n")
        print(f"Wrote Prediciton Outputs to {write_to_file}")
        f.close()
    return math.sqrt(avg_loss)

def test_model(test_loader, model, task, write_to_file=None, vis_preds=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.device = device
    criterion = MSELoss()

    model.eval()
    test_rmse = test(model, test_loader, criterion, write_to_file=write_to_file, vis_preds=vis_preds, task=task, print_met=False)

    print(f'Test RMSE: {test_rmse:.4f}')
    print()
    return test_rmse
    

