import os
import torch
import torch_geometric
import numpy as np
import torch.nn.functional as F
from torch.nn import MSELoss
from tqdm import tqdm
import math
import matplotlib.pyplot as plt

def test(model, loader, criterion, write_to_file, vis_preds, task, print_met=True, log_train=False):
    total_loss = 0.0
    total_samples = 0
    f = None
    all_preds = None

    if print_met: print("-----------------------------------------")

    with torch.no_grad():
        if write_to_file: f = open(write_to_file, "w")
        for data in loader:
            data = data.to(model.device)
            label = data.y.reshape(-1, model.output_dim)[0]
            num_samples = len(data.y.reshape(-1, model.output_dim))
            out = model(data)
            if log_train: out = torch.exp(out)
            
            if vis_preds:
                if all_preds is not None: 
                    all_preds = torch.vstack((all_preds, out))
                else: all_preds = out

            loss = criterion(out, data.y.reshape(-1, model.output_dim))
            total_loss += loss.item()
            total_samples += num_samples

            if print_met:
                print(f"Predicted: {out}, True: {label}, RMSE: {math.sqrt(loss.item()/num_samples)}")
            if f:
                f.write(f"Predicted: {out}, True: {label}, RMSE: {math.sqrt(loss.item()/num_samples)}\n")

    if vis_preds:
        data_dict = {}

        vegf_label_x, vegf_label_y, vegf_pred_x, vegf_pred_y = None, None, None, None
        ter_label_x, ter_label_y, ter_pred_x, ter_pred_y = None, None, None, None

        if task == "Both":
            vegf_label_x = label[0].numpy(force=True)
            vegf_label_y = label[1].numpy(force=True)

            vegf_pred_x = all_preds[:,0].numpy(force=True)
            vegf_pred_y = all_preds[:,1].numpy(force=True)

            ter_label_x = label[2].numpy(force=True)
            ter_label_y = 0.0

            ter_pred_x = all_preds[:,2].numpy(force=True)
            ter_pred_y = np.zeros_like(ter_pred_x)

        elif task == "VEGF":
            vegf_label_x = label[0].numpy(force=True)
            vegf_label_y = label[1].numpy(force=True)
            vegf_pred_x = all_preds[:,0].numpy(force=True)
            vegf_pred_y = all_preds[:,1].numpy(force=True)
        
        elif task == "TER":
            ter_label_x = label[0].numpy(force=True)
            ter_label_y = 0.0
            ter_pred_x = all_preds[:,0].numpy(force=True)
            ter_pred_y = np.zeros_like(ter_pred_x)

        if vegf_label_x is not None:
            data_dict["VEGF"]  = [(vegf_label_x, vegf_label_y), (vegf_pred_x, vegf_pred_y)]
        if ter_label_x is not None:
            data_dict["TER"]  = [(ter_label_x, ter_label_y), (ter_pred_x, ter_pred_y)]
        
        for task_type in data_dict:
            x_data,y_data = data_dict[task_type][0]
            label_x,label_y = data_dict[task_type][1]
            plt.plot(x_data, y_data, "b+", label="Model Predictions")
            plt.plot(label_x, label_y, "ro", label="Ground Truth")
            if task_type == "TER":
                plt.xlabel("TER")
            else:
                plt.xlabel('VEGF Numerator')
                plt.ylabel('VEGF Denominator')
            plt.title('Predicted and Ground Truth Values')
            plt.legend()
            plt.savefig(f"{vis_preds}_{task_type}.jpg")
            plt.close()
            print(f"Saved predictions graph to {f"{vis_preds}_{task_type}.jpg"}")

    avg_loss = total_loss / total_samples
    if f: 
        f.write(f"Average Loss: {math.sqrt(avg_loss)}\n")
        print(f"Wrote Prediciton Outputs to {write_to_file}")
        f.close()
    return math.sqrt(avg_loss)

def test_multi(model, loaders, criterion, write_to_file, vis_preds, task, print_met=True):
    total_loss = 0.0
    total_preds = 0
    f = None
    all_preds = None

    with torch.no_grad():
        if write_to_file: f = open(write_to_file, "w")
        for loader in loaders:
            for data in loader:
                data = data.to(model.device)
                num_samples = len(data.y.reshape(-1, model.output_dim))
                out = model(data)
                
                if vis_preds:
                    if all_preds is not None: 
                        all_preds = torch.vstack((all_preds, out))
                    else: all_preds = out

                loss = criterion(out, data.y.reshape(-1, model.output_dim))
                total_loss += loss.item()
                total_preds += num_samples

                if print_met:
                    print(f"Predicted: {out}, True: {data.y.reshape(-1, model.output_dim)}, RMSE: {math.sqrt(loss.item()/num_samples)}")
                if f:
                    f.write(f"Predicted: {out}, True: {data.y.reshape(-1, model.output_dim)}, RMSE: {math.sqrt(loss.item()/num_samples)}\n")

    if vis_preds:
        label = data.y.reshape(-1, model.output_dim)[0]

        data_dict = {}

        vegf_label_x, vegf_label_y, vegf_pred_x, vegf_pred_y = None, None, None, None
        ter_label_x, ter_label_y, ter_pred_x, ter_pred_y = None, None, None, None

        if task == "Both":
            vegf_label_x = label[0].numpy(force=True)
            vegf_label_y = label[1].numpy(force=True)

            vegf_pred_x = all_preds[:,0].numpy(force=True)
            vegf_pred_y = all_preds[:,1].numpy(force=True)

            ter_label_x = label[2].numpy(force=True)
            ter_label_y = 0.0

            ter_pred_x = all_preds[:,2].numpy(force=True)
            ter_pred_y = np.zeros_like(ter_pred_x)

        elif task == "VEGF":
            vegf_label_x = label[0].numpy(force=True)
            vegf_label_y = label[1].numpy(force=True)
            vegf_pred_x = all_preds[:,0].numpy(force=True)
            vegf_pred_y = all_preds[:,1].numpy(force=True)
        
        elif task == "TER":
            ter_label_x = label[0].numpy(force=True)
            ter_label_y = 0.0
            ter_pred_x = all_preds[:,0].numpy(force=True)
            ter_pred_y = np.zeros_like(ter_pred_x)

        if vegf_label_x is not None:
            data_dict["VEGF"]  = [(vegf_label_x, vegf_label_y), (vegf_pred_x, vegf_pred_y)]
        if ter_label_x is not None:
            data_dict["TER"]  = [(ter_label_x, ter_label_y), (ter_pred_x, ter_pred_y)]
        
        for task_type in data_dict:
            x_data,y_data = data_dict[task_type][0]
            label_x,label_y = data_dict[task_type][1]
            plt.plot(x_data, y_data, "b+", label="Model Predictions")
            plt.plot(label_x, label_y, "ro", label="Ground Truth")
            if task_type == "TER":
                plt.xlabel("TER")
            else:
                plt.xlabel('VEGF Numerator')
                plt.ylabel('VEGF Denominator')
            plt.title('Predicted and Ground Truth Values')
            plt.legend()
            plt.savefig(f"{vis_preds}_{task_type}.jpg")
            plt.close()
            print(f"Saved predictions graph to {f"{vis_preds}_{task_type}.jpg"}")

    avg_loss = total_loss / total_preds

    if f: 
        f.write(f"Average Loss: {math.sqrt(avg_loss)}\n")
        print(f"Wrote Prediciton Outputs to {write_to_file}")
        f.close()

    print(f"Sanity Check: {total_preds} test samples")
    return math.sqrt(avg_loss)

def test_model(test_loader, model, task, write_to_file=None, vis_preds=None, test_multiple=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.device = device
    criterion = MSELoss(reduction='sum')

    model.eval()
    if test_multiple or (isinstance(test_loader, list) and len(test_loader) > 1):
        test_rmse = test_multi(model, test_loader, criterion, write_to_file=write_to_file, vis_preds=vis_preds, task=task, print_met=True)
    else:
        if isinstance(test_loader, list): test_loader = test_loader[0]
        test_rmse = test(model, test_loader, criterion, write_to_file=write_to_file, vis_preds=vis_preds, task=task, print_met=True)

    print(f'Test RMSE: {test_rmse:.4f}')
    print()
    return test_rmse
    

