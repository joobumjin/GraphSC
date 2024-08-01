import torch
from torch.nn import MSELoss
from tqdm import tqdm
import math
import matplotlib.pyplot as plt

def train(model, train_loader, optimizer, criterion):
    model.train()
    for data in tqdm(train_loader, desc="Training", leave=False):
        data = data.to(model.device)  # Move data to the same device as the model
        out = model(data)
        out = out.view(-1)  # Flatten the output to match target shape
        loss = criterion(out, data.y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def test(model, loader, criterion, print_met=True):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for data in tqdm(loader, desc="Testing", leave=False):
            data = data.to(model.device)
            out = model(data)
            out = out.view(-1) 
            loss = criterion(out, data.y)
            total_loss += loss.item()

            if print_met:
                print(f"Predicted: {out}, True: {data.y}, RMSE: {math.sqrt(loss.item())}")

    avg_loss = total_loss / len(loader.dataset)
    return math.sqrt(avg_loss)

def train_model(train_loader, val_loader, model, output_filepath, learning_rate, num_epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using,", device)
    model = model.to(device)
    model.device = device
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = MSELoss()

    train_losses = []
    val_losses = []

    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")
        train(model, train_loader, optimizer, criterion)
        train_rmse = test(model, train_loader, criterion, False)
        val_rmse = test(model, val_loader, criterion, False)

        train_losses.append(train_rmse)
        val_losses.append(val_rmse)

        print(f'Epoch: {epoch:03d}, Train RMSE: {train_rmse:.4f}, Val RMSE: {val_rmse:.4f}')
        print()
    
    torch.save(model.state_dict(), output_filepath)
    print("Saved the model to:", output_filepath)

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training RMSE')
    plt.plot(val_losses, label='Validation RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.title('Training and Validation RMSE')
    plt.legend()
    plt.show()




# import torch
# from torch.nn import MSELoss
# from tqdm import tqdm
# import math
# import matplotlib.pyplot as plt

# def train(model, train_loader, optimizer, criterion):
#     model.train()
#     for data in tqdm(train_loader, desc="Training", leave=False):
#         data = data.to(model.device)  # Move data to the same device as the model
#         out = model(data)
#         # print(out,data.y)
#         loss = criterion(out, data.y)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

# def test(model, loader, criterion, print_met=False):
#     model.eval()
#     total_loss = 0.0
#     with torch.no_grad():
#         for data in tqdm(loader, desc="Testing", leave=False):
#             data = data.to(model.device)
#             out = model(data)
#             loss = criterion(out, data.y)
#             total_loss += loss.item()

#             if print_met:
#                 print(f"Predicted: {out}, True: {data.y}, RMSE: {math.sqrt(loss.item())}")

#     avg_loss = total_loss / len(loader.dataset)
#     return math.sqrt(avg_loss)

# def train_model(train_loader, val_loader, model, output_filepath, learning_rate, num_epochs):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print("Using,", device)
#     model = model.to(device)
#     model.device = device
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#     criterion = MSELoss()

#     train_losses = []
#     val_losses = []

#     for epoch in range(1, num_epochs + 1):
#         print(f"Epoch {epoch}/{num_epochs}")
#         train(model, train_loader, optimizer, criterion)
#         train_rmse = test(model, train_loader, criterion, False)
#         val_rmse = test(model, val_loader, criterion, True)

#         train_losses.append(train_rmse)
#         val_losses.append(val_rmse)

#         print(f'Epoch: {epoch:03d}, Train RMSE: {train_rmse:.4f}, Val RMSE: {val_rmse:.4f}')
#         print()
    
#     torch.save(model.state_dict(), output_filepath)
#     print("Saved the model to:", output_filepath)

#     plt.figure(figsize=(10, 6))
#     plt.plot(train_losses, label='Training RMSE')
#     plt.plot(val_losses, label='Validation RMSE')
#     plt.xlabel('Epoch')
#     plt.ylabel('RMSE')
#     plt.title('Training and Validation RMSE')
#     plt.legend()
#     plt.show()
