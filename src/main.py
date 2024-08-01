import pickle
import train_model
import test_acc
import gnn_model
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch

def check_for_nan(dataset):
    for i, data in enumerate(dataset):
        if torch.isnan(data.x).any():
            print(f"NaN found in features at index {i}")
        if torch.isnan(data.y).any():
            print(f"NaN found in target at index {i}")

def load_dataset_from_pickle(pickle_file):
    with open(pickle_file, 'rb') as f:
        dataset = pickle.load(f)
    if isinstance(dataset, list) and all(isinstance(d, Data) for d in dataset):
        return dataset
    else:
        raise ValueError("The loaded dataset is not a list of Data objects).")

if __name__ == "__main__":    
    train_pickle_file = 'GNN/data/Predict2/train_dataset.pkl'
    dataset = load_dataset_from_pickle(train_pickle_file)

    train_index = int(len(dataset) * 0.95)
    train_dataset = dataset[:train_index]
    val_dataset = dataset[train_index:]

    check_for_nan(train_dataset)


    batch_size = 20
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    num_features = train_dataset[0].x.shape[1]  # Number of features per node
    num_targets = train_dataset[0].y.shape[0]

    model = gnn_model.GCN(num_features, num_targets)
    
    learning_rate = 0.001 #1e-3
    num_epochs = 25

    output_filepath = f'GNN/models/TER_Abs_model_b{batch_size}_e{num_epochs}_lr{learning_rate}.pth'

    print("___________________________________")
    print()
    print("Learning Rate:", learning_rate)
    print("Batch Size:", batch_size)
    print("Epochs:", num_epochs )
    print(f'Number of training graphs: {len(train_dataset)}')
    print(f'Number of test graphs: {len(val_dataset)}')
    print(f'Number of features: {num_features}')
    print(f'Number of targets: {num_targets}')
    print("___________________________________")

    # data = dataset[0] 
    # print()
    # print(data)
    # print('=============================================================')
    # print(f'Number of nodes: {data.num_nodes}')
    # print(f'Number of edges: {data.num_edges}')
    # print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    # print(f'Has isolated nodes: {data.has_isolated_nodes()}')
    # print(f'Has self-loops: {data.has_self_loops()}')
    # print(f'Is undirected: {data.is_undirected()}')
    # print('=============================================================')
    # print()
    # print(f'Number of training graphs: {len(train_dataset)}')
    # print(f'Number of test graphs: {len(val_dataset)}')
    # print('=============================================================')
    # print()
    # # for step, data in enumerate(train_loader):
    # #     print(f'Step {step + 1}:')
    # #     print('=======')
    # #     print(f'Number of graphs in the current batch: {data.num_graphs}')
    # #     print(data)
    # #     print()
    # # print()
    # exit()

    
    # train_model.train_model(train_loader, val_loader, model, output_filepath, learning_rate, num_epochs)

    # TER
    test_pickle_file = 'GNN/data/Predict2/test_dataset.pkl'
    model_path = 'GNN/models/both_Abs_model_b20_e100_lr0.001.pth'
    
    # VEGF
    # test_pickle_file = 'GNN/data/Predict1/VEGF/Test_VEGF.pkl'
    # model_path = 'GNN/models/VEGF_Abs_model_b20_e25_lr0.001.pth'

    # test_pickle_file = 'GNN/data/Predict2/test_dataset.pkl'
    # model_path = 'GNN/models/Abs_model_b20_e100_lr0.001.pth'

    test_dataset = load_dataset_from_pickle(test_pickle_file)
    test_loader = DataLoader(test_dataset)

    num_features = test_dataset[0].x.shape[1]  # Number of features per node
    num_targets = test_dataset[0].y.shape[0]

    model = gnn_model.GCN(num_features, num_targets)
    model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
    
    test_acc.test_model(test_loader, model)



