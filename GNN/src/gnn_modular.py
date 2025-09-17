import torch
import torch.nn.functional as F
import torch_geometric as geom
from torch_geometric.nn import GATv2Conv, global_mean_pool, BatchNorm
import torch.nn as nn
from torch.nn import Linear, Dropout, LeakyReLU

class Modular_GNN(torch.nn.Module):
    def __init__(self, num_node_features, output_dim,  num_dense = 2, num_gcn = 3, hidden_channels=128, dense_hidden=128, num_heads=8, dropout_p=0.5):
        super().__init__()
        self.num_node_features = num_node_features
        self.output_dim = output_dim
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads

        edge_dim = 0 if output_dim == 1 else 1

        #build GAT Network
        gat_layers = [
            (GATv2Conv(self.num_node_features, self.hidden_channels, heads=self.num_heads, concat=True, edge_dim=edge_dim), 'x, edge_index -> x'),
            (BatchNorm(self.hidden_channels * self.num_heads), 'x -> x'),
            LeakyReLU(inplace=True)
        ] #input layer

        for _ in range(num_gcn - 2):
            gat_layers += [
              (GATv2Conv(self.hidden_channels * self.num_heads, self.hidden_channels, heads=self.num_heads, concat=True, edge_dim=edge_dim), 'x, edge_index -> x'),
              (BatchNorm(self.hidden_channels * self.num_heads), 'x -> x'),
              LeakyReLU(inplace=True)
            ]
      
        gat_layers += [
            (GATv2Conv(self.hidden_channels * self.num_heads, self.output_dim, heads=1, concat=False, edge_dim=edge_dim), 'x, edge_index -> x'),
            (BatchNorm(self.output_dim), 'x -> x'),
            LeakyReLU(inplace=True)

        ]

        self.gat_net = geom.Sequential('x, edge_index -> x', gat_layers)

        #Build Dense Prediction Head
        dense_layers = [
          Linear(self.output_dim, self.hidden_channels),
          LeakyReLU(inplace=True),
          Dropout(p=dropout_p)
        ]
        for _ in range(num_dense - 2):
          dense_layers += [
            Linear(self.hidden_channels, self.hidden_channels),
            LeakyReLU(inplace=True),
            Dropout(p=dropout_p)
          ]
        dense_layers.append(Linear(self.hidden_channels, self.output_dim))

        self.dense_head = nn.Sequential(*dense_layers)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.gat_net(x, edge_index)

        x = global_mean_pool(x, batch)

        x = self.dense_head(x)

        return x