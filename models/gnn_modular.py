import torch
import torch.nn.functional as F
import torch_geometric.nn as geom
from torch_geometric.nn import GraphConv, GCNConv, GATConv, GATv2Conv, global_mean_pool, BatchNorm
import torch.nn as nn
from torch.nn import Linear, Dropout, LeakyReLU, Identity
import inspect

def get_remaining_params(args, cls):
    signature = inspect.signature(cls.__init__)
    
    parameters = [param.name for param in signature.parameters.values() if param.name != 'self']

    return {key:args[key] for key in set(args.keys()).intersection(parameters)}

class Modular_GNN(torch.nn.Module):
    def __init__(self, num_node_features, output_dim,  num_dense = 2, num_gcn = 3, hidden_channels=128, dense_hidden=128, num_heads=8, dropout_p=0.5, gnn_layer = GATv2Conv):
      super().__init__()

      assert num_dense >= 2 and num_gcn >= 2

      self.num_node_features = num_node_features
      self.output_dim = output_dim
      self.dense_hidden = dense_hidden
      self.hidden_channels = hidden_channels
      self.num_heads = num_heads
      edge_dim = 0 if output_dim == 1 else 1

      args = {"in_channels": self.num_node_features, 
              "out_channels": self.hidden_channels, 
              "heads": self.num_heads, 
              "edge_dim": edge_dim, 
              "concat": True,}
      
      args = get_remaining_params(args, gnn_layer)
      if "heads" not in args.keys(): self.num_heads = 1

      #build GAT Network
      gat_layers = [
          (gnn_layer(**args), 'x, edge_index -> x'),
          (BatchNorm(self.hidden_channels * self.num_heads), 'x -> x'),
          LeakyReLU(inplace=True)
      ] #input layer

      args["in_channels"] = self.hidden_channels * self.num_heads 
      for _ in range(num_gcn - 2):
          gat_layers += [
            (gnn_layer(**args), 'x, edge_index -> x'),
            (BatchNorm(self.hidden_channels * self.num_heads), 'x -> x'),
            LeakyReLU(inplace=True)
          ]
    
      args["out_channels"] = self.output_dim 
      if "heads" in args.keys(): 
        args["heads"] = 1
        args["concat"] = False
      gat_layers += [
          (gnn_layer(**args), 'x, edge_index -> x'),
          (BatchNorm(self.output_dim), 'x -> x')
      ]

      self.gat_net = geom.Sequential('x, edge_index', gat_layers)

      #Build Dense Prediction Head
      dense_layers = [
        Linear(self.output_dim, self.dense_hidden),
        LeakyReLU(inplace=True),
        Dropout(p=dropout_p)
      ]
      for _ in range(num_dense - 2):
        dense_layers += [
          Linear(self.dense_hidden, self.dense_hidden),
          LeakyReLU(inplace=True),
          Dropout(p=dropout_p)
        ]
      dense_layers.append(Linear(self.dense_hidden, self.output_dim))

      self.dense_head = nn.Sequential(*dense_layers)

    def forward(self, data):
      x, edge_index, batch = data.x, data.edge_index, data.batch

      x = self.gat_net(x, edge_index)

      x = global_mean_pool(x, batch)

      x = self.dense_head(x)

      return x
    
    def guillotine_last(self):
      self.dense_head[-1] = Identity()

    def extend_dense(self, layers):
       self.dense_head.extend(nn.Sequential(layers))