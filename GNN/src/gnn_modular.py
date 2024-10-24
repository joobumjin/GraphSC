import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, BatchNorm
from torch.nn import ModuleList, Linear, Dropout

class Modular_GCN(torch.nn.Module):
    def __init__(self, num_node_features, output_dim, num_dense = 2, num_gcn = 3, hidden_channels=128, num_heads=8):
        super().__init__()
        self.num_node_features = num_node_features
        self.output_dim = output_dim
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads

        edge_dim = 0 if output_dim == 1 else 1

        gat_layers = [
            GATv2Conv(self.num_node_features, self.hidden_channels, heads=self.num_heads, concat=True, edge_dim=edge_dim),
        ]
        b_norm_layers = [BatchNorm(self.hidden_channels * self.num_heads),]
        for _ in range(num_gcn - 2):
            gat_layers.append(GATv2Conv(self.hidden_channels * self.num_heads, self.hidden_channels, heads=self.num_heads, concat=True, edge_dim=edge_dim))
            b_norm_layers.append(BatchNorm(self.hidden_channels * self.num_heads))
        gat_layers.append(GATv2Conv(self.hidden_channels * self.num_heads, self.output_dim, heads=1, concat=False, edge_dim=edge_dim))
        b_norm_layers.append(BatchNorm(self.output_dim))

        self.gats = ModuleList(gat_layers)
        self.b_norms = ModuleList(b_norm_layers)

        self.dropout = Dropout(p=0.5)
        dense_layers = [Linear(self.output_dim, self.hidden_channels)]
        dense_layers = dense_layers + [Linear(self.hidden_channels, self.hidden_channels) for _ in range(num_dense - 2)]
        dense_layers.append(Linear(self.hidden_channels, self.output_dim)) #for donor prediction, try both output dim of 1 and 2

        self.dense_head = ModuleList(dense_layers)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for gat, b_norm in zip(self.gats, self.b_norms):
          x = gat(x, edge_index)
          x = b_norm(x)
          x = F.leaky_relu(x)

        x = global_mean_pool(x, batch)

        for dense in self.dense_head:
          x = dense(x)
          x = F.leaky_relu(x)
          x = self.dropout(x)

        return x