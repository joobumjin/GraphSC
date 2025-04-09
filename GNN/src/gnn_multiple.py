import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, BatchNorm, JumpingKnowledge
from torch.nn import Linear, Dropout

class GCN_G2_D2(torch.nn.Module):
    def __init__(self, num_node_features, output_dim, hidden_channels=128, dense_hidden=128, num_heads=8, dropout_p=0.5):
        super().__init__()
        self.num_node_features = num_node_features
        self.output_dim = output_dim
        self.hidden_channels = hidden_channels
        self.dense_hidden = dense_hidden
        self.num_heads = num_heads
        self.edge_dim = 0 if output_dim == 1 else 1

        self.conv1 = GATv2Conv(self.num_node_features, self.hidden_channels, heads=self.num_heads, concat=True, edge_dim=self.edge_dim)
        self.norm1 = BatchNorm(self.hidden_channels * self.num_heads)
        
        self.conv2 = GATv2Conv(self.hidden_channels * self.num_heads, self.output_dim, heads=1, concat=False, edge_dim=self.edge_dim)
        self.norm2 = BatchNorm(self.output_dim)
        
        self.dropout = Dropout(p=dropout_p)

        self.linear1 = Linear(self.output_dim, self.dense_hidden)
        self.linear2 = Linear(self.dense_hidden, self.output_dim)


    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = F.leaky_relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        x = F.leaky_relu(x)

        x = global_mean_pool(x, batch)

        x = F.leaky_relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)

        return x

class GCN_G2_D3(torch.nn.Module):
    def __init__(self, num_node_features, output_dim, hidden_channels=128, dense_hidden=128, num_heads=8, dropout_p=0.5):
        super().__init__()
        self.num_node_features = num_node_features
        self.output_dim = output_dim
        self.hidden_channels = hidden_channels
        self.dense_hidden = dense_hidden
        self.num_heads = num_heads
        self.edge_dim = 0 if output_dim == 2 else 1

        self.conv1 = GATv2Conv(self.num_node_features, self.hidden_channels, heads=self.num_heads, concat=True, edge_dim=self.edge_dim)
        self.norm1 = BatchNorm(self.hidden_channels * self.num_heads)

        self.conv2 = GATv2Conv(self.hidden_channels * self.num_heads, self.output_dim, heads=1, concat=False, edge_dim=self.edge_dim)
        self.norm2 = BatchNorm(self.output_dim)
        
        self.dropout = Dropout(p=dropout_p)

        self.linear1 = Linear(self.output_dim, self.dense_hidden)
        self.linear2 = Linear(self.dense_hidden, self.dense_hidden)
        self.linear3 = Linear(self.dense_hidden, self.output_dim)


    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = F.leaky_relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        x = F.leaky_relu(x)

        x = global_mean_pool(x, batch)

        x = F.leaky_relu(self.linear1(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.linear2(x))
        x = self.dropout(x)
        x = self.linear3(x)

        return x
    
class GCN_G2_D4(torch.nn.Module):
    def __init__(self, num_node_features, output_dim, hidden_channels=128, dense_hidden=128, num_heads=8, dropout_p=0.5):
        super().__init__()
        self.num_node_features = num_node_features
        self.output_dim = output_dim
        self.hidden_channels = hidden_channels
        self.dense_hidden = dense_hidden
        self.num_heads = num_heads
        self.edge_dim = 0 if output_dim == 2 else 1

        self.conv1 = GATv2Conv(self.num_node_features, self.hidden_channels, heads=self.num_heads, concat=True, edge_dim=self.edge_dim)
        self.norm1 = BatchNorm(self.hidden_channels * self.num_heads)
        
        self.conv2 = GATv2Conv(self.hidden_channels * self.num_heads, self.output_dim, heads=1, concat=False, edge_dim=self.edge_dim)
        self.norm2 = BatchNorm(self.output_dim)
        
        self.dropout = Dropout(p=dropout_p)

        self.linear1 = Linear(self.output_dim, self.dense_hidden)
        self.linear2 = Linear(self.dense_hidden, self.dense_hidden)
        self.linear3 = Linear(self.dense_hidden, self.dense_hidden)
        self.linear4 = Linear(self.dense_hidden, self.output_dim)


    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = self.dropout(x)
        x = F.leaky_relu(x)

        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        x = F.leaky_relu(x)

        x = global_mean_pool(x, batch)

        x = F.leaky_relu(self.linear1(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.linear2(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.linear3(x))
        x = self.dropout(x)
        x = self.linear4(x)

        return x
    
class GCN_G2_D5(torch.nn.Module):
    def __init__(self, num_node_features, output_dim, hidden_channels=128, dense_hidden=128, num_heads=8, dropout_p=0.5):
        super().__init__()
        self.num_node_features = num_node_features
        self.output_dim = output_dim
        self.hidden_channels = hidden_channels
        self.dense_hidden = dense_hidden
        self.num_heads = num_heads
        self.edge_dim = 0 if output_dim == 2 else 1

        self.conv1 = GATv2Conv(self.num_node_features, self.hidden_channels, heads=self.num_heads, concat=True, edge_dim=self.edge_dim)
        self.norm1 = BatchNorm(self.hidden_channels * self.num_heads)
        
        self.conv2 = GATv2Conv(self.hidden_channels * self.num_heads, self.output_dim, heads=1, concat=False, edge_dim=self.edge_dim)
        self.norm2 = BatchNorm(self.output_dim)
        
        self.dropout = Dropout(p=dropout_p)

        self.linear1 = Linear(self.output_dim, self.dense_hidden)
        self.linear2 = Linear(self.dense_hidden, self.dense_hidden)
        self.linear3 = Linear(self.dense_hidden, self.dense_hidden)
        self.linear4 = Linear(self.dense_hidden, self.dense_hidden)
        self.linear5 = Linear(self.dense_hidden, self.output_dim)


    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = self.dropout(x)
        x = self.norm1(x)
        x = F.leaky_relu(x)

        x = self.conv2(x, edge_index)
        x = self.dropout(x)
        x = self.norm2(x)
        x = F.leaky_relu(x)

        x = global_mean_pool(x, batch)

        x = F.leaky_relu(self.linear1(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.linear2(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.linear3(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.linear4(x))
        x = self.dropout(x)
        x = self.linear5(x)

        return x

######################################################################################################################

class GCN_G3_D2(torch.nn.Module):
    def __init__(self, num_node_features, output_dim, hidden_channels=128, dense_hidden=128, num_heads=8, dropout_p=0.5):
        super().__init__()
        self.num_node_features = num_node_features
        self.output_dim = output_dim
        self.hidden_channels = hidden_channels
        self.dense_hidden = dense_hidden
        self.num_heads = num_heads
        self.edge_dim = 0 if output_dim == 2 else 1

        self.conv1 = GATv2Conv(self.num_node_features, self.hidden_channels, heads=self.num_heads, concat=True, edge_dim=self.edge_dim)
        self.norm1 = BatchNorm(self.hidden_channels * self.num_heads)

        self.conv2 = GATv2Conv(self.hidden_channels * self.num_heads, self.hidden_channels, heads=self.num_heads, concat=True, edge_dim=self.edge_dim)
        self.norm2 = BatchNorm(self.hidden_channels * self.num_heads)
        
        self.conv3 = GATv2Conv(self.hidden_channels * self.num_heads, self.output_dim, heads=1, concat=False, edge_dim=self.edge_dim)
        self.norm3 = BatchNorm(self.output_dim)
        
        self.dropout = Dropout(p=dropout_p)

        self.linear1 = Linear(self.output_dim, self.dense_hidden)
        self.linear2 = Linear(self.dense_hidden, self.output_dim)


    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = self.dropout(x)
        x = self.norm1(x)
        x = F.leaky_relu(x)

        x = self.conv2(x, edge_index)
        x = self.dropout(x)
        x = self.norm2(x)
        x = F.leaky_relu(x)

        x = self.conv3(x, edge_index)
        x = self.norm3(x)
        x = F.leaky_relu(x)

        x = global_mean_pool(x, batch)

        x = F.leaky_relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)

        return x

class GCN_G3_D3(torch.nn.Module):
    def __init__(self, num_node_features, output_dim, hidden_channels=128, dense_hidden=128, num_heads=8, dropout_p=0.5):
        super().__init__()
        self.num_node_features = num_node_features
        self.output_dim = output_dim
        self.hidden_channels = hidden_channels
        self.dense_hidden = dense_hidden
        self.num_heads = num_heads
        self.edge_dim = 0 if output_dim == 2 else 1

        self.conv1 = GATv2Conv(self.num_node_features, self.hidden_channels, heads=self.num_heads, concat=True, edge_dim=self.edge_dim)
        self.norm1 = BatchNorm(self.hidden_channels * self.num_heads)

        self.conv2 = GATv2Conv(self.hidden_channels * self.num_heads, self.hidden_channels, heads=self.num_heads, concat=True, edge_dim=self.edge_dim)
        self.norm2 = BatchNorm(self.hidden_channels * self.num_heads)
        
        self.conv3 = GATv2Conv(self.hidden_channels * self.num_heads, self.output_dim, heads=1, concat=False, edge_dim=self.edge_dim)
        self.norm3 = BatchNorm(self.output_dim)
        
        self.dropout = Dropout(p=dropout_p)

        self.linear1 = Linear(self.output_dim, self.dense_hidden)
        self.linear2 = Linear(self.dense_hidden, self.dense_hidden)
        self.linear3 = Linear(self.dense_hidden, self.output_dim)


    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = self.dropout(x)
        x = self.norm1(x)
        x = F.leaky_relu(x)

        x = self.conv2(x, edge_index)
        x = self.dropout(x)
        x = self.norm2(x)
        x = F.leaky_relu(x)

        x = self.conv3(x, edge_index)
        x = self.dropout(x)
        x = self.norm3(x)
        x = F.leaky_relu(x)

        x = global_mean_pool(x, batch)

        x = F.leaky_relu(self.linear1(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.linear2(x))
        x = self.dropout(x)
        x = self.linear3(x)

        return x
    
class GCN_G3_D4(torch.nn.Module):
    def __init__(self, num_node_features, output_dim, hidden_channels=128, dense_hidden=128, num_heads=8, dropout_p=0.5):
        super().__init__()
        self.num_node_features = num_node_features
        self.output_dim = output_dim
        self.hidden_channels = hidden_channels
        self.dense_hidden = dense_hidden
        self.num_heads = num_heads
        self.edge_dim = 0 if output_dim == 2 else 1

        self.conv1 = GATv2Conv(self.num_node_features, self.hidden_channels, heads=self.num_heads, concat=True, edge_dim=self.edge_dim)
        self.norm1 = BatchNorm(self.hidden_channels * self.num_heads)

        self.conv2 = GATv2Conv(self.hidden_channels * self.num_heads, self.hidden_channels, heads=self.num_heads, concat=True, edge_dim=self.edge_dim)
        self.norm2 = BatchNorm(self.hidden_channels * self.num_heads)
        
        self.conv3 = GATv2Conv(self.hidden_channels * self.num_heads, self.output_dim, heads=1, concat=False, edge_dim=self.edge_dim)
        self.norm3 = BatchNorm(self.output_dim)
        
        self.dropout = Dropout(p=dropout_p)

        self.linear1 = Linear(self.output_dim, self.dense_hidden)
        self.linear2 = Linear(self.dense_hidden, self.dense_hidden)
        self.linear3 = Linear(self.dense_hidden, self.dense_hidden)
        self.linear4 = Linear(self.dense_hidden, self.output_dim)


    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = self.dropout(x)
        x = self.norm1(x)
        x = F.leaky_relu(x)

        x = self.conv2(x, edge_index)
        x = self.dropout(x)
        x = self.norm2(x)
        x = F.leaky_relu(x)

        x = self.conv3(x, edge_index)
        x = self.dropout(x)
        x = self.norm3(x)
        x = F.leaky_relu(x)

        x = global_mean_pool(x, batch)

        x = F.leaky_relu(self.linear1(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.linear2(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.linear3(x))
        x = self.dropout(x)
        x = self.linear4(x)

        return x
    
class GCN_G3_D5(torch.nn.Module):
    def __init__(self, num_node_features, output_dim, hidden_channels=128, dense_hidden=128, num_heads=8, dropout_p=0.5):
        super().__init__()
        self.num_node_features = num_node_features
        self.output_dim = output_dim
        self.hidden_channels = hidden_channels
        self.dense_hidden = dense_hidden
        self.num_heads = num_heads
        self.edge_dim = 0 if output_dim == 2 else 1

        self.conv1 = GATv2Conv(self.num_node_features, self.hidden_channels, heads=self.num_heads, concat=True, edge_dim=self.edge_dim)
        self.norm1 = BatchNorm(self.hidden_channels * self.num_heads)

        self.conv2 = GATv2Conv(self.hidden_channels * self.num_heads, self.hidden_channels, heads=self.num_heads, concat=True, edge_dim=self.edge_dim)
        self.norm2 = BatchNorm(self.hidden_channels * self.num_heads)
        
        self.conv3 = GATv2Conv(self.hidden_channels * self.num_heads, self.output_dim, heads=1, concat=False, edge_dim=self.edge_dim)
        self.norm3 = BatchNorm(self.output_dim)
        
        self.dropout = Dropout(p=dropout_p)

        self.linear1 = Linear(self.output_dim, self.dense_hidden)
        self.linear2 = Linear(self.dense_hidden, self.dense_hidden)
        self.linear3 = Linear(self.dense_hidden, self.dense_hidden)
        self.linear4 = Linear(self.dense_hidden, self.dense_hidden)
        self.linear5 = Linear(self.dense_hidden, self.output_dim)


    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = self.dropout(x)
        x = self.norm1(x)
        x = F.leaky_relu(x)

        x = self.conv2(x, edge_index)
        x = self.dropout(x)
        x = self.norm2(x)
        x = F.leaky_relu(x)

        x = self.conv3(x, edge_index)
        x = self.dropout(x)
        x = self.norm3(x)
        x = F.leaky_relu(x)

        x = global_mean_pool(x, batch)

        x = F.leaky_relu(self.linear1(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.linear2(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.linear3(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.linear4(x))
        x = self.dropout(x)
        x = self.linear5(x)

        return x

######################################################################################################################

class GCN_G4_D2(torch.nn.Module):
    def __init__(self, num_node_features, output_dim, hidden_channels=128, dense_hidden=128, num_heads=8, dropout_p=0.5):
        super().__init__()
        self.num_node_features = num_node_features
        self.output_dim = output_dim
        self.hidden_channels = hidden_channels
        self.dense_hidden = dense_hidden
        self.num_heads = num_heads
        self.edge_dim = 0 if output_dim == 2 else 1

        self.conv1 = GATv2Conv(self.num_node_features, self.hidden_channels, heads=self.num_heads, concat=True, edge_dim=self.edge_dim)
        self.norm1 = BatchNorm(self.hidden_channels * self.num_heads)

        self.conv2 = GATv2Conv(self.hidden_channels * self.num_heads, self.hidden_channels, heads=self.num_heads, concat=True, edge_dim=self.edge_dim)
        self.norm2 = BatchNorm(self.hidden_channels * self.num_heads)

        self.conv3 = GATv2Conv(self.hidden_channels * self.num_heads, self.hidden_channels, heads=self.num_heads, concat=True, edge_dim=self.edge_dim)
        self.norm3 = BatchNorm(self.hidden_channels * self.num_heads)
        
        self.conv4 = GATv2Conv(self.hidden_channels * self.num_heads, self.output_dim, heads=1, concat=False, edge_dim=self.edge_dim)
        self.norm4 = BatchNorm(self.output_dim)
        
        self.dropout = Dropout(p=dropout_p)

        self.linear1 = Linear(self.output_dim, self.dense_hidden)
        self.linear2 = Linear(self.dense_hidden, self.output_dim)


    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = self.dropout(x)
        x = self.norm1(x)
        x = F.leaky_relu(x)

        x = self.conv2(x, edge_index)
        x = self.dropout(x)
        x = self.norm2(x)
        x = F.leaky_relu(x)

        x = self.conv3(x, edge_index)
        x = self.dropout(x)
        x = self.norm3(x)
        x = F.leaky_relu(x)

        x = self.conv4(x, edge_index)
        x = self.dropout(x)
        x = self.norm4(x)
        x = F.leaky_relu(x)

        x = global_mean_pool(x, batch)

        x = F.leaky_relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)

        return x

class GCN_G4_D3(torch.nn.Module):
    def __init__(self, num_node_features, output_dim, hidden_channels=128, dense_hidden=128, num_heads=8, dropout_p=0.5):
        super().__init__()
        self.num_node_features = num_node_features
        self.output_dim = output_dim
        self.hidden_channels = hidden_channels
        self.dense_hidden = dense_hidden
        self.num_heads = num_heads
        self.edge_dim = 0 if output_dim == 2 else 1

        self.conv1 = GATv2Conv(self.num_node_features, self.hidden_channels, heads=self.num_heads, concat=True, edge_dim=self.edge_dim)
        self.norm1 = BatchNorm(self.hidden_channels * self.num_heads)

        self.conv2 = GATv2Conv(self.hidden_channels * self.num_heads, self.hidden_channels, heads=self.num_heads, concat=True, edge_dim=self.edge_dim)
        self.norm2 = BatchNorm(self.hidden_channels * self.num_heads)

        self.conv3 = GATv2Conv(self.hidden_channels * self.num_heads, self.hidden_channels, heads=self.num_heads, concat=True, edge_dim=self.edge_dim)
        self.norm3 = BatchNorm(self.hidden_channels * self.num_heads)
        
        self.conv4 = GATv2Conv(self.hidden_channels * self.num_heads, self.output_dim, heads=1, concat=False, edge_dim=self.edge_dim)
        self.norm4 = BatchNorm(self.output_dim)
        
        self.dropout = Dropout(p=dropout_p)

        self.linear1 = Linear(self.output_dim, self.dense_hidden)
        self.linear2 = Linear(self.dense_hidden, self.dense_hidden)
        self.linear3 = Linear(self.dense_hidden, self.output_dim)


    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = self.dropout(x)
        x = self.norm1(x)
        x = F.leaky_relu(x)

        x = self.conv2(x, edge_index)
        x = self.dropout(x)
        x = self.norm2(x)
        x = F.leaky_relu(x)

        x = self.conv3(x, edge_index)
        x = self.dropout(x)
        x = self.norm3(x)
        x = F.leaky_relu(x)

        x = self.conv4(x, edge_index)
        x = self.dropout(x)
        x = self.norm4(x)
        x = F.leaky_relu(x)

        x = global_mean_pool(x, batch)

        x = F.leaky_relu(self.linear1(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.linear2(x))
        x = self.dropout(x)
        x = self.linear3(x)

        return x
    
class GCN_G4_D4(torch.nn.Module):
    def __init__(self, num_node_features, output_dim, hidden_channels=128, dense_hidden=128, num_heads=8, dropout_p=0.5):
        super().__init__()
        self.num_node_features = num_node_features
        self.output_dim = output_dim
        self.hidden_channels = hidden_channels
        self.dense_hidden = dense_hidden
        self.num_heads = num_heads
        self.edge_dim = 0 if output_dim == 2 else 1

        self.conv1 = GATv2Conv(self.num_node_features, self.hidden_channels, heads=self.num_heads, concat=True, edge_dim=self.edge_dim)
        self.norm1 = BatchNorm(self.hidden_channels * self.num_heads)

        self.conv2 = GATv2Conv(self.hidden_channels * self.num_heads, self.hidden_channels, heads=self.num_heads, concat=True, edge_dim=self.edge_dim)
        self.norm2 = BatchNorm(self.hidden_channels * self.num_heads)

        self.conv3 = GATv2Conv(self.hidden_channels * self.num_heads, self.hidden_channels, heads=self.num_heads, concat=True, edge_dim=self.edge_dim)
        self.norm3 = BatchNorm(self.hidden_channels * self.num_heads)
        
        self.conv4 = GATv2Conv(self.hidden_channels * self.num_heads, self.output_dim, heads=1, concat=False, edge_dim=self.edge_dim)
        self.norm4 = BatchNorm(self.output_dim)
        
        self.dropout = Dropout(p=dropout_p)

        self.linear1 = Linear(self.output_dim, self.dense_hidden)
        self.linear2 = Linear(self.dense_hidden, self.dense_hidden)
        self.linear3 = Linear(self.dense_hidden, self.dense_hidden)
        self.linear4 = Linear(self.dense_hidden, self.output_dim)


    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = self.dropout(x)
        x = self.norm1(x)
        x = F.leaky_relu(x)

        x = self.conv2(x, edge_index)
        x = self.dropout(x)
        x = self.norm2(x)
        x = F.leaky_relu(x)

        x = self.conv3(x, edge_index)
        x = self.dropout(x)
        x = self.norm3(x)
        x = F.leaky_relu(x)

        x = self.conv4(x, edge_index)
        x = self.dropout(x)
        x = self.norm4(x)
        x = F.leaky_relu(x)

        x = global_mean_pool(x, batch)

        x = F.leaky_relu(self.linear1(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.linear2(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.linear3(x))
        x = self.dropout(x)
        x = self.linear4(x)

        return x
    
class GCN_G4_D5(torch.nn.Module):
    def __init__(self, num_node_features, output_dim, hidden_channels=128, dense_hidden=128, num_heads=8, dropout_p=0.5):
        super().__init__()
        self.num_node_features = num_node_features
        self.output_dim = output_dim
        self.hidden_channels = hidden_channels
        self.dense_hidden = dense_hidden
        self.num_heads = num_heads
        self.edge_dim = 0 if output_dim == 2 else 1

        self.conv1 = GATv2Conv(self.num_node_features, self.hidden_channels, heads=self.num_heads, concat=True, edge_dim=self.edge_dim)
        self.norm1 = BatchNorm(self.hidden_channels * self.num_heads)

        self.conv2 = GATv2Conv(self.hidden_channels * self.num_heads, self.hidden_channels, heads=self.num_heads, concat=True, edge_dim=self.edge_dim)
        self.norm2 = BatchNorm(self.hidden_channels * self.num_heads)

        self.conv3 = GATv2Conv(self.hidden_channels * self.num_heads, self.hidden_channels, heads=self.num_heads, concat=True, edge_dim=self.edge_dim)
        self.norm3 = BatchNorm(self.hidden_channels * self.num_heads)
        
        self.conv4 = GATv2Conv(self.hidden_channels * self.num_heads, self.output_dim, heads=1, concat=False, edge_dim=self.edge_dim)
        self.norm4 = BatchNorm(self.output_dim)
        
        self.dropout = Dropout(p=dropout_p)

        self.linear1 = Linear(self.output_dim, self.dense_hidden)
        self.linear2 = Linear(self.dense_hidden, self.dense_hidden)
        self.linear3 = Linear(self.dense_hidden, self.dense_hidden)
        self.linear4 = Linear(self.dense_hidden, self.dense_hidden)
        self.linear5 = Linear(self.dense_hidden, self.output_dim)


    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = self.dropout(x)
        x = self.norm1(x)
        x = F.leaky_relu(x)

        x = self.conv2(x, edge_index)
        x = self.dropout(x)
        x = self.norm2(x)
        x = F.leaky_relu(x)

        x = self.conv3(x, edge_index)
        x = self.dropout(x)
        x = self.norm3(x)
        x = F.leaky_relu(x)

        x = self.conv4(x, edge_index)
        x = self.dropout(x)
        x = self.norm4(x)
        x = F.leaky_relu(x)

        x = global_mean_pool(x, batch)

        x = F.leaky_relu(self.linear1(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.linear2(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.linear3(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.linear4(x))
        x = self.dropout(x)
        x = self.linear5(x)

        return x
    
######################################################################################################################

class GCN_G5_D2(torch.nn.Module):
    def __init__(self, num_node_features, output_dim, hidden_channels=128, dense_hidden=128, num_heads=8, dropout_p=0.5):
        super().__init__()
        self.num_node_features = num_node_features
        self.output_dim = output_dim
        self.hidden_channels = hidden_channels
        self.dense_hidden = dense_hidden
        self.num_heads = num_heads
        self.edge_dim = 0 if output_dim == 2 else 1

        self.conv1 = GATv2Conv(self.num_node_features, self.hidden_channels, heads=self.num_heads, concat=True, edge_dim=self.edge_dim)
        self.norm1 = BatchNorm(self.hidden_channels * self.num_heads)

        self.conv2 = GATv2Conv(self.hidden_channels * self.num_heads, self.hidden_channels, heads=self.num_heads, concat=True, edge_dim=self.edge_dim)
        self.norm2 = BatchNorm(self.hidden_channels * self.num_heads)

        self.conv3 = GATv2Conv(self.hidden_channels * self.num_heads, self.hidden_channels, heads=self.num_heads, concat=True, edge_dim=self.edge_dim)
        self.norm3 = BatchNorm(self.hidden_channels * self.num_heads)

        self.conv4 = GATv2Conv(self.hidden_channels * self.num_heads, self.hidden_channels, heads=self.num_heads, concat=True, edge_dim=self.edge_dim)
        self.norm4 = BatchNorm(self.hidden_channels * self.num_heads)
        
        self.conv5 = GATv2Conv(self.hidden_channels * self.num_heads, self.output_dim, heads=1, concat=False, edge_dim=self.edge_dim)
        self.norm5 = BatchNorm(self.output_dim)
        
        self.dropout = Dropout(p=dropout_p)

        self.linear1 = Linear(self.output_dim, self.dense_hidden)
        self.linear2 = Linear(self.dense_hidden, self.output_dim)


    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = self.dropout(x)
        x = self.norm1(x)
        x = F.leaky_relu(x)

        x = self.conv2(x, edge_index)
        x = self.dropout(x)
        x = self.norm2(x)
        x = F.leaky_relu(x)

        x = self.conv3(x, edge_index)
        x = self.dropout(x)
        x = self.norm3(x)
        x = F.leaky_relu(x)

        x = self.conv4(x, edge_index)
        x = self.dropout(x)
        x = self.norm4(x)
        x = F.leaky_relu(x)
        x = self.conv5(x, edge_index)
        x = self.dropout(x)
        x = self.norm5(x)
        x = F.leaky_relu(x)

        x = global_mean_pool(x, batch)

        x = F.leaky_relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)

        return x

class GCN_G5_D3(torch.nn.Module):
    def __init__(self, num_node_features, output_dim, hidden_channels=128, dense_hidden=128, num_heads=8, dropout_p=0.5):
        super().__init__()
        self.num_node_features = num_node_features
        self.output_dim = output_dim
        self.hidden_channels = hidden_channels
        self.dense_hidden = dense_hidden
        self.num_heads = num_heads
        self.edge_dim = 0 if output_dim == 2 else 1

        self.conv1 = GATv2Conv(self.num_node_features, self.hidden_channels, heads=self.num_heads, concat=True, edge_dim=self.edge_dim)
        self.norm1 = BatchNorm(self.hidden_channels * self.num_heads)

        self.conv2 = GATv2Conv(self.hidden_channels * self.num_heads, self.hidden_channels, heads=self.num_heads, concat=True, edge_dim=self.edge_dim)
        self.norm2 = BatchNorm(self.hidden_channels * self.num_heads)

        self.conv3 = GATv2Conv(self.hidden_channels * self.num_heads, self.hidden_channels, heads=self.num_heads, concat=True, edge_dim=self.edge_dim)
        self.norm3 = BatchNorm(self.hidden_channels * self.num_heads)

        self.conv4 = GATv2Conv(self.hidden_channels * self.num_heads, self.hidden_channels, heads=self.num_heads, concat=True, edge_dim=self.edge_dim)
        self.norm4 = BatchNorm(self.hidden_channels * self.num_heads)
        
        self.conv5 = GATv2Conv(self.hidden_channels * self.num_heads, self.output_dim, heads=1, concat=False, edge_dim=self.edge_dim)
        self.norm5 = BatchNorm(self.output_dim)
        
        self.dropout = Dropout(p=dropout_p)

        self.linear1 = Linear(self.output_dim, self.dense_hidden)
        self.linear2 = Linear(self.dense_hidden, self.dense_hidden)
        self.linear3 = Linear(self.dense_hidden, self.output_dim)


    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = self.dropout(x)
        x = self.norm1(x)
        x = F.leaky_relu(x)

        x = self.conv2(x, edge_index)
        x = self.dropout(x)
        x = self.norm2(x)
        x = F.leaky_relu(x)

        x = self.conv3(x, edge_index)
        x = self.dropout(x)
        x = self.norm3(x)
        x = F.leaky_relu(x)

        x = self.conv4(x, edge_index)
        x = self.dropout(x)
        x = self.norm4(x)
        x = F.leaky_relu(x)

        x = self.conv5(x, edge_index)
        x = self.dropout(x)
        x = self.norm5(x)
        x = F.leaky_relu(x)

        x = global_mean_pool(x, batch)

        x = F.leaky_relu(self.linear1(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.linear2(x))
        x = self.dropout(x)
        x = self.linear3(x)

        return x
    
class GCN_G5_D4(torch.nn.Module):
    def __init__(self, num_node_features, output_dim, hidden_channels=128, dense_hidden=128, num_heads=8, dropout_p=0.5):
        super().__init__()
        self.num_node_features = num_node_features
        self.output_dim = output_dim
        self.hidden_channels = hidden_channels
        self.dense_hidden = dense_hidden
        self.num_heads = num_heads
        self.edge_dim = 0 if output_dim == 2 else 1

        self.conv1 = GATv2Conv(self.num_node_features, self.hidden_channels, heads=self.num_heads, concat=True, edge_dim=self.edge_dim)
        self.norm1 = BatchNorm(self.hidden_channels * self.num_heads)

        self.conv2 = GATv2Conv(self.hidden_channels * self.num_heads, self.hidden_channels, heads=self.num_heads, concat=True, edge_dim=self.edge_dim)
        self.norm2 = BatchNorm(self.hidden_channels * self.num_heads)

        self.conv3 = GATv2Conv(self.hidden_channels * self.num_heads, self.hidden_channels, heads=self.num_heads, concat=True, edge_dim=self.edge_dim)
        self.norm3 = BatchNorm(self.hidden_channels * self.num_heads)

        self.conv4 = GATv2Conv(self.hidden_channels * self.num_heads, self.hidden_channels, heads=self.num_heads, concat=True, edge_dim=self.edge_dim)
        self.norm4 = BatchNorm(self.hidden_channels * self.num_heads)
        
        self.conv5 = GATv2Conv(self.hidden_channels * self.num_heads, self.output_dim, heads=1, concat=False, edge_dim=self.edge_dim)
        self.norm5 = BatchNorm(self.output_dim)
        
        self.dropout = Dropout(p=dropout_p)

        self.linear1 = Linear(self.output_dim, self.dense_hidden)
        self.linear2 = Linear(self.dense_hidden, self.dense_hidden)
        self.linear3 = Linear(self.dense_hidden, self.dense_hidden)
        self.linear4 = Linear(self.dense_hidden, self.output_dim)


    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = self.dropout(x)
        x = self.norm1(x)
        x = F.leaky_relu(x)

        x = self.conv2(x, edge_index)
        x = self.dropout(x)
        x = self.norm2(x)
        x = F.leaky_relu(x)

        x = self.conv3(x, edge_index)
        x = self.dropout(x)
        x = self.norm3(x)
        x = F.leaky_relu(x)

        x = self.conv4(x, edge_index)
        x = self.dropout(x)
        x = self.norm4(x)
        x = F.leaky_relu(x)

        x = self.conv5(x, edge_index)
        x = self.dropout(x)
        x = self.norm5(x)
        x = F.leaky_relu(x)

        x = global_mean_pool(x, batch)

        x = F.leaky_relu(self.linear1(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.linear2(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.linear3(x))
        x = self.dropout(x)
        x = self.linear4(x)

        return x
    
class GCN_G5_D5(torch.nn.Module):
    def __init__(self, num_node_features, output_dim, hidden_channels=128, dense_hidden=128, num_heads=8, dropout_p=0.5):
        super().__init__()
        self.num_node_features = num_node_features
        self.output_dim = output_dim
        self.hidden_channels = hidden_channels
        self.dense_hidden = dense_hidden
        self.num_heads = num_heads
        self.edge_dim = 0 if output_dim == 2 else 1

        self.conv1 = GATv2Conv(self.num_node_features, self.hidden_channels, heads=self.num_heads, concat=True, edge_dim=self.edge_dim)
        self.norm1 = BatchNorm(self.hidden_channels * self.num_heads)

        self.conv2 = GATv2Conv(self.hidden_channels * self.num_heads, self.hidden_channels, heads=self.num_heads, concat=True, edge_dim=self.edge_dim)
        self.norm2 = BatchNorm(self.hidden_channels * self.num_heads)

        self.conv3 = GATv2Conv(self.hidden_channels * self.num_heads, self.hidden_channels, heads=self.num_heads, concat=True, edge_dim=self.edge_dim)
        self.norm3 = BatchNorm(self.hidden_channels * self.num_heads)

        self.conv4 = GATv2Conv(self.hidden_channels * self.num_heads, self.hidden_channels, heads=self.num_heads, concat=True, edge_dim=self.edge_dim)
        self.norm4 = BatchNorm(self.hidden_channels * self.num_heads)
        
        self.conv5 = GATv2Conv(self.hidden_channels * self.num_heads, self.output_dim, heads=1, concat=False, edge_dim=self.edge_dim)
        self.norm5 = BatchNorm(self.output_dim)
        
        self.dropout = Dropout(p=dropout_p)

        self.linear1 = Linear(self.output_dim, self.dense_hidden)
        self.linear2 = Linear(self.dense_hidden, self.dense_hidden)
        self.linear3 = Linear(self.dense_hidden, self.dense_hidden)
        self.linear4 = Linear(self.dense_hidden, self.dense_hidden)
        self.linear5 = Linear(self.dense_hidden, self.output_dim)


    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = self.dropout(x)
        x = self.norm1(x)
        x = F.leaky_relu(x)

        x = self.conv2(x, edge_index)
        x = self.dropout(x)
        x = self.norm2(x)
        x = F.leaky_relu(x)

        x = self.conv3(x, edge_index)
        x = self.dropout(x)
        x = self.norm3(x)
        x = F.leaky_relu(x)

        x = self.conv4(x, edge_index)
        x = self.dropout(x)
        x = self.norm4(x)
        x = F.leaky_relu(x)

        x = self.conv5(x, edge_index)
        x = self.dropout(x)
        x = self.norm5(x)
        x = F.leaky_relu(x)

        x = global_mean_pool(x, batch)

        x = F.leaky_relu(self.linear1(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.linear2(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.linear3(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.linear4(x))
        x = self.dropout(x)
        x = self.linear5(x)

        return x