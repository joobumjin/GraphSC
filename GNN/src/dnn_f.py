import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, BatchNorm, JumpingKnowledge
from torch.nn import Conv2d, MaxPool2d, ModuleList, Linear, Dropout

class InceptionLayer(torch.nn.Module):
    def __init__(self, channel_dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.oneConv = Conv2d(channel_dict["in"], channel_dict["oneConv"], (1,1))

        self.threeConv_1 = Conv2d(channel_dict["in"], channel_dict["threeConv_1"], (1,1), padding="same")
        self.threeConv_2 = Conv2d(channel_dict["threeConv_1"], channel_dict["threeConv_2"], (3,3), padding="same")

        self.fiveConv_1 = Conv2d(channel_dict["in"], channel_dict["fiveConv_1"], (1,1), padding="same")
        self.fiveConv_2 = Conv2d(channel_dict["fiveConv_1"], channel_dict["fiveConv_2"], (5,5), padding="same")

        self.poolConv_1 = MaxPool2d((3,3), padding="same")
        self.poolConv_2 = Conv2d(channel_dict["in"], channel_dict["poolConv_2"], (5,5), padding="same")

    def forward(self, data):
        oneConv_out = self.conv_1(data)

        threeConv_out = self.threeConv_1(data)
        threeConv_out = self.threeConv_2(threeConv_out)

        fiveConv_out = self.fiveConv_1(data)
        fiveConv_out = self.fiveConv_2(fiveConv_out)

        poolConv_out = self.poolConv_1(data)
        poolConv_out = self.poolConv_2(poolConv_out)

        return torch.cat((oneConv_out, threeConv_out, fiveConv_out, poolConv_out), dim=1)

class DNN_F(torch.nn.Module):
    def __init__(self, num_node_features, output_dim):
        super().__init__()
        self.num_node_features = num_node_features
        self.output_dim = output_dim
        self.edge_dim = 0 if output_dim == 1 else 1

        self.layers = ModuleList([])


    def forward(self, data):
        pass