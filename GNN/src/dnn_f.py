import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, BatchNorm, JumpingKnowledge
from torch.nn import Conv2d, MaxPool2d, AvgPool2d, BatchNorm2d, Sequential, Linear, Dropout

class SameMaxPool2d(torch.nn.Module):
    def __init__(self, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, data):
        # Calculate padding
        padding_h = (self.kernel_size - 1) // 2
        padding_w = (self.kernel_size - 1) // 2

        # Apply max pooling with calculated padding
        return torch.nn.functional.max_pool2d(data, self.kernel_size, self.stride, padding=(padding_h, padding_w))

class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        
        self.kernel_size = kernel_size
        self.conv = Conv2d(in_channels, out_channels, kernel_size, padding="same")
        self.batchnorm2d = BatchNorm2d(out_channels)

    def forward(self, data):
        x = self.conv(data)
        x = self.batchnorm2d(x)

        return F.relu(x)

class InceptionLayer(torch.nn.Module):
    def __init__(self, channel_dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.oneConv = Conv2d(channel_dict["in"], channel_dict["oneConv"], (1,1))

        self.threeConv_1 = Conv2d(channel_dict["in"], channel_dict["threeConv_1"], (1,1), padding="same")
        self.threeConv_2 = Conv2d(channel_dict["threeConv_1"], channel_dict["threeConv_2"], (3,3), padding="same")

        self.fiveConv_1 = Conv2d(channel_dict["in"], channel_dict["fiveConv_1"], (1,1), padding="same")
        self.fiveConv_2 = Conv2d(channel_dict["fiveConv_1"], channel_dict["fiveConv_2"], (5,5), padding="same")

        self.poolConv_1 = SameMaxPool2d(3)
        self.poolConv_2 = Conv2d(channel_dict["in"], channel_dict["poolConv_2"], (5,5), padding="same")

        self.batch_norm = BatchNorm2d(num_features=channel_dict["out"])

    def forward(self, data):
        oneConv_out = self.oneConv(data)

        threeConv_out = self.threeConv_1(data)
        threeConv_out = self.threeConv_2(threeConv_out)

        fiveConv_out = self.fiveConv_1(data)
        fiveConv_out = self.fiveConv_2(fiveConv_out)

        poolConv_out = self.poolConv_1(data)
        poolConv_out = self.poolConv_2(poolConv_out)

        catted = torch.cat((oneConv_out, threeConv_out, fiveConv_out, poolConv_out), dim=1)        
        return self.batch_norm(catted)

class DNN_F(torch.nn.Module):
    def __init__(self, output_dim):
        super().__init__()

        self.output_dim = output_dim

        self.layers = Sequential(
            Conv2d(3, 32, 3, padding="same"),
            SameMaxPool2d(3, 2),
            Conv2d(32, 32, 3, padding="same"),
            SameMaxPool2d(3, 2),
            Conv2d(32, 64, 3, padding="same"),
            SameMaxPool2d(3, 2),
            InceptionLayer(channel_dict={"in": 64, "oneConv": 32, "threeConv_1": 48, "threeConv_2": 64, "fiveConv_1": 8, "fiveConv_2": 16, "poolConv_2": 16, "out":128}),
            SameMaxPool2d(3, 2),
            InceptionLayer(channel_dict={"in": 128, "oneConv": 64, "threeConv_1": 96, "threeConv_2": 128, "fiveConv_1": 16, "fiveConv_2": 32, "poolConv_2": 32, "out":256}),
            SameMaxPool2d(3, 2),
            InceptionLayer(channel_dict={"in": 256, "oneConv": 192, "threeConv_1": 96, "threeConv_2": 208, "fiveConv_1": 16, "fiveConv_2": 48, "poolConv_2": 64, "out":512}),
            InceptionLayer(channel_dict={"in": 512, "oneConv": 160, "threeConv_1": 112, "threeConv_2": 224, "fiveConv_1": 24, "fiveConv_2": 64, "poolConv_2": 64, "out":512}),
            InceptionLayer(channel_dict={"in": 512, "oneConv": 160, "threeConv_1": 112, "threeConv_2": 224, "fiveConv_1": 24, "fiveConv_2": 64, "poolConv_2": 64, "out":512}),
            SameMaxPool2d(3, 2),
            InceptionLayer(channel_dict={"in": 512, "oneConv": 384, "threeConv_1": 192, "threeConv_2": 384, "fiveConv_1": 48, "fiveConv_2": 128, "poolConv_2": 128, "out":1024}),
            InceptionLayer(channel_dict={"in": 1024, "oneConv": 384, "threeConv_1": 192, "threeConv_2": 384, "fiveConv_1": 48, "fiveConv_2": 128, "poolConv_2": 128, "out":1024}),
            InceptionLayer(channel_dict={"in": 1024, "oneConv": 384, "threeConv_1": 192, "threeConv_2": 384, "fiveConv_1": 48, "fiveConv_2": 128, "poolConv_2": 128, "out":1024}),
            SameMaxPool2d(3, 2),
            InceptionLayer(channel_dict={"in": 1024, "oneConv": 384, "threeConv_1": 192, "threeConv_2": 384, "fiveConv_1": 48, "fiveConv_2": 128, "poolConv_2": 128, "out":1024}),
            InceptionLayer(channel_dict={"in": 1024, "oneConv": 384, "threeConv_1": 192, "threeConv_2": 384, "fiveConv_1": 48, "fiveConv_2": 128, "poolConv_2": 128, "out":1024}),
            InceptionLayer(channel_dict={"in": 1024, "oneConv": 384, "threeConv_1": 192, "threeConv_2": 384, "fiveConv_1": 48, "fiveConv_2": 128, "poolConv_2": 128, "out":1024}),
            SameMaxPool2d(3, 2),
            AvgPool2d((4,4)),
            Conv2d(1024, output_dim, (1,1), padding="same")
        )


    def forward(self, data):
        return F.relu(self.layers(data.x)).squeeze(dim=(2,3)) #squeeze width and height dimensions