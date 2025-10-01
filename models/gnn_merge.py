import torch
import torch.nn.functional as F
from torch.nn import ModuleList, Linear, Dropout, Identity
from collections import namedtuple

DataPoint = namedtuple('DataPoint', ['x', 'edge_index', 'batch'])

class GNN_Merge(torch.nn.Module):
    def __init__(self, gnn1, gnn2, output_dim=1, head_depth=1, freeze_gnns = False):
        super().__init__()
        self.gnn1 = gnn1
        self.gnn2 = gnn2
        self.output_dim = output_dim

        if freeze_gnns:
            for param in gnn1.parameters():
                param.requires_grad = False
            for param in gnn2.parameters():
                param.requires_grad = False

        self.gnn1.out_linear = Identity()
        self.gnn2.out_linear = Identity()

        self.dropout = Dropout(p=0.5)

        self.pred_head = ModuleList([Linear(gnn1.dense_hidden * 2, gnn1.dense_hidden * 2) for _ in range(head_depth-1)] + \
                                    [Linear(gnn1.dense_hidden * 2, self.output_dim)])

    def forward(self, pair_data):
        data_1, data_2 = DataPoint(pair_data.x1, pair_data.edge_index1, pair_data.x1_batch), \
                         DataPoint(pair_data.x2, pair_data.edge_index2, pair_data.x2_batch)
        
        emb1, emb2 = self.gnn1(data_1), self.gnn2(data_2)

        x = torch.cat((emb1, emb2), dim=-1)
        x = self.dropout(x)
        for layer in self.pred_head:
            x = layer(x)
        x = F.sigmoid(x)

        return x