import torch
import torch.nn.functional as F
from torch.nn import Linear, Dropout, Identity

class GCN_Merge(torch.nn.Module):
    def __init__(self, gnn1, gnn2):
        super().__init__()
        self.gnn1 = gnn1
        self.gnn2 = gnn2

        for param in gnn1.parameters():
            param.requires_grad = False
        for param in gnn2.parameters():
            param.requires_grad = False

        self.gnn1.out_linear = Identity()
        self.gnn2.out_linear = Identity()

        self.dropout = Dropout(p=0.5)

        self.pred_head = Linear(gnn1.dense_hidden * 2, 1)
        for param in self.pred_head.parameters():
            param.requires_grad = False

    def forward(self, dual_data):
        data_1, data_2 = dual_data.x1, dual_data.x2
        emb1, emb2 = self.gnn1(data_1), self.gnn2(data_2)

        x = torch.cat((emb1, emb2), dim=-1)
        x = self.dropout(x)
        x = self.pred_head(x)
        x = F.sigmoid(x)

        return x