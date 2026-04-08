from torch import nn
from .layers import QHGNN_conv
import torch.nn.functional as F


class QHGNN(nn.Module):
    def __init__(self, in_ch, n_class, n_hid, dropout=0.1):
        super(QHGNN, self).__init__()
        self.dropout = dropout
        # Dropout means we randomly zero out some node features during training, 
        # This means the model can't over-rely on any single feature.
        self.hgc1 = QHGNN_conv(in_ch, n_hid)
        # in_ch is the number of input features per node, n_hid is the number of hidden nodes
        self.hgc2 = QHGNN_conv(n_hid, n_class)

    def forward(self, x, LS, RS, Q):
        edge_nodes = QHGNN_conv._edge_node_indices(LS)
        x = F.relu(self.hgc1(x, LS, Q, RS, edge_nodes=edge_nodes))
        x = F.dropout(x, self.dropout)
        x = self.hgc2(x, LS, Q, RS, edge_nodes=edge_nodes)
        return x

