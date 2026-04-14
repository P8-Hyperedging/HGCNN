from torch import nn
from .layers import QHGNN_conv_v2
import torch.nn.functional as F


class QHGNN_v2(nn.Module):
    def __init__(self, in_ch, n_class, n_hid, dropout=0.1):
        super(QHGNN_v2, self).__init__()
        self.dropout = dropout
        # Dropout means we randomly zero out some node features during training, 
        # This means the model can't over-rely on any single feature.
        self.hgc1 = QHGNN_conv_v2(in_ch, n_hid)
        # in_ch is the number of input features per node, n_hid is the number of hidden nodes
        self.hgc2 = QHGNN_conv_v2(n_hid, n_class)

    def forward(self, x, LS, RS, Q):
        x = F.relu(self.hgc1(x, LS, Q, RS))
        x = F.dropout(x, self.dropout)
        x = self.hgc2(x, LS, Q, RS)
        return x

