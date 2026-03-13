from torch import nn
from .layers import QHGNN_conv
import torch.nn.functional as F


class QHGNN(nn.Module):
    def __init__(self, in_ch, n_class, n_hid, dropout=0.5):
        super(QHGNN, self).__init__()
        self.dropout = dropout
        self.hgc1 = QHGNN_conv(in_ch, n_hid)
        self.hgc2 = QHGNN_conv(n_hid, n_class)

    def forward(self, x, G, Q):
        x = F.relu(self.hgc1(x, G, Q))
        x = F.dropout(x, self.dropout)
        x = self.hgc2(x, G, Q)
        return x

