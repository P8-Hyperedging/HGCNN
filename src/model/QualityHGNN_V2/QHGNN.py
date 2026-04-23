from torch import nn
from .layers import QHGNN_conv_v2
import torch.nn.functional as F


class QHGNN_v2(nn.Module):
    def __init__(self, in_ch, n_class, n_hid, quality_weight=1.0, dropout=0.1):
        super(QHGNN_v2, self).__init__()
        self.dropout = dropout
        self.hgc1 = QHGNN_conv_v2(in_ch, n_hid)
        self.bn1  = nn.BatchNorm1d(n_hid)
        self.hgc2 = QHGNN_conv_v2(n_hid, n_class, quality_weight=quality_weight, quality=True)

    def forward(self, x, LS, RS, Q):
        x = self.bn1(F.relu(self.hgc1(x, LS, Q, RS)))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.hgc2(x, LS, Q, RS)
        return x
