from torch import nn
from .layers import QHGNN_conv
import torch.nn.functional as F


class QHGNN(nn.Module):
    def __init__(self, in_ch, n_class, n_hid, dropout=0.1):
        super(QHGNN, self).__init__()
        self.dropout = dropout
        self._cached_ls_id = None
        self._cached_membership = None
        # Dropout means we randomly zero out some node features during training, 
        # This means the model can't over-rely on any single feature.
        self.hgc1 = QHGNN_conv(in_ch, n_hid)
        # in_ch is the number of input features per node, n_hid is the number of hidden nodes
        self.hgc2 = QHGNN_conv(n_hid, n_class)

    def _get_cached_membership(self, LS):
        ls_id = id(LS)
        if self._cached_ls_id != ls_id:
            self._cached_membership = QHGNN_conv._membership_from_ls(LS)
            self._cached_ls_id = ls_id
        return self._cached_membership

    def forward(self, x, LS, RS, Q):
        membership = self._get_cached_membership(LS)
        x = F.relu(self.hgc1(x, LS, Q, RS, membership=membership))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.hgc2(x, LS, Q, RS, membership=membership)
        return x

