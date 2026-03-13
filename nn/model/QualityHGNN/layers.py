import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class QHGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(QHGNN_conv, self).__init__()

        self.weight = Parameter(torch.Tensor(in_ft, out_ft)) # Create new feature matrix for hidden layer
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, G: torch.Tensor, Q: torch.Tensor):
        x = x.matmul(self.weight) 
        # Create new feature matrix by multiplying with learnable weights.
        if self.bias is not None:
            x = x + self.bias
        # Multiply G by quality matrix Q to change the contribution of each hyperedge
        Q_G = G * Q 
        
        # Multiply the feature matrix x with modified laplacian (aggregation)
        x = Q_G.matmul(x)
        return x

    
class QHGNN_fc(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(QHGNN_fc, self).__init__()
        self.fc = nn.Linear(in_ch, out_ch)

    def forward(self, x):
        return self.fc(x)


class QHGNN_embedding(nn.Module):
    def __init__(self, in_ch, n_hid, dropout=0.5):
        super(QHGNN_embedding, self).__init__()
        self.dropout = dropout
        self.hgc1 = QHGNN_conv(in_ch, n_hid)
        self.hgc2 = QHGNN_conv(n_hid, n_hid)

    def forward(self, x, G, Q):
        x = F.relu(self.hgc1(x, G, Q))
        x = F.dropout(x, self.dropout)
        x = F.relu(self.hgc2(x, G, Q))
        return x


class QHGNN_classifier(nn.Module):
    def __init__(self, n_hid, n_class):
        super(QHGNN_classifier, self).__init__()
        self.fc1 = nn.Linear(n_hid, n_class)

    def forward(self, x):
        x = self.fc1(x)
        return x