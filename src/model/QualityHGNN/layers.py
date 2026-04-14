import math
import torch
import torch.nn as nn
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

        # We start with ZEROS!!, so the model is more consistent for development.

    def forward(self, x: torch.Tensor, G: torch.Tensor):
        x = x.matmul(self.weight)
        # Create new feature matrix by multiplying with learnable weights.
        if self.bias is not None:
            x = x + self.bias

        # Multiply the feature matrix x with modified laplacian (aggregation)
        x = G.matmul(x)
        return x
