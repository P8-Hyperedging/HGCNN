import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class QHGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(QHGNN_conv, self).__init__()
        self.weight = Parameter(torch.Tensor(in_ft, out_ft)) # Create new feature matrix for hidden layer

        # Attention parameters (GAT-style, split into source and target halves)
        self.a_src = Parameter(torch.Tensor(out_ft, 1))
        self.a_dst = Parameter(torch.Tensor(out_ft, 1))

        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.reset_parameters()

    def reset_parameters(self):

        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

        # We start with ZEROS!!, so the model is more consistent for development.

    def forward(self, x: torch.Tensor, G: torch.Tensor):
        # 1. Transform features
        h = x.matmul(self.weight)

        # 2. Sparse GAT-style attention over edges in G
        # Compute per-node scores for source and destination sides
        attn_src = h.matmul(self.a_src).squeeze(1)  # (N,)
        attn_dst = h.matmul(self.a_dst).squeeze(1)  # (N,)

        # Get non-zero edge indices from sparse G
        indices = G.coalesce().indices()
        row, col = indices[0], indices[1]

        # Attention score for each edge = LeakyReLU(a_src[i] + a_dst[j])
        edge_attn = self.leakyrelu(attn_src[row] + attn_dst[col])

        # Sparse softmax: compute attention only over existing edges
        N = h.size(0)
        attn_sparse = torch.sparse_coo_tensor(
            torch.stack([row, col]), edge_attn, size=(N, N)
        ).coalesce()

        attention = torch.sparse.softmax(attn_sparse, dim=1)

        # 3. Aggregate neighbor features weighted by attention
        h_prime = torch.sparse.mm(attention, h)

        if self.bias is not None:
            h_prime = h_prime + self.bias

        return h_prime

