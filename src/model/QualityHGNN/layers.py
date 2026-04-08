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

    @staticmethod
    def _left_mul(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        if left.is_sparse:
            return torch.sparse.mm(left, right)
        return left.matmul(right)

    @staticmethod
    def _membership_from_ls(LS: torch.Tensor):
        if LS.is_sparse:
            ls = LS.coalesce()
            row_idx, col_idx = ls.indices()
        else:
            row_idx, col_idx = torch.nonzero(LS > 0, as_tuple=True)

        counts = torch.bincount(col_idx, minlength=LS.shape[1])
        return row_idx, col_idx, counts
        
    def forward(self, x: torch.Tensor, LS: torch.Tensor, Q: torch.Tensor, RS: torch.Tensor, membership=None):
        x = x.matmul(self.weight) 
        if self.bias is not None:
            x = x + self.bias

        q_diag = Q if Q.dim() == 1 else torch.diagonal(Q)
        q_diag = q_diag.to(dtype=x.dtype, device=x.device)
        if membership is None:
            membership = self._membership_from_ls(LS)

        row_idx, col_idx, counts = membership
        num_edges = LS.shape[1]

        with torch.no_grad(): 
            if row_idx.numel() == 0:
                q_updated = torch.zeros_like(q_diag)
            else:
                node_features = x.index_select(0, row_idx)

                edge_feature_sum = torch.zeros(
                    (num_edges, x.shape[1]),
                    device=x.device,
                    dtype=x.dtype
                )
                edge_feature_sum.index_add_(0, col_idx, node_features)

                counts_clamped = counts.clamp_min(1).to(device=x.device, dtype=x.dtype)
                centroids = edge_feature_sum / counts_clamped.unsqueeze(1)

                repeated_centroids = centroids.index_select(0, col_idx)
                distances = torch.norm(node_features - repeated_centroids, dim=1)

                total_distance = torch.zeros(num_edges, device=x.device, dtype=x.dtype)
                total_distance.index_add_(0, col_idx, distances)

                q_updated = q_diag * total_distance.clamp_min(1e-8).reciprocal()
                q_updated = q_updated.masked_fill(counts == 0, 0)
                q_updated.clamp_(min=0, max=1)

        # Equivalent to LS @ diag(q_updated) @ RS @ x, but avoids materializing G (N x N).
        rs_x = self._left_mul(RS, x)
        weighted_rs_x = rs_x * q_updated.unsqueeze(1)
        x = self._left_mul(LS, weighted_rs_x)
        return x

