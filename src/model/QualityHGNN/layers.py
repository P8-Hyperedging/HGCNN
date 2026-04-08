import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from utils.qualityutils import calculate_centroid_torch, calculate_total_distance_to_centroid_torch


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
    def _edge_node_indices(LS: torch.Tensor):
        num_edges = LS.shape[1]

        if LS.is_sparse:
            ls = LS.coalesce()
            row_idx, col_idx = ls.indices()
            if row_idx.numel() == 0:
                return [row_idx for _ in range(num_edges)]

            sorted_cols, perm = torch.sort(col_idx)
            sorted_rows = row_idx[perm]
            counts = torch.bincount(sorted_cols, minlength=num_edges)

            edge_nodes = []
            cursor = 0
            for count in counts.tolist():
                edge_nodes.append(sorted_rows[cursor:cursor + count])
                cursor += count
            return edge_nodes

        edge_nodes = []
        for i in range(num_edges):
            node_indices = torch.nonzero(LS[:, i] > 0, as_tuple=True)[0]
            edge_nodes.append(node_indices)
        return edge_nodes
        
    def forward(self, x: torch.Tensor, LS: torch.Tensor, Q: torch.Tensor, RS: torch.Tensor, edge_nodes=None):
        x = x.matmul(self.weight) 
        if self.bias is not None:
            x = x + self.bias

        q_diag = Q if Q.dim() == 1 else torch.diagonal(Q)
        if edge_nodes is None:
            edge_nodes = self._edge_node_indices(LS)

        with torch.no_grad(): 
            q_updated = q_diag.clone()
            for i, node_indices in enumerate(edge_nodes):
                if node_indices.numel() == 0:
                    q_updated[i] = 0
                    continue

                feature_matrix = x.index_select(0, node_indices)
                centroid = calculate_centroid_torch(feature_matrix)
                total_distance = calculate_total_distance_to_centroid_torch(feature_matrix, centroid).clamp_min(1e-8)
                q_updated[i] = (1 / total_distance) * q_diag[i]

            q_updated.clamp_(min=0, max=1)

        # Equivalent to LS @ diag(q_updated) @ RS @ x, but avoids materializing G (N x N).
        rs_x = self._left_mul(RS, x)
        weighted_rs_x = rs_x * q_updated.unsqueeze(1)
        x = self._left_mul(LS, weighted_rs_x)
        return x

