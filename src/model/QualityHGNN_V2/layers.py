import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class QHGNN_conv_v2(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True, quality_weight=1.0, quality=False):
        super(QHGNN_conv_v2, self).__init__()

        self.quality_weight = quality_weight

        self.quality = quality
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

    def forward(self, x: torch.Tensor, LS: torch.Tensor, Q: torch.Tensor, RS: torch.Tensor):
        """Q is a 1D vector of diagonal values (not a full matrix)."""
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias

        if not self.quality:
            G = LS.matmul(RS)
            x = G.matmul(x)
            return x

        with torch.no_grad():
            membership = (LS > 0).float()                          # (N, E)
            nodes_per_edge = membership.sum(dim=0).clamp(min=1)    # (E,)

            # Centroids for all hyperedges at once: (E, F)
            centroids = membership.T.matmul(x) / nodes_per_edge.unsqueeze(1)

            # Compute total distance per hyperedge in chunks to limit memory
            E = LS.shape[1]
            chunk_size = 100
            total_dists = torch.zeros(E, device=x.device)

            for start in range(0, E, chunk_size):
                end = min(start + chunk_size, E)
                # (N, chunk, F) - broadcast node features against chunk centroids
                diffs = x.unsqueeze(1) - centroids[start:end].unsqueeze(0)
                dists = diffs.norm(dim=2)                          # (N, chunk)
                dists = dists * membership[:, start:end]           # zero out non-members
                total_dists[start:end] = dists.sum(dim=0)


            # Normalize total_dists by its mean to keep distance_scores in reasonable range
            mean_total_dists = total_dists.mean()
            total_dists_normalized = total_dists / (mean_total_dists + 1e-8)  # Add small epsilon to avoid division by zero
            distance_scores = 1.0 / (1.0 + total_dists_normalized)            # (E,)
            distance_scores = distance_scores * self.quality_weight           # scale by quality_weight
            Q_updated = torch.clamp(distance_scores * Q, min=0, max=10)

            G = (LS * Q_updated.unsqueeze(0)).matmul(RS)

        x = G.matmul(x)
        return x
