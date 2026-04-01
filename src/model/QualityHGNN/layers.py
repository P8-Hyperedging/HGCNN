import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torchgen import local
from utils.qualityutils import calculate_centroid, calculate_centroid_torch, calculate_total_distance_to_centroid, calculate_total_distance_to_centroid_torch


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
        
    def forward(self, x: torch.Tensor, LS: torch.tensor, Q: torch.Tensor, RS: torch.Tensor):
        x = x.matmul(self.weight) 
        # Create new feature matrix by multiplying with learnable weights.
        if self.bias is not None:
            x = x + self.bias

        Q_updated = Q.clone()
        for i in range(LS.shape[1]): 
            node_mask = LS[:, i] > 0 
            node_indices = torch.nonzero(node_mask).squeeze() 
            
            # Handle case where only one node is in the hyperedge
            if node_indices.dim() == 0:  # Single node case
                feature_matrix = x[node_indices].unsqueeze(0)  # Make it 2D: (1, features)
            else:
                feature_matrix = x[node_indices]
            
            centroid = calculate_centroid_torch(feature_matrix)
            total_distance = calculate_total_distance_to_centroid_torch(feature_matrix, centroid) + 1e-8  # avoid division by zero
            
            Q_updated[i, i] = (1 / total_distance) * Q[i, i] 
        G = LS.matmul(Q_updated).matmul(RS) # G = DV2_H.dot(W_diag).dot(invDE_HT_DV2)

        x = G.matmul(x)
        return x # Return both Q and x

