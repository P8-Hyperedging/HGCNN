import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torchgen import local
from utils.qualityutils import calculate_centroid, calculate_centroid_torch, calculate_total_distance_to_centroid, calculate_total_distance_to_centroid_torch


class QHGNN_conv_v2(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(QHGNN_conv_v2, self).__init__()

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
        if self.bias is not None:
            x = x + self.bias

        with torch.no_grad(): 
            Q = Q.clone()  # Clone so original Q is not affected
            for i in range(LS.shape[1]): 
                node_mask = LS[:, i] > 0 
                node_indices = torch.nonzero(node_mask).squeeze() 
                
                if node_indices.dim() == 0:
                    feature_matrix = x[node_indices].unsqueeze(0)
                else:
                    feature_matrix = x[node_indices]
                
                centroid = calculate_centroid_torch(feature_matrix)
                total_distance = calculate_total_distance_to_centroid_torch(feature_matrix, centroid)
                #print(f"Total distance to centroid for hyperedge {i}: {total_distance.item()}")
                distance_score = 1 / (1 + total_distance)
                #print(f"Distance score for hyperedge {i}: {distance_score.item()}")
                #print(f"Original quality score for hyperedge {i}: {Q[i, i].item()}")
                updated_score = distance_score * Q[i, i]
                #print(f"Updated quality score for hyperedge {i}: {updated_score.item()}")
                Q[i, i] = updated_score

            Q = torch.clamp(Q, min=0, max=10)
            self.Q_updated = Q  # Store updated Q for tracking
            print (f"Q sample values after update: {Q.diag()[:10]}")
            G = LS.matmul(Q).matmul(RS)
        x = G.matmul(x)
        return x

