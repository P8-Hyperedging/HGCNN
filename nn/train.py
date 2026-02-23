import os

import torch
from torch_geometric.data import Data

from nn.data import load_business_data, load_review_data, sort_businesses_by_review_count, load_user_data
from nn.graph import build_bipartite_graph, visualize_bipartite

edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

data = Data(x=x, edge_index=edge_index)

base_path = os.path.dirname(os.path.realpath(__file__))

businesses = load_business_data(base_path)
users = load_user_data(base_path)
reviews = load_review_data(businesses, users, base_path, limit=50)

print("Sample Reviews:")
for r in reviews[:5]:
    print(r)

G = build_bipartite_graph(reviews)
visualize_bipartite(G)