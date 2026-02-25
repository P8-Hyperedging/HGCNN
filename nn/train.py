import os
from torch_geometric.data import Data
from collections import defaultdict
from nn.data import load_review_data
from nn.graph import build_hypergraph_data

base_path = os.path.dirname(os.path.realpath(__file__))

reviews = load_review_data(base_path, limit=50000)

x, hyperedge_index, _ = build_hypergraph_data(reviews)

data = Data(x=x)
data.hyperedge_index = hyperedge_index

hyperedge_sizes = defaultdict(int)
for he in hyperedge_index[1]:
    hyperedge_sizes[he.item()] += 1

print("Nodes (# businesses, # features per):", data.x.shape)
print("Hyperedges (# entities connected, # incidence connections):", data.hyperedge_index.shape)
print("Number of hyperedges (unique # of users):",
      hyperedge_index[1].max().item() + 1)
print("Hyperedge size stats (# reviews per user):")
print("Min:", min(hyperedge_sizes.values()))
print("Max:", max(hyperedge_sizes.values()))
print("Average:",
      sum(hyperedge_sizes.values()) / len(hyperedge_sizes))