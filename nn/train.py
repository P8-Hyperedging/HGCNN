import os
from torch_geometric.data import Data
from collections import defaultdict
from data.data import load_review_data
from model.graph import build_hypergraph_data

base_path = os.path.dirname(os.path.realpath(__file__))

reviews = load_review_data(base_path, limit=500000)

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
'''
import torch
import numpy as np
import time
import copy
from torch_geometric.data import Data
from torch import optim

from data.data import load_business_data, load_postgres_business_data, load_postgres_review_data, load_review_data, sort_businesses_by_review_count, load_user_data
from model.graph import build_bipartite_graph, visualize_bipartite
from data.knn_preprocessing import create_business_feature_matrix, create_label_vector
from utils.utils import construct_H_with_KNN, generate_G_from_H
from model.MoonLabHGNN.HGNN import HGNN
from model.MoonLabHGNN.train import train_model_moonlab

edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

data = Data(x=x, edge_index=edge_index)


businesses = load_postgres_business_data(limit=10000)
#users = load_user_data(base_path)
#reviews = load_review_data(businesses, users, base_path, limit=50)

print("Sample Reviews:")
for r in businesses[:10]:
    print(r)


fm = create_business_feature_matrix(businesses)

for i in fm[:10]:
    print(i)

lv = create_label_vector(businesses)

for i in lv[:10]:
    print(i)

split = len(businesses) // 10

training_range = np.arange(0, split)
testing_range = np.arange(split+1, 2 * split)

print( f"Training range: {0} - {split}, Testing range: {split+1} - {2*split}" )

H = construct_H_with_KNN(fm, K_neigs=20, split_diff_scale=False, is_probH=True, m_prob=1)

print(f"H shape: {H.shape}")

G = generate_G_from_H(H)

print(f"G shape: {G.shape}")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

fts = torch.Tensor(fm).to(device)
lbls = torch.Tensor(lv).long().to(device)
G = torch.Tensor(G).to(device)
idx_train = torch.Tensor(training_range).long().to(device)
idx_test = torch.Tensor(testing_range).long().to(device)

n_class = int(lbls.max()) + 1

model_ft = HGNN(
    in_ch=fts.shape[1],
    n_class=n_class,
    n_hid=128,     
    dropout=0.5
).to(device)

optimizer = optim.Adam(model_ft.parameters(), lr=0.001, weight_decay=5e-4)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100], gamma=0.5)
criterion = torch.nn.CrossEntropyLoss()

model_ft = train_model_moonlab(model_ft, criterion, optimizer, scheduler, num_epochs=1000, print_freq=10, idx_train=idx_train, idx_test=idx_test, fts=fts, lbls=lbls, G=G)
#G = build_bipartite_graph(reviews)
#visualize_bipartite(G)
'''