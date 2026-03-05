import os

import torch
import numpy as np
import time
import copy
from torch_geometric.data import Data
from torch import optim

from data.data import load_business_data, load_review_data, sort_businesses_by_review_count, load_user_data
from model.graph import build_bipartite_graph, visualize_bipartite
from data.knn_preprocessing import create_business_feature_matrix, create_label_vector
from utils.utils import construct_H_with_KNN, generate_G_from_H
from model.MoonLabHGNN.HGNN import HGNN
from model.MoonLabHGNN.train import train_model_moonlab
from data.data2 import load


'''
edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

data = Data(x=x, edge_index=edge_index)

base_path = os.path.dirname(os.path.realpath(__file__))

businesses = load_business_data(base_path, limit=10000)
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

'''

class Args:
    def __init__(self):
        self.data = 'cocitation'  # 'coauthorship' or 'cocitation'
        self.dataset = 'pubmed'        # 'cora', 'dblp' for coauthorship; 'cora', 'citeseer', 'pubmed' for cocitation
        self.split = 2             

args = Args()

# Load the dataset
print(f"Loading {args.data}/{args.dataset} dataset...")
dataset, train_idx, test_idx = load(args)

# Extract data
hypergraph = dataset['hypergraph']  # dict: {edge_id: [node_ids]}
features = dataset['features']       # numpy array (n_nodes x n_features)
labels = dataset['labels']           # numpy array (n_nodes x n_classes) - one-hot encoded
n_nodes = dataset['n']

print(f"Number of nodes: {n_nodes}")
print(f"Number of hyperedges: {len(hypergraph)}")
print(f"Feature dimension: {features.shape[1]}")
print(f"Number of classes: {labels.shape[1]}")

fm = features  # This is your feature matrix
lv = labels.argmax(axis=1) 

H = construct_H_with_KNN(fm, K_neigs=20, split_diff_scale=False, is_probH=True, m_prob=1)

print(f"H shape: {H.shape}")

G = generate_G_from_H(H)

print(f"G shape: {G.shape}")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

fts = torch.Tensor(fm).to(device)
lbls = torch.Tensor(lv).long().to(device)
G = torch.Tensor(G).to(device)
idx_train = torch.Tensor(train_idx).long().to(device)
idx_test = torch.Tensor(test_idx).long().to(device)

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