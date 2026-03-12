import torch
import numpy as np
import time
import copy
from torch_geometric.data import Data
from torch import optim

from data.data import *
from data.n_preprocessing import *
from utils.utils import *
from model.MoonLabHGNN.HGNN import HGNN
from model.MoonLabHGNN.train import train_model_moonlab

edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

data = Data(x=x, edge_index=edge_index)

reviews = load_postgres_review_data(limit=50000, min_reviews_per_user=3)
H, business_ids, business_to_idx = build_hypergraph_incidence_matrix(reviews)
hours = load_postgres_business_list_opening_hours(business_ids)
businesses = load_postgres_business_list_data(business_ids)

fm = create_business_feature_matrix(businesses, hours)
print(f"Feature matrix shape: {fm.shape}")
#for i in range(50):
    #print(f"Business {i} features: {fm[i]}")

lv = create_label_vector(businesses)

split = len(businesses) // 10

training_range = np.arange(0, split)
testing_range = np.arange(split+1, 2 * split)

print(f"Training range: {0} - {split}, Testing range: {split+1} - {2*split}")

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