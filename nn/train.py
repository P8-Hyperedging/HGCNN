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
from model.HGNN import HGNN

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

def train_model(model, criterion, optimizer, scheduler, num_epochs=25, print_freq=500):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        if epoch % print_freq == 0:
            print('-' * 10)
            print(f'Epoch {epoch}/{num_epochs - 1}')

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            idx = idx_train if phase == 'train' else idx_test

            # Iterate over data.
            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(fts, G)
                loss = criterion(outputs[idx], lbls[idx])
                _, preds = torch.max(outputs, 1)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # statistics
            running_loss += loss.item() * fts.size(0)
            running_corrects += torch.sum(preds[idx] == lbls.data[idx])

            epoch_loss = running_loss / len(idx)
            epoch_acc = running_corrects.double() / len(idx)

            if epoch % print_freq == 0:
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        if epoch % print_freq == 0:
            print(f'Best val Acc: {best_acc:4f}')
            print('-' * 20)

    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

model_ft = train_model(model_ft, criterion, optimizer, scheduler, num_epochs=1000, print_freq=10)
#G = build_bipartite_graph(reviews)
#visualize_bipartite(G)