
import time
import copy
import numpy as np
import torch
from torch import optim

from data.data import load_postgres_business_list_data, load_postgres_business_list_opening_hours, load_postgres_review_data
from data.n_preprocessing import build_hypergraph_incidence_matrix, create_business_feature_matrix, create_label_vector, create_quality_matrix
from utils.utils import generate_G_from_H
from utils.utils import generate_G_from_H
from .HGNN import HGNN
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn

class Train_MoonLabHGNN:
    def __init__(self):
        self.reviews = load_postgres_review_data()


    def train(self, num_epochs=100, 
            lr=0.001, 
            hidden_layer_size=128, 
            train_proportion=0.8, 
            dropout=0.5, 
            weight_decay=5e-4, 
            gamma=0.5, 
            milestones_input="50,100"
            ):
        H, business_ids, business_to_idx = build_hypergraph_incidence_matrix(self.reviews)
        hours = load_postgres_business_list_opening_hours(business_ids)
        businesses = load_postgres_business_list_data(business_ids)

        fm = create_business_feature_matrix(businesses, hours)
        print(f"Feature matrix shape: {fm.shape}")

        lv = create_label_vector(businesses)

        n = len(businesses)
        split = int(n * train_proportion)

        training_range = np.arange(0, split)
        testing_range = np.arange(split, n)

        print(f"Training range: {0} - {split}, Testing range: {split+1} - {2*split}")

        print(f"H shape: {H.shape}")

        G = generate_G_from_H(H)
        print(f"G shape: {G.shape}")


        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        fts = torch.Tensor(fm).to(device)
        lbls = torch.Tensor(lv).long().to(device)
        G = torch.Tensor(G).to(device)
        Q = create_quality_matrix(G).to(device)
        idx_train = torch.Tensor(training_range).long().to(device)
        idx_test = torch.Tensor(testing_range).long().to(device)

        n_class = int(lbls.max()) + 1

        model_ft = HGNN(
            in_ch=fts.shape[1],
            n_class=n_class,
            n_hid=hidden_layer_size, 
            dropout=dropout
        ).to(device)

        optimizer = optim.Adam(model_ft.parameters(), lr, weight_decay=weight_decay)
        milestones = [int(x) for x in milestones_input.split(',')]
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
        criterion = torch.nn.CrossEntropyLoss()

        model_ft = train_model_moonlab(model_ft, criterion, optimizer, scheduler, num_epochs, print_freq=10, idx_train=idx_train, idx_test=idx_test, fts=fts, lbls=lbls, G=G)

def train_model_moonlab(model, criterion, optimizer, scheduler, num_epochs=25, print_freq=500, idx_train=None, idx_test=None, fts=None, lbls=None, G=None):
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