
import time
import copy
import torch
import random
import json
import sys

import torch
from model.QualityHGNN.QHGNN import QHGNN
from torch import optim, split

from data.data import *
from data.n_preprocessing import *
from utils.utils import *


class Train_QHGNN:
    def __init__(self):
        self.reviews = load_postgres_review_data()

    def train(self, num_epochs=100, 
              lr=0.001, 
              hidden_layer_size=128, 
              train_proportion=0.8, 
              dropout=0.5, 
              weight_decay=5e-4, 
              gamma=0.5, 
              milestones_input="50,100",
              method_name="QHGNN"
              ):
        total_runtime_start = time.time()

        rng = np.random.default_rng()
        seed = int(rng.integers(low=0, high=np.iinfo(np.uint32).max, size=1)[0])

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

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

        model_ft = QHGNN(
            in_ch=fts.shape[1],
            n_class=n_class,
            n_hid=hidden_layer_size, 
            dropout=dropout
        ).to(device)

        optimizer = optim.Adam(model_ft.parameters(), lr, weight_decay=weight_decay)
        milestones = [int(x) for x in milestones_input.split(',')]
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
        criterion = torch.nn.CrossEntropyLoss()

        model_ft, valid_acc, train_runtime = train_model_QHGNN(model_ft, criterion, optimizer, scheduler, num_epochs, print_freq=10, idx_train=idx_train, idx_test=idx_test, fts=fts, lbls=lbls, G=G, Q=Q)

        total_runtime = time.time() - total_runtime_start

        parameters = {
            "Hidden Layer Size": hidden_layer_size,
            "Learning Rate": lr,
            "Weight Decay": weight_decay,
            "Epochs": num_epochs,
            "Train Proportion": train_proportion,
            "Dropout": dropout,
        }

        parameters_json = json.dumps(parameters)


        output_metrics_to_db(
            model_name=method_name,
            training_time=train_runtime,
            total_runtime=total_runtime,
            parameters=parameters_json,
            #Psycopg2 doesn't play nice when a tensor is parsed. Therefore check if valid_acc is tensor and convert to float if so.
            valid_acc=float(valid_acc.item()*100) if torch.is_tensor(valid_acc) else valid_acc*100,
            seed=seed
        )


def train_model_QHGNN(model, criterion, optimizer, scheduler, num_epochs=25, print_freq=500, idx_train=None, idx_test=None, fts=None, lbls=None, G=None, Q=None):
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
                outputs = model(fts, G, Q)
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
    return model, best_acc, time_elapsed