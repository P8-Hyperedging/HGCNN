
import json
import random
import time
import copy
import numpy as np
import torch
import sys
from torch import optim

import modelResult
from data.data import load_postgres_business_list_data, load_postgres_business_list_opening_hours, load_postgres_review_data
from data.n_preprocessing import build_hypergraph_incidence_matrix, create_business_feature_matrix, create_label_vector
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
            hidden_layer_size=256, 
            train_proportion=0.8, 
            dropout=0.5, 
            weight_decay=5e-4, 
            gamma=0.5, 
            milestones_input="50,100",
            model_name="MoonLabHGNN",
            job_id=None,
            seed = None,
            socket_logger=None
            ) -> modelResult.ModelResult:
        total_runtime_start = time.time()

        if seed == None:
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

        model_ft, valid_acc, train_runtime = train_model_moonlab(model_ft, criterion, optimizer, scheduler, num_epochs, print_freq=10, idx_train=idx_train, idx_test=idx_test, fts=fts, lbls=lbls, G=G, job_id=job_id, socket_logger=socket_logger)

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

        # Convert tensor to float for JSON serialization
        valid_acc_float = float(valid_acc.item()) if torch.is_tensor(valid_acc) else float(valid_acc)
        return modelResult.ModelResult(model_name, train_runtime, 0, valid_acc_float, 0, total_runtime, parameters_json, seed, job_id, num_epochs)

def train_model_moonlab(model, criterion, optimizer, scheduler, num_epochs=25, print_freq=500, idx_train=None, idx_test=None, fts=None, lbls=None, G=None, job_id=None, socket_logger=None):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        if epoch % print_freq == 0:
            seperator = '-' * 10
            msg = f'Epoch {epoch}/{num_epochs - 1}'
            if socket_logger:
                socket_logger(seperator, job_id=job_id, progress=epoch)
                socket_logger(msg, job_id=job_id, progress=epoch)
            else:
                print(seperator)
                print(msg)


        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            idx = idx_train if phase == 'train' else idx_test

            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(fts, G)
                loss = criterion(outputs[idx], lbls[idx])
                _, preds = torch.max(outputs, 1)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            epoch_loss = loss.item()
            epoch_acc = (preds[idx] == lbls[idx]).float().mean()

            if epoch % print_freq == 0:
                msg = f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}'

                # Use socket_logger if provided, else fallback to print
                if socket_logger:
                    socket_logger(msg, job_id=job_id, progress=epoch)
                else:
                    print(msg)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            if epoch % print_freq == 0:
                msg_best = f'Best val Acc: {best_acc:.4f}'
                separator = '-' * 20

                if socket_logger:
                    socket_logger(msg_best, job_id=job_id, progress=epoch)
                    socket_logger(separator, job_id=job_id, progress=epoch)
                else:
                    print(msg_best)
                    print(separator)

    time_elapsed = time.time() - since
    # At the end of training
    time_msg = f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s'
    best_msg = f'Best val Acc: {best_acc:.4f}'

    if socket_logger:
        # Send final messages
        socket_logger(time_msg, job_id=job_id, progress=num_epochs)
        socket_logger(best_msg, job_id=job_id, progress=num_epochs)
    
        # Optional: special final status so frontend knows training is finished
        socket_logger("TRAINING_COMPLETE", job_id=job_id, progress=num_epochs)
    else:
        print(time_msg)
        print(best_msg)

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_acc, time_elapsed
