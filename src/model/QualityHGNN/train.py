
import time
import copy
import numpy as np
import torch
from torchmetrics.classification import MulticlassConfusionMatrix, MulticlassF1Score, MulticlassPrecision, MulticlassRecall
import random
import json
import sys

import torch
from model.QualityHGNN.QHGNN import QHGNN
#from sklearn.model_selection import train_test_split
from torch import device, optim, split

from data.data import *
from data.n_preprocessing import *
from utils.utils import *


class Train_QHGNN:
    def __init__(self):
        self.reviews = load_postgres_review_data()

    def train(self, num_epochs=200, 
              lr=0.009, 
              hidden_layer_size=256, 
              train_proportion=0.8, 
              dropout=0.5, 
              weight_decay=5e-4, 
              gamma=0.5, 
              milestones_input="50,100",
              model_name="QHGNN",
              job_id=None,
              seed = None,
              socket_logger=None
              ):
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
        print(f"H shape: {H.shape}")

        hours = load_postgres_business_list_opening_hours(business_ids)
        businesses = load_postgres_business_list_data(business_ids)

        fm = create_business_feature_matrix(businesses, hours)
        print(f"Feature matrix shape: {fm.shape}")

        lv = create_label_vector(businesses)

        # Print label distribution
        unique, counts = np.unique(lv, return_counts=True)
        print("Label distribution:")
        for label, count in zip(unique, counts):
            print(f"  Class {label}: {count} ({count/len(lv)*100:.1f}%)")

        n = len(businesses)
        split = int(n * train_proportion)

        training_range = np.arange(0, split)
        testing_range = np.arange(split, n)

        print(f"Total nodes: {n}, Training nodes: {len(training_range)}, Testing nodes: {len(testing_range)}")


        Q =  diags(create_quality_matrix_from_H(self.reviews))
        print(f"Q shape: {Q.shape}")

        (DV2_H, W_diag, invDE_HT_DV2) = generate_G_from_H(H, True)
        # Generating G terms

        G = DV2_H.dot(W_diag).dot(invDE_HT_DV2).tocoo()
        print(f"G shape: {G.shape}")


        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        if (torch.cuda.is_available()):
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("Using CPU")


        fts = torch.Tensor(fm).to(device)
        lbls = torch.Tensor(lv).long().to(device)
        G = torch.sparse_coo_tensor(
            torch.from_numpy(np.array([G.row, G.col], dtype=np.int64)),
            torch.from_numpy(G.data.astype(np.float32)),
            size=G.shape
        ).coalesce().to(device)
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

        model_ft, valid_acc, train_runtime = train_model_QHGNN(model_ft, criterion, optimizer, scheduler, num_epochs, print_freq=10, idx_train=idx_train, idx_test=idx_test, fts=fts, lbls=lbls, G=G, job_id=job_id, socket_logger=socket_logger)

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
            model_name=model_name,
            job_id=job_id,
            training_time=train_runtime,
            total_runtime=total_runtime,
            parameters=parameters_json,
            #Psycopg2 doesn't play nice when a tensor is parsed. Therefore check if valid_acc is tensor and convert to float if so.
            valid_acc=float(valid_acc.item()*100) if torch.is_tensor(valid_acc) else valid_acc*100,
            seed=seed
        )


def train_model_QHGNN(model, criterion, optimizer, scheduler, num_epochs=25, print_freq=1, idx_train=None, idx_test=None, fts=None, lbls=None, G=None, job_id=None, socket_logger=None):
    since = time.time()

    n_class = int(lbls.max()) + 1
    f1_metric = MulticlassF1Score(num_classes=n_class, average="macro").to(fts.device)
    precision_metric = MulticlassPrecision(num_classes=n_class, average="macro").to(fts.device)
    recall_metric = MulticlassRecall(num_classes=n_class, average="macro").to(fts.device)
    confusion_metric = MulticlassConfusionMatrix(num_classes=n_class).to(fts.device)
    majority_class = torch.bincount(lbls[idx_train]).argmax()
    baseline_acc = (lbls[idx_test] == majority_class).float().mean()

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

            if phase == 'val':
                f1_metric.reset()
                precision_metric.reset()
                recall_metric.reset()
                confusion_metric.reset()
                f1 = f1_metric(preds[idx], lbls[idx])
                precision = precision_metric(preds[idx], lbls[idx])
                recall = recall_metric(preds[idx], lbls[idx])
                confusion = confusion_metric(preds[idx], lbls[idx]).detach().cpu().tolist()
                confusion_rows = "\n".join(str(row) for row in confusion) # Newline between rows to make it readable

            if epoch % print_freq == 0:
                if phase == 'val':
                    delta_acc = epoch_acc - baseline_acc
                    msg = (
                        f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} '
                        f'MajorityOnlyAcc: {baseline_acc:.4f} DeltaAcc: {delta_acc:.4f} '
                        f'MacroP: {precision:.4f} MacroR: {recall:.4f} MacroF1: {f1:.4f}'
                    )
                else:
                    msg = f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}'
                if socket_logger:
                    socket_logger(msg, job_id=job_id, progress=epoch)
                else:
                    print(msg)
                    if phase == 'val':
                        print('Confusion Matrix:')
                        print(confusion_rows)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())


        if epoch % print_freq == 0:
            msg = f'Best val Acc: {best_acc:4f}'
            if socket_logger:
                socket_logger(msg, job_id, progress=epoch)
            else:
                print(msg)

    time_elapsed = time.time() - since
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
