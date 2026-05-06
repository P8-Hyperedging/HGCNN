
import time
import copy
import numpy as np
import torch
from torchmetrics.classification import MulticlassConfusionMatrix, MulticlassF1Score, MulticlassPrecision, MulticlassRecall
import random
import json
import sys

import torch
import modelResult
from model.QualityHGNN_V2.QHGNN import QHGNN_v2
from torch import device, optim, split

from data.data import *
from data.n_preprocessing import *
from utils.utils import *


class Train_QHGNN_v2:
    def __init__(self):
        self.reviews = load_postgres_review_data()

        # H, business_ids, _ = build_hypergraph_incidence_matrix(self.reviews)  # Commented out - using KNN method instead
        # Extract business_ids from reviews for data loading
        _, business_ids, _ = build_hypergraph_incidence_matrix(self.reviews)

        hours = load_postgres_business_list_opening_hours(business_ids)
        businesses = load_postgres_business_list_data(business_ids)

        fm = create_business_feature_matrix(businesses, hours)
        print(f"Feature matrix shape: {fm.shape}")

        # Build hypergraph using KNN method for ablation study
        H = construct_H_with_KNN(fm, K_neigs=[10])
        print(f"H shape (from KNN): {H.shape}")

        lv = create_label_vector(businesses)

        unique, counts = np.unique(lv, return_counts=True)
        print("Label distribution:")
        for label, count in zip(unique, counts):
            print(f"  Class {label}: {count} ({count/len(lv)*100:.1f}%)")

        # Q = create_quality_matrix_from_H(self.reviews)  # Commented out - using identity matrix instead
        Q = np.eye(H.shape[0])  # Replace with diagonal matrix of ones (identity matrix)
        print(f"Q shape: {Q.shape}")

        (DV2_H, _, invDE_HT_DV2) = generate_G_from_H(H, True)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("Using CPU")

        self.n = len(businesses)

        self.fts  = torch.Tensor(fm).to(self.device)
        self.lbls = torch.Tensor(lv).long().to(self.device)
        print("Unsparsing :(")
        self.LS   = torch.Tensor(DV2_H.toarray()).to(self.device)
        self.RS   = torch.Tensor(invDE_HT_DV2.toarray()).to(self.device)
        self.Q    = torch.Tensor(Q).to(self.device)
        self.n_class = int(self.lbls.max()) + 1

        print(f"LS shape: {self.LS.shape}")
        print(f"Q shape: {self.Q.shape}")

    def train(self, num_epochs=200,
              lr=0.001,
              hidden_layer_size=256,
              train_proportion=0.5,
              dropout=0.5,
              weight_decay=5e-4,
              gamma=0.5,
              milestones_input="50,100",
              model_name="QHGNN_v2",
              quality_weight=1.0,
              job_id=None,
              seed = None,
              socket_logger=None,
              patience=100
              )-> modelResult.ModelResult:
        total_runtime_start = time.time()

        if seed == None:
            rng = np.random.default_rng()
            seed = int(rng.integers(low=0, high=np.iinfo(np.uint32).max, size=1)[0])

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        train_split, valid_split, test_split = rand_train_test_idx_simple(self.n, train_prop=train_proportion)
        print(f"Total nodes: {self.n}, Training nodes: {len(train_split)}, Validation nodes: {len(valid_split)}")

        idx_train = train_split.long().to(self.device)
        idx_test  = valid_split.long().to(self.device)
        idx_valid = test_split.long().to(self.device)

        model_ft = QHGNN_v2(
            in_ch=self.fts.shape[1],
            n_class=self.n_class,
            n_hid=hidden_layer_size,
            quality_weight=quality_weight,
            dropout=dropout
        ).to(self.device)

        optimizer = optim.Adam(model_ft.parameters(), lr, weight_decay=weight_decay)
        milestones = [int(x) for x in milestones_input.split(',')]
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
        criterion = torch.nn.CrossEntropyLoss()

        model_ft, valid_acc, valid_f1, train_runtime, total_epochs = self.train_model_QHGNN_v2(
            model_ft, 
            criterion, 
            optimizer, 
            scheduler, 
            num_epochs, 
            print_freq=10, 
            idx_train=idx_train, 
            idx_valid=idx_valid, 
            fts=self.fts, 
            lbls=self.lbls, 
            LS=self.LS, 
            RS=self.RS, 
            Q=self.Q, 
            job_id=job_id, 
            socket_logger=socket_logger, 
            patience=patience)

        model_ft.eval()
        with torch.no_grad():
            outputs = model_ft(self.fts, self.LS, self.RS, self.Q)
            if idx_test.numel() == 0:
                test_acc = torch.tensor(0.0, device=self.fts.device)
            else:
                test_preds = torch.argmax(outputs[idx_test], dim=1)
                test_acc = (test_preds == self.lbls[idx_test]).float().mean()

        total_runtime = time.time() - total_runtime_start

        parameters = {
            "Hidden Layer Size": hidden_layer_size,
            "Learning Rate": lr,
            "Weight Decay": weight_decay,
            "Quality Weight": quality_weight,
            "Epochs": num_epochs,
            "Train Proportion": train_proportion,
            "Dropout": dropout,
            "Total Epochs": total_epochs,
            "Patience": patience,
        }

        parameters_json = json.dumps(parameters)


        log_model_metrics(
            model_name=model_name,
            job_id=job_id,
            training_time=train_runtime,
            total_runtime=total_runtime,
            parameters=parameters_json,
            valid_acc=float(valid_acc.item() * 100) if torch.is_tensor(valid_acc) else valid_acc * 100,
            valid_f1=float(valid_f1.item()) if torch.is_tensor(valid_f1) else float(valid_f1),
            seed=seed,
            test_acc=float(test_acc.item() * 100) if torch.is_tensor(test_acc) else test_acc * 100,
        )

        # Convert tensors to Python floats for JSON serialization
        valid_acc_float = float(valid_acc.item() * 100) if torch.is_tensor(valid_acc) else float(valid_acc * 100)
        test_acc_float = float(test_acc.item() * 100) if torch.is_tensor(test_acc) else float(test_acc * 100)

        return modelResult.ModelResult(model_name, train_runtime, 0, valid_acc_float, test_acc_float, total_runtime, parameters_json, seed, job_id, total_epochs)


    def train_model_QHGNN_v2(self, model, criterion, optimizer, scheduler, num_epochs=25, print_freq=1, idx_train=None, idx_valid=None, fts=None, lbls=None, LS=None, RS=None, Q=None, job_id=None, socket_logger=None, patience=None):
        since = time.time()

        # Early stopping parameters
        epochs_no_improve = 0
        stop_training = False
        total_epochs = 0


        n_class = int(lbls.max()) + 1
        f1_metric = MulticlassF1Score(num_classes=n_class, average="macro").to(fts.device)
        precision_metric = MulticlassPrecision(num_classes=n_class, average="macro").to(fts.device)
        recall_metric = MulticlassRecall(num_classes=n_class, average="macro").to(fts.device)
        confusion_metric = MulticlassConfusionMatrix(num_classes=n_class).to(fts.device)
        majority_class = torch.bincount(lbls[idx_train]).argmax()
        baseline_acc = (lbls[idx_valid] == majority_class).float().mean()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        best_f1 = 0.0

        for epoch in range(num_epochs):
            total_epochs += 1

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
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                idx = idx_train if phase == 'train' else idx_valid

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(fts, LS, RS, Q)
                    loss = criterion(outputs[idx], lbls[idx])
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        scheduler.step()

                epoch_loss = loss.item()
                epoch_acc = (preds[idx] == lbls[idx]).float().mean()

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
                    best_f1 = f1 if f1 > best_f1 else best_f1
                    # Early stopping check based on validation accuracy improvement
                    if epoch_acc > best_acc + 1e-4:
                        best_acc = epoch_acc
                        best_model_wts = copy.deepcopy(model.state_dict())
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1

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

                # Early stopping check after validation phase
                if epochs_no_improve >= patience:
                    stop_training = True
                    if socket_logger:
                        socket_logger(f'Early stopping at epoch {epoch} with best val Acc: {best_acc:.4f}', job_id=job_id, progress=epoch)
                    else:
                        print(f'Early stopping at epoch {epoch} with best val Acc: {best_acc:.4f}')
                    break

            if stop_training:
                break

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
        return model, best_acc, best_f1, time_elapsed, total_epochs
