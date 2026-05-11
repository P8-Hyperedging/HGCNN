'''This whole file is adapted from https://github.com/jianhao2016/AllSet'''


import json
import os
import random
import time
import torch
import argparse
import sys

import numpy as np
import os.path as osp
import torch.nn as nn
import torch.nn.functional as F

import modelResult
from .layers import *
from .AllSetTransformer import *

from data.AllSetTransformer_preprocessing import *
from parameters import get_allset_parameters



def parse_method(self, data):
    if self.LearnMask:
        return SetGNN(self, data.norm)
    else:
        return SetGNN(self)


class Logger(object):
    """ Adapted from https://github.com/snap-stanford/ogb/ (This is adapted by Allsets)"""

    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 3
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

@torch.no_grad()
def evaluate(model, data, split_idx, eval_func, result=None):
    if result is not None:
        out = result
    else:
        model.eval()
        out = model(data)
        out = F.log_softmax(out, dim=1)

    train_acc = eval_func(
        data.y[split_idx['train']], out[split_idx['train']])
    valid_acc = eval_func(
        data.y[split_idx['valid']], out[split_idx['valid']])
    test_acc = eval_func(
        data.y[split_idx['test']], out[split_idx['test']])

#     Also keep track of losses
    train_loss = F.nll_loss(
        out[split_idx['train']], data.y[split_idx['train']])
    valid_loss = F.nll_loss(
        out[split_idx['valid']], data.y[split_idx['valid']])
    test_loss = F.nll_loss(
        out[split_idx['test']], data.y[split_idx['test']])
    return train_acc, valid_acc, test_acc, train_loss, valid_loss, test_loss, out


def eval_acc(y_true, y_pred):
    acc_list = []
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=False).detach().cpu().numpy()


    is_labeled = y_true == y_true
    correct = y_true[is_labeled] == y_pred[is_labeled]
    acc_list.append(float(np.sum(correct))/len(correct))

    return sum(acc_list)/len(acc_list)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Train_AllSetTransformer:
    total_start_time = time.time()
    def __init__(self):
        self.dname = 'yelp'
        self.All_num_layers = 1
        self.MLP_num_layers = 2
        self.Classifier_num_layers = 1
        self.heads = 1
        self.feature_noise = 0.0
        self.add_self_loop = False
        self.exclude_self = False
        self.normtype = 'all_one'
        self.aggregate = 'mean'
        self.normalization = 'ln'
        self.deepset_input_norm = False
        self.GPR = False
        self.LearnMask = False
        self.PMA = True
        self.display_step = 10
        self.runs = 1

    def train(self, 
              hidden_layer_size=64, 
              lr=0.001, 
              weight_decay=0.0, 
              num_epochs=1000,
              train_proportion=0.5,
              valid_proportion=0.25,
              dropout=0.5, 
              model_name = 'AllSetTransformer',
              job_id = None,
              seed = None,
              socket_logger=None
              ) -> modelResult.ModelResult:
        total_start_time = time.time()

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
        
        valid_proportion = (1 - train_proportion) / 2 

        # Load from database directly instead of using the wrapper class that Allset used.
        data = load_yelp_dataset(train_percent = train_proportion)
        self.num_features = data.num_node_features
        self.num_classes = 9
        data.y = data.y - data.y.min()
        if not hasattr(data, 'n_x'):
            data.n_x = torch.tensor([data.x.shape[0]])
        if not hasattr(data, 'num_hyperedges'):
            # note that we assume the he_id is consecutive.
            data.num_hyperedges = torch.tensor(
                [data.edge_index[0].max()-data.n_x[0]+1])

        data = ExtractV2E(data)
        if self.add_self_loop:
            data = Add_Self_Loops(data)
        if self.exclude_self:
            data = expand_edge_index(data)


        data = norm_contruction(data, option=self.normtype)
        
        
        logger = Logger(self.runs, self)
        
        criterion = nn.NLLLoss()
        eval_func = eval_acc

        # This is needed inside the model at setGNN,
        # so we assign it to self.
        self.dropout = dropout
        self.MLP_hidden = hidden_layer_size
        self.Classifier_hidden = hidden_layer_size
        
        model = parse_method(self, data)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model, data = model.to(device), data.to(device)
        num_params = count_parameters(model)

        model.train()
        
        print(train_proportion)
        print(valid_proportion)

        split_idx_lst = []
        for run in range(self.runs):
            split_idx = rand_train_test_idx(
                data.y, train_prop=train_proportion, valid_prop=valid_proportion)
            split_idx_lst.append(split_idx)
        
        split_idx = split_idx_lst[0]
        print(f"total nodes : {data.y.shape[0]}")
        print(f"Train set size: {split_idx['train'].shape[0]}")
        print(f"Valid set size: {split_idx['valid'].shape[0]}")
        print(f"Test set size: {split_idx['test'].shape[0]}")
        print(f"Num classes: {self.num_classes}")

        ### Training loop ###
        train_start_time = time.time()
        for run in range(self.runs):
            split_idx = split_idx_lst[run]
            train_idx = split_idx['train'].to(device)
            model.reset_parameters()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

            for epoch in range(num_epochs):
                # Training part
                model.train()
                optimizer.zero_grad()
                out = model(data)
                out = F.log_softmax(out, dim=1)
                loss = criterion(out[train_idx], data.y[train_idx])
                loss.backward()
                optimizer.step()
                # Evaluation part
                result = evaluate(model, data, split_idx, eval_func)
                logger.add_result(run, result[:3])
        
                if epoch % self.display_step == 0 and self.display_step > 0:
                    epoch = f"Epoch: {epoch:01d}"
                    loss = (
                        f"Train Loss: {loss:.4f}, "
                        f"Valid Loss: {result[4]:.4f}, "
                        f"Test  Loss: {result[5]:.4f}, "
                    )
                    accuracy = (
                        f"Train Acc: {100 * result[0]:.2f}%, "
                        f"Valid Acc: {100 * result[1]:.2f}%, "
                        f"Test  Acc: {100 * result[2]:.2f}%"
                    )

                    if socket_logger:
                        socket_logger(epoch, job_id=job_id, progress=epoch)
                        socket_logger(loss, job_id=job_id, progress=epoch)
                        socket_logger(accuracy, job_id=job_id, progress=epoch)
                    else:
                        print(epoch)
                        print(loss)
                        print(accuracy)

            train_acc = 100*result[0]
            valid_acc = 100*result[1]
            test_acc = 100*result[2]

            train_runtime = time.time()- train_start_time

        
        total_runtime = time.time() - total_start_time
        time_msg = f'Training complete in {total_runtime // 60:.0f}m {total_runtime % 60:.0f}s'
        if socket_logger:
            socket_logger(time_msg, job_id=job_id, progress=num_epochs)

        parameters = {
            "Hidden Layer Size": hidden_layer_size,
            "Learning Rate": lr,
            "Weight Decay": weight_decay,
            "Epochs": num_epochs,
            "Train Proportion": train_proportion,
            "Valid Proportion": valid_proportion,
            "Dropout": dropout,
        }

        parameters_json = json.dumps(parameters)

        # Convert tensors to float for JSON serialization
        train_acc_float = float(train_acc.item()) if torch.is_tensor(train_acc) else float(train_acc)
        valid_acc_float = float(valid_acc.item()) if torch.is_tensor(valid_acc) else float(valid_acc)
        test_acc_float = float(test_acc.item()) if torch.is_tensor(test_acc) else float(test_acc)
        
        print('All done with AllSet!')
        return modelResult.ModelResult(model_name, train_runtime, train_acc_float, valid_acc_float, test_acc_float, total_runtime, parameters_json, seed, job_id, num_epochs)
