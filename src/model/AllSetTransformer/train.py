'''This whole file is adapted from https://github.com/jianhao2016/AllSet'''


import json
import os
import random
import time
import torch
import argparse

import numpy as np
import os.path as osp
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

from .layers import *
from .AllSetTransformer import *

from data.AllSetTransformer_preprocessing import *
from data.data import output_metrics_to_db
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
        self.display_step = 100
        self.runs = 1

    def train(self, 
              hidden_layer_size=64, 
              lr=0.001, 
              weight_decay=0.0, 
              epochs=1000,
              train_proportion=0.5,
              valid_proportion=0.25,
              dropout=0.0, 
              model_name = 'AllSetTransformer'
              ):
        
        total_start_time = time.time()

        seed = random.randint(0, 2**32 - 1)

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True

        # Load from database directly instead of using the wrapper class that Allset used.
        data = load_yelp_dataset(train_percent = train_proportion)
        self.num_features = data.num_node_features
        self.num_classes = len(data.y.unique())

        # Shift the y label to start with 0
        self.num_classes = len(data.y.unique())
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
        device = torch.device('cpu')

        model, data = model.to(device), data.to(device)
        num_params = count_parameters(model)

        model.train()

        split_idx_lst = []
        for run in range(self.runs):
            split_idx = rand_train_test_idx(
                data.y, train_prop=train_proportion, valid_prop=valid_proportion)
            split_idx_lst.append(split_idx)
        


        ### Training loop ###
        train_start_time = time.time()
        for run in tqdm(range(self.runs)):
            split_idx = split_idx_lst[run]
            train_idx = split_idx['train'].to(device)
            model.reset_parameters()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

            for epoch in range(epochs):
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
                    print(f'Epoch: {epoch:01d}, '
                        f'Train Loss: {loss:.4f}, '
                        f'Valid Loss: {result[4]:.4f}, '
                        f'Test  Loss: {result[5]:.4f}, '
                        f'Train Acc: {100 * result[0]:.2f}%, '
                        f'Valid Acc: {100 * result[1]:.2f}%, '
                        f'Test  Acc: {100 * result[2]:.2f}%')
            train_acc = 100*result[0]
            valid_acc = 100*result[1]
            test_acc = 100*result[2]

            train_runtime = time.time()- train_start_time

        
        total_runtime = time.time() - total_start_time

        parameters = {
            "Hidden Layer Size": hidden_layer_size,
            "Learning Rate": lr,
            "Weight Decay": weight_decay,
            "Epochs": epochs,
            "Train Proportion": train_proportion,
            "Valid Proportion": valid_proportion,
            "Dropout": dropout,
        }

        parameters_json = json.dumps(parameters)


        output_metrics_to_db(
            model_name,
            train_runtime,
            total_runtime,
            parameters_json,
            train_acc,
            valid_acc,
            test_acc,
            seed=seed
        )

        print('All done with AllSet!')
