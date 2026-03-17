'''This whole file is adapted from https://github.com/jianhao2016/AllSet'''


import os
import time
import torch
import argparse

import numpy as np
import os.path as osp
import scipy.sparse as sp
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from tqdm import tqdm

from layers import *
from models import *

from data.AllSetTransformer_preprocessing import *



def parse_method(args, data):
    if args.LearnMask:
        return SetGNN(args, data.norm)
    return SetGNN(args)


class Logger(object):
    """ Adapted from https://github.com/snap-stanford/ogb/ (This is adapted by Allsets)"""

    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 3
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            print(f'Run {run + 1:02d}:')
            print(f'Highest Train: {result[:, 0].max():.2f}')
            print(f'Highest Valid: {result[:, 1].max():.2f}')
            print(f'  Final Train: {result[argmax, 0]:.2f}')
            print(f'   Final Test: {result[argmax, 2]:.2f}')
        else:
            result = 100 * torch.tensor(self.results)

            best_results = []
            for r in result:
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                train2 = r[r[:, 1].argmax(), 0].item()
                test = r[r[:, 1].argmax(), 2].item()
                best_results.append((train1, valid, train2, test))

            best_result = torch.tensor(best_results)

            print(f'All runs:')
            r = best_result[:, 0]
            print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 1]
            print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 2]
            print(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 3]
            print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')

            return best_result[:, 1], best_result[:, 3]

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


if __name__ == '__main__':
    existing_dataset = ['yelp']
    # Parsing all arguments manually here, since Allset use argparse in a 
    # dynamic way to handle different methods, we only need one though
    args = argparse.Namespace(  
        method='AllSetTransformer',
        dname='yelp',
        All_num_layers=1,
        MLP_num_layers=2,
        MLP_hidden=64,
        Classifier_num_layers=1,
        Classifier_hidden=64,
        heads=1,
        lr=0.001,
        wd=0.0,
        epochs=1000, # Allsets use 500 epochs, we changed this to 1000 since we use 1000 for HNHN
                     # and we want to give all methods the same number of epochs for a fair comparison.
        runs=1,  # Allsets use 20 runs, but we just change this to 1 for testing for now.
        feature_noise=0.0,
        add_self_loop=False,
        exclude_self=False,
        normtype='all_one',
        train_prop=0.5,
        valid_prop=0.25,
        dropout=0.0,
        aggregate='mean',
        normalization='ln',
        deepset_input_norm=False,
        GPR=False,
        LearnMask=False,
        PMA=True,
        display_step=25
    )
    
    if args.dname in existing_dataset:
        f_noise = args.feature_noise
        # Load from database directly instead of using the wrapper class that Allset used.
        data = load_yelp_dataset(train_percent = args.train_prop)
        args.num_features = data.num_node_features
        args.num_classes = len(data.y.unique())
        if args.dname in ['yelp']:
            #         Shift the y label to start with 0
            args.num_classes = len(data.y.unique())
            data.y = data.y - data.y.min()
        if not hasattr(data, 'n_x'):
            data.n_x = torch.tensor([data.x.shape[0]])
        if not hasattr(data, 'num_hyperedges'):
            # note that we assume the he_id is consecutive.
            data.num_hyperedges = torch.tensor(
                [data.edge_index[0].max()-data.n_x[0]+1])

    data = ExtractV2E(data)
    if args.add_self_loop:
        data = Add_Self_Loops(data)
    if args.exclude_self:
        data = expand_edge_index(data)

    # Compute deg normalization: option in ['all_one','deg_half_sym'] (use args.normtype)
    # data.norm = torch.ones_like(data.edge_index[0])
    data = norm_contruction(data, option=args.normtype)
    
    
    
    logger = Logger(args.runs, args)
    
    criterion = nn.NLLLoss()
    eval_func = eval_acc
    
    model = parse_method(args, data)
    device = torch.device('cpu')

    model, data = model.to(device), data.to(device)
    num_params = count_parameters(model)

    model.train()
    # print('MODEL:', model)
    
    split_idx_lst = []
    for run in range(args.runs):
        split_idx = rand_train_test_idx(
            data.y, train_prop=args.train_prop, valid_prop=args.valid_prop)
        split_idx_lst.append(split_idx)
    


    ### Training loop ###
    runtime_list = []
    total_start_time = time.time()
    for run in tqdm(range(args.runs)):
        start_time = time.time()
        split_idx = split_idx_lst[run]
        train_idx = split_idx['train'].to(device)
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

        best_val = float('-inf')
        for epoch in range(args.epochs):
            #         Training part
            model.train()
            optimizer.zero_grad()
            out = model(data)
            out = F.log_softmax(out, dim=1)
            loss = criterion(out[train_idx], data.y[train_idx])
            loss.backward()
            optimizer.step()
    #         Evaluation part
            result = evaluate(model, data, split_idx, eval_func)
            logger.add_result(run, result[:3])
    
            if epoch % args.display_step == 0 and args.display_step > 0:
                print(f'Epoch: {epoch:02d}, '
                      f'Train Loss: {loss:.4f}, '
                      f'Valid Loss: {result[4]:.4f}, '
                      f'Test  Loss: {result[5]:.4f}, '
                      f'Train Acc: {100 * result[0]:.2f}%, '
                      f'Valid Acc: {100 * result[1]:.2f}%, '
                      f'Test  Acc: {100 * result[2]:.2f}%')

        end_time = time.time()
        runtime_list.append(end_time - start_time)

        total_end_time = time.time()
        total_runtime = total_end_time - total_start_time

    
        # logger.print_statistics(run)
    

    avg_time, std_time = np.mean(runtime_list), np.std(runtime_list)

    best_val, best_test = logger.print_statistics()
    res_root = 'hyperparameter_tunning'
    if not osp.isdir(res_root):
        os.makedirs(res_root)

    filename = f'{res_root}/{args.dname}_noise_{args.feature_noise}.csv'
    write_header = not osp.exists(filename) or osp.getsize(filename) == 0
    print(f"Saving results to {filename}")

    with open(filename, 'a+') as write_obj:
        if write_header:
            header = 'Method,Best_Val,Best_Test,Num_Params,Total_Runtime(s)\n'
            write_obj.write(header)
        cur_line = f'{args.method}_{args.lr}_{args.wd}_{args.heads}'
        # cur_line += f',{best_val.mean():.3f} +- {best_val.std():.3f}'   # Don't currently need avg
        # cur_line += f',{best_test.mean():.3f} +- {best_test.std():.3f}' # kept for future if needed
        cur_line += f',{num_params}' #, {avg_time:.2f}s, {std_time:.2f}s' 
        # cur_line += f',{avg_time//60}min{(avg_time % 60):.2f}s'
        cur_line += f',{total_runtime:.2f}s'
        cur_line += f'\n'
        write_obj.write(cur_line)

    all_args_file = f'{res_root}/all_args_{args.dname}_noise_{args.feature_noise}.csv'
    with open(all_args_file, 'a+') as f:
        f.write(str(args))
        f.write('\n')

    print('All done!')
    quit()
