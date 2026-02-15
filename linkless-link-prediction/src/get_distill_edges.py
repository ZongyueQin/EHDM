import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import argparse
import itertools
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.utils import negative_sampling
import numpy as np
import torch_geometric.transforms as T
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from logger import Logger, ProductionLogger
from utils import get_dataset, do_edge_split
from models import MLP, GCN, SAGE, LinkPredictor
from torch_sparse import SparseTensor
from sklearn.metrics import *
from os.path import exists
from torch_cluster import random_walk
from torch.nn.functional import cosine_similarity
import torch_geometric
from train_teacher_gnn import test_transductive, test_production
import scipy.sparse as ssp
from torch_geometric.data import Data
from copy import deepcopy

dir_path  = '/local2/qzy_scai/EHDM/HeaRT/'

def read_data(data_name, neg_mode):
    data_name = data_name

    node_set = set()
    train_pos, valid_pos, test_pos = [], [], []
    train_neg, valid_neg, test_neg = [], [], []

    for split in ['train', 'test', 'valid']:

        if neg_mode == 'equal':
            path = dir_path+'/dataset' + '/{}/{}_pos.txt'.format(data_name, split)

        elif neg_mode == 'all':
            path = dir_path+'/dataset' + '/{}/allneg/{}_pos.txt'.format(data_name, split)

        for line in open(path, 'r'):
            sub, obj = line.strip().split('\t')
            sub, obj = int(sub), int(obj)
            
            node_set.add(sub)
            node_set.add(obj)
            
            if sub == obj:
                continue

            if split == 'train': 
                train_pos.append((sub, obj))
                

            if split == 'valid': valid_pos.append((sub, obj))  
            if split == 'test': test_pos.append((sub, obj))
    
    num_nodes = len(node_set)
    print('the number of nodes in ' + data_name + ' is: ', num_nodes)

    for split in ['test', 'valid']:

        if neg_mode == 'equal':
            path = dir_path+'/dataset' + '/{}/{}_neg.txt'.format(data_name, split)

        elif neg_mode == 'all':
            path = dir_path+'/dataset' + '/{}/allneg/{}_neg.txt'.format(data_name, split)

        for line in open(path, 'r'):
            sub, obj = line.strip().split('\t')
            sub, obj = int(sub), int(obj)
            # if sub == obj:
            #     continue
            
            if split == 'valid': 
                valid_neg.append((sub, obj))
               
            if split == 'test': 
                test_neg.append((sub, obj))

    train_edge = torch.transpose(torch.tensor(train_pos), 1, 0)
    edge_index = torch.cat((train_edge,  train_edge[[1,0]]), dim=1)
    edge_weight = torch.ones(edge_index.size(1))


    A = ssp.csr_matrix((edge_weight.view(-1), (edge_index[0], edge_index[1])), shape=(num_nodes, num_nodes)) 

    adj = SparseTensor.from_edge_index(edge_index, edge_weight, [num_nodes, num_nodes])
          

    train_pos_tensor = torch.tensor(train_pos)

    valid_pos = torch.tensor(valid_pos)
    valid_neg =  torch.tensor(valid_neg)

    test_pos =  torch.tensor(test_pos)
    test_neg =  torch.tensor(test_neg)

    idx = torch.randperm(train_pos_tensor.size(0))
    idx = idx[:valid_pos.size(0)]
    train_val = train_pos_tensor[idx]


    feature_embeddings = torch.load(dir_path+'/dataset' + '/{}/{}'.format(data_name, 'gnn_feature'))
    feature_embeddings = feature_embeddings['entity_embedding']

    data = {}
    data['adj'] = adj
    data['train_pos'] = train_pos_tensor
    data['train_val'] = train_val

    data['valid_pos'] = valid_pos
    data['valid_neg'] = valid_neg
    data['test_pos'] = test_pos
    data['test_neg'] = test_neg

    data['x'] = feature_embeddings

    return data

def neighbor_samplers(row, col, sample, x, step, ps_method, ns_rate, hops, device='cuda'):
    batch = sample

    if ps_method == 'rw':
        pos_batch = random_walk(row, col, batch, walk_length=step*hops,
                                coalesced=False)
    elif ps_method == 'nb':
        pos_batch = None
        for i in range(step):
            if pos_batch is None:
                pos_batch = random_walk(row, col, batch, walk_length=hops, coalesced=False)
            else:
                pos_batch = torch.cat((pos_batch, random_walk(row, col, batch, walk_length=hops,coalesced=False)[:,1:]), 1)

    neg_batch = torch.randint(0, x.size(0), (batch.numel(), step*hops*ns_rate),
                                  dtype=torch.long)

    if device == 'cuda':
        return pos_batch.to("cuda"), neg_batch.to("cuda")
    else:
        return pos_batch, neg_batch



def get_samples(data, split_edge,
                         args, device):
    if args.transductive == "transductive":
        pos_train_edge = split_edge['train']['edge'].to(data.x.device)
        row, col = data.adj_t
    else:
        pos_train_edge = data.edge_index.t()
        row, col = data.edge_index

    edge_index = torch.stack([col, row], dim=0)



    node_loader = iter(DataLoader(range(data.x.size(0)), args.node_batch_size, shuffle=True))
    all_pos_samples = []
    all_neg_samples = []
    for link_perm in DataLoader(range(pos_train_edge.size(0)), args.link_batch_size, shuffle=True):

        node_perm = next(node_loader).to(data.x.device)

        edge = pos_train_edge[link_perm].t()

        if args.LLP_R or args.LLP_D:
            sample_step = args.rw_step
            # sampled the neary nodes and randomely sampled nodes
            pos_sample, neg_sample = neighbor_samplers(row, col, node_perm, data.x, 100*sample_step, args.ps_method, args.ns_rate, args.hops, 'cpu')

            ### calculate the distribution based matching loss 
            all_pos_samples.append(pos_sample.cpu())
            all_neg_samples.append(neg_sample.cpu())
    return all_pos_samples, all_neg_samples



def parse_args():
    parser = argparse.ArgumentParser(description='OGBL-DDI (GNN)')
    parser.add_argument('--distill_teacher', type=str, default='None')
    parser.add_argument('--distill_pred_path', type=str, default='None')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--encoder', type=str, default='sage')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--link_batch_size', type=int, default=64*1024)
    parser.add_argument('--node_batch_size', type=int, default=64*1024)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--epochs', type=int, default=20000)
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--dataset_dir', type=str, default='../data')
    parser.add_argument('--datasets', type=str, default='collab')
    parser.add_argument('--predictor', type=str, default='mlp', choices=['inner','mlp'])
    parser.add_argument('--patience', type=int, default=100, help='number of patience steps for early stopping')
    parser.add_argument('--metric', type=str, default='Hits@20', choices=['auc', 'hits@20', 'hits@50'], help='main evaluation metric')
    parser.add_argument('--use_valedges_as_input', action='store_true')
    parser.add_argument('--True_label', default=0.1, type=float, help="true_label loss")
    parser.add_argument('--KD_RM', default=0, type=float, help="Representation-based matching KD") 
    parser.add_argument('--KD_LM', default=0, type=float, help="logit-based matching KD") 
    parser.add_argument('--LLP_D', default=1, type=float, help="distribution-based matching kd")
    parser.add_argument('--LLP_R', default=1, type=float, help="rank-based matching kd") 
    parser.add_argument('--margin', default=0.1, type=float, help="margin for rank-based kd") 
    parser.add_argument('--rw_step', type=int, default=3, help="nearby nodes sampled times")
    parser.add_argument('--ns_rate', type=int, default=1, help="randomly sampled rate over # nearby nodes") 
    parser.add_argument('--hops', type=int, default=2, help="random_walk step for each sampling time")
    parser.add_argument('--ps_method', type=str, default='nb', help="positive sampling is rw or nb")
    parser.add_argument('--transductive', type=str, default='transductive', choices=['transductive', 'production'])
    parser.add_argument('--minibatch', action='store_true')

    args = parser.parse_args()
    return args


def main(args, dataset = None, llp_d = None, llp_r = None, true_label = None, dropout = None, lr = None, distill_pred_path = None):
    if dataset is not None:
        args.datasets = dataset
    if llp_d is not None:
        args.LLP_D = llp_d
    if llp_r is not None:
        args.LLP_R = llp_r
    if true_label is not None:
        args.True_label = true_label
    if dropout is not None:
        args.dropout = dropout
    if lr is not None:
        args.lr = lr
    if distill_pred_path is not None:
        args.distill_pred_path = distill_pred_path
    print(args)


    device = "cpu"
    device = torch.device(device)
    mini_batch_device = 'cpu'

    ### Prepare the datasets
    if args.transductive == "transductive":
        if args.datasets != "collab":
            if args.datasets in ['cora', 'citeseer', 'pubmed']:
                heart_data = read_data(args.datasets, 'equal')
                new_data = Data(x=heart_data['x'], adj_t = heart_data['train_pos'].T)
                data = new_data.to(device)
                split_edge = {}
                split_edge['train'] = {'edge': heart_data['train_pos']}
                split_edge['valid'] = {'edge': heart_data['valid_pos'], 'edge_neg': heart_data['valid_neg']}
                split_edge['test'] = {'edge': heart_data['test_pos'], 'edge_neg': heart_data['test_neg']}
            else:
            
                dataset = get_dataset(args.dataset_dir, args.datasets)
                data = dataset[0]

                if exists("../data/" + args.datasets + ".pkl"):
                    split_edge = torch.load("../data/" + args.datasets + ".pkl")
                else:
                    split_edge = do_edge_split(dataset)
                    torch.save(split_edge, "../data/" + args.datasets + ".pkl")
            
                edge_index = split_edge['train']['edge'].t()
                data.adj_t = edge_index
            args.metric = 'Hits@20'

        elif args.datasets == "collab":
            dataset = PygLinkPropPredDataset(name='ogbl-collab')
            data = dataset[0]
            edge_index = data.edge_index
            data.edge_weight = data.edge_weight.view(-1).to(torch.float)
            data = T.ToSparseTensor()(data)

            split_edge = dataset.get_edge_split()
            data.adj_t = edge_index
            args.metric = 'Hits@50'

   
    
    all_pos_samples = []
    all_neg_samples = []
    for i in range(1):
        pos_samples, neg_samples = get_samples(data, split_edge,
                                args, device)
        all_pos_samples += pos_samples
        all_neg_samples += neg_samples
    all_pos_samples = torch.cat(all_pos_samples, dim=0)
    all_neg_samples = torch.cat(all_neg_samples, dim=0)
    print(all_pos_samples.size(), all_neg_samples.size())
    torch.save((all_pos_samples, all_neg_samples), f'../data/{args.datasets}_edges.pth') 
    return
    

args = parse_args()
main(args)
