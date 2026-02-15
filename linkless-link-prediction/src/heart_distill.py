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



def cosine_loss(s, t):
    return 1-cosine_similarity(s, t.detach(), dim=-1).mean()

def kl_loss(s,t,T):
    y_s = F.log_softmax(s/T, dim=-1)
    y_t = F.softmax(t/T, dim=-1)
    loss = F.kl_div(y_s,y_t,size_average=False) * (T**2) / y_s.size()[0]
    return loss

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

def train_minibatch(model, predictor, t_pos_pred, t_neg_pred, pos_nb, neg_nb, data, split_edge, optimizer, args, device):
    
    if args.transductive == "transductive":
        pos_train_edge = split_edge['train']['edge']
        row, col = data.adj_t
    else:
        pos_train_edge = data.edge_index.t()
        row, col = data.edge_index

    edge_index = torch.stack([col, row], dim=0)

    model.train()
    predictor.train()

    mse_loss = nn.MSELoss()
    bce_loss = nn.BCELoss()
    margin_rank_loss = nn.MarginRankingLoss(margin=args.margin)

    total_loss = total_examples = 0

    node_loader = iter(DataLoader(range(data.x.size(0)), args.node_batch_size, shuffle=True))
    pos_cnt = t_pos_pred.size(1)
    neg_cnt = t_neg_pred.size(1)
    for link_perm in DataLoader(range(pos_train_edge.size(0)), args.link_batch_size, shuffle=True):
        optimizer.zero_grad()

        node_perm = next(node_loader)

        edge = pos_train_edge[link_perm].t()

        if args.datasets != "collab":
            neg_edge = negative_sampling(edge_index, num_nodes=data.x.size(0),
                                 num_neg_samples=link_perm.size(0), method='dense')
        elif args.datasets == "collab":
            neg_edge = torch.randint(0, data.x.size()[0], [edge.size(0), edge.size(1)], dtype=torch.long)

        train_edges = torch.cat((edge, neg_edge), dim=-1).to(device)

        src = train_edges[0]
        dst = train_edges[1]

        # sampled the neary nodes and randomely sampled nodes 
#        sample_step = args.rw_step
#        pos_sample, neg_sample = neighbor_samplers(row, col, node_perm, data.x, sample_step, args.ps_method, args.ns_rate, args.hops)
        pos_idx = torch.randint(pos_cnt, (sample_step*args.hops,))
        pos_samples = pos_nb[node_perm]
        pos_samples = pos_samples[:, pos_idx+1]
        neg_idx = torch.randint(neg_cnt, (sample_step*args.hops*args.ns_rate,))
        neg_samples = neg_nb[node_perm]
        neg_samples = neg_samples[:, neg_idx]
        samples = torch.cat((pos_sample, neg_sample), 1)
        this_target = torch.cat((torch.reshape(samples, (-1,)), src, dst), 0)
        h = model(data.x[this_target.to("cpu")].to(device))

        ### calculate the distribution based matching loss 
        for_loss = torch.reshape(h[:samples.size(0) * samples.size(1)], (samples.size(0), samples.size(1), args.hidden_channels))
        src_h = h[samples.size(0) * samples.size(1): samples.size(0) * samples.size(1)+src.size(0)]
        dst_h = h[samples.size(0) * samples.size(1)+src.size(0): ]

        batch_emb = torch.reshape(for_loss[:,0,:], (samples[:,0].size(0), 1, h.size(1))).repeat(1,sample_step*args.hops*(1+args.ns_rate),1)
#        t_emb = torch.reshape(t_h[samples[:,0]].to(device), (samples[:,0].size(0), 1, t_h.size(1))).repeat(1,sample_step*args.hops*(1+args.ns_rate),1)
        s_r = predictor(batch_emb, for_loss[:,1:,:])
#        t_r = teacher_predictor(t_emb, t_h[samples[:, 1:]].to(device))
        pos_t_r = t_pos_pred[node_perm]
        pos_t_r = pos_t_r[:, pos_idx]
        neg_t_r = t_neg_pred[node_perm]
        neg_t_r = neg_t_r[:, neg_idx]
        t_r = torch.cat((pos_t_r, neg_t_r), dim=1)
        llp_d_loss = kl_loss(torch.reshape(s_r, (s_r.size()[0], s_r.size()[1])), torch.reshape(t_r, (t_r.size()[0], t_r.size()[1])), 1)

        #### calculate the rank based matching loss
        rank_loss = torch.tensor(0.0).to(device)
        sampled_nodes = [l_i for l_i in range(sample_step*args.hops*(1+args.ns_rate))]
        dim_pairs = [x for x in itertools.combinations(sampled_nodes, r=2)]
        dim_pairs = np.array(dim_pairs).T
        teacher_rank_list = torch.zeros((len(t_r), dim_pairs.shape[1],1)).to(device)
                    
        mask = t_r[:, dim_pairs[0]] > (t_r[:, dim_pairs[1]] + args.margin)
        teacher_rank_list[mask] = 1
        mask2 = t_r[:, dim_pairs[0]] < (t_r[:, dim_pairs[1]] - args.margin)
        teacher_rank_list[mask2] = -1
        first_rank_list = s_r[:, dim_pairs[0]].squeeze()
        second_rank_list = s_r[:, dim_pairs[1]].squeeze()
        llp_r_loss = margin_rank_loss(first_rank_list, second_rank_list, teacher_rank_list.squeeze())
        

        train_label = torch.cat((torch.ones(edge.size()[1]), torch.zeros(neg_edge.size()[1])), dim=0).to(h.device)
        out = predictor(src_h, dst_h).squeeze()
        label_loss = bce_loss(out, train_label)
       
        if args.LLP_D or args.LLP_R:
            loss = args.True_label * label_loss + args.LLP_D * llp_d_loss + args.LLP_R * llp_r_loss

        loss.backward()

        torch.nn.utils.clip_grad_norm_(data.x, 1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)

        optimizer.step()

        num_examples = edge.size(1)
        total_loss += loss.item() * num_examples
        total_examples += num_examples
        
    return total_loss / total_examples


def get_samples(model, predictor, data, split_edge,
                         args, device):
    if args.transductive == "transductive":
        pos_train_edge = split_edge['train']['edge'].to(data.x.device)
        row, col = data.adj_t
    else:
        pos_train_edge = data.edge_index.t()
        row, col = data.edge_index

    edge_index = torch.stack([col, row], dim=0)

    model.train()
    predictor.train()

    mse_loss = nn.MSELoss()
    bce_loss = nn.BCELoss()
    margin_rank_loss = nn.MarginRankingLoss(margin=args.margin)

    total_loss = total_examples = 0

    node_loader = iter(DataLoader(range(data.x.size(0)), args.node_batch_size, shuffle=True))
    all_pos_samples = []
    all_neg_samples = []
    for link_perm in DataLoader(range(pos_train_edge.size(0)), args.link_batch_size, shuffle=True):

        node_perm = next(node_loader).to(data.x.device)

        h = model(data.x)

        edge = pos_train_edge[link_perm].t()

        if args.LLP_R or args.LLP_D:
            sample_step = args.rw_step
            # sampled the neary nodes and randomely sampled nodes
            pos_sample, neg_sample = neighbor_samplers(row, col, node_perm, data.x, 100*sample_step, args.ps_method, args.ns_rate, args.hops, 'cpu')

            ### calculate the distribution based matching loss 
            all_pos_samples.append(pos_sample.cpu())
            all_neg_samples.append(neg_sample.cpu())
    return all_pos_samples, all_neg_samples


def train(model, predictor, t_pos_pred, t_neg_pred, pos_nb, neg_nb, data, split_edge,
                         optimizer, args, device):

    if args.transductive == "transductive":
        pos_train_edge = split_edge['train']['edge'].to(data.x.device)
        row, col = data.adj_t
    else:
        pos_train_edge = data.edge_index.t()
        row, col = data.edge_index

    edge_index = torch.stack([col, row], dim=0)

    model.train()
    predictor.train()

    mse_loss = nn.MSELoss()
    bce_loss = nn.BCELoss()
    margin_rank_loss = nn.MarginRankingLoss(margin=args.margin)

    total_loss = total_examples = 0

    node_loader = iter(DataLoader(range(data.x.size(0)), args.node_batch_size, shuffle=True))
    pos_cnt = t_pos_pred.size(1)
    neg_cnt = t_neg_pred.size(1)

    for link_perm in DataLoader(range(pos_train_edge.size(0)), args.link_batch_size, shuffle=True):
        optimizer.zero_grad()

        try:
            node_perm = next(node_loader).to(data.x.device)
        except:
            node_loader = iter(DataLoader(range(data.x.size(0)), args.node_batch_size, shuffle=True))
            node_perm = next(node_loader).to(data.x.device)


        h = model(data.x)

        edge = pos_train_edge[link_perm].t()

        if args.LLP_R or args.LLP_D:
            sample_step = args.rw_step
            # sampled the neary nodes and randomely sampled nodes
            # pos_sample, neg_sample = neighbor_samplers(row, col, node_perm, data.x, sample_step, args.ps_method, args.ns_rate, args.hops)
            pos_idx = torch.randint(pos_cnt, (sample_step*args.hops,))
#            pos_idx = torch.LongTensor(list(range(sample_step*args.hops)))
#            pos_sample = pos_nb[node_perm]
            anchor_nodes = pos_nb[:,0:1] 
            pos_sample = pos_nb[:, pos_idx+1]
            neg_idx = torch.randint(neg_cnt, (sample_step*args.hops*args.ns_rate,))
 #           neg_idx = torch.LongTensor(list(range(sample_step *args.hops)))

#            neg_sample = neg_nb[node_perm]
            neg_sample = neg_nb[:, neg_idx]

            ### calculate the distribution based matching loss 
            samples = torch.cat((anchor_nodes, pos_sample, neg_sample), 1)
#            print(samples.size())
            samples = samples[node_perm]
 #           print(samples.size())
            batch_emb = torch.reshape(h[samples[:,0]], (samples[:,0].size(0), 1, h.size(1))).repeat(1,sample_step*args.hops*(1+args.ns_rate),1)
            #t_emb = torch.reshape(t_h[samples[:,0]], (samples[:,0].size(0), 1, h.size(1))).repeat(1,sample_step*args.hops*(1+args.ns_rate),1)
            s_r = predictor(batch_emb, h[samples[:, 1:]])
            #t_r = teacher_predictor(t_emb, t_h[samples[:, 1:]])
  #          print(t_pos_pred.size())
            pos_t_r = t_pos_pred[node_perm]
   #         print(pos_t_r.size())

            pos_t_r = pos_t_r[:, pos_idx]
    #        print(pos_t_r.size())
            neg_t_r = t_neg_pred[node_perm]
            neg_t_r = neg_t_r[:, neg_idx]
            t_r = torch.cat((pos_t_r, neg_t_r), dim=1)

            llp_d_loss = kl_loss(torch.reshape(s_r, (s_r.size()[0], s_r.size()[1])), torch.reshape(t_r, (t_r.size()[0], t_r.size()[1])), 1)

            #### calculate the rank based matching loss
            rank_loss = torch.tensor(0.0).to("cuda")
            sampled_nodes = [l_i for l_i in range(sample_step*args.hops*(1+args.ns_rate))]
            dim_pairs = [x for x in itertools.combinations(sampled_nodes, r=2)]
            dim_pairs = np.array(dim_pairs).T
            teacher_rank_list = torch.zeros((len(t_r), dim_pairs.shape[1],1)).to(t_r.device)
                      
            mask = t_r[:, dim_pairs[0]] > (t_r[:, dim_pairs[1]] + args.margin)
            teacher_rank_list[mask] = 1
            mask2 = t_r[:, dim_pairs[0]] < (t_r[:, dim_pairs[1]] - args.margin)
            teacher_rank_list[mask2] = -1
            first_rank_list = s_r[:, dim_pairs[0]].squeeze()
            second_rank_list = s_r[:, dim_pairs[1]].squeeze()
            llp_r_loss = margin_rank_loss(first_rank_list, second_rank_list, teacher_rank_list.squeeze())

        if args.datasets != "collab":
            neg_edge = negative_sampling(edge_index, num_nodes=data.x.size(0),
                                 num_neg_samples=link_perm.size(0), method='dense')
        elif args.datasets == "collab":
            neg_edge = torch.randint(0, data.x.size()[0], [edge.size(0), edge.size(1)], dtype=torch.long, device=h.device)

        ### calculate the true_label loss
        train_edges = torch.cat((edge, neg_edge), dim=-1)
        train_label = torch.cat((torch.ones(edge.size()[1]), torch.zeros(neg_edge.size()[1])), dim=0).to(h.device)
        out = predictor(h[train_edges[0]], h[train_edges[1]]).squeeze()
        label_loss = bce_loss(out, train_label)
       
        #t_out = teacher_predictor(t_h[train_edges[0]], t_h[train_edges[1]]).squeeze().detach()

        if args.LLP_D or args.LLP_R:
            loss = args.True_label * label_loss + args.LLP_D * llp_d_loss + args.LLP_R * llp_r_loss
        else:
            loss = args.True_label * label_loss 

        loss.backward()

        torch.nn.utils.clip_grad_norm_(data.x, 1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)

        optimizer.step()

        num_examples = edge.size(1)
        total_loss += loss.item() * num_examples
        total_examples += num_examples
        
    return total_loss / total_examples

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
    parser.add_argument('--runs', type=int, default=5)
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

    Logger_file = "../results/" + args.datasets + "_KD_" + args.transductive + ".txt"
    file = open(Logger_file, "a")
    file.write(str(args)+"\n")
    if args.KD_RM != 0:
        file.write("Logit-matching\n")
    elif args.KD_LM != 0:
        file.write("Representation-matching\n")
    elif args.LLP_D != 0 or args.LLP_R != 0:
        file.write("LLP (Relational Distillation)\n")
    file.close()

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    #device = 'cpu'
    device = torch.device(device)
    mini_batch_device = 'cpu'

    ### Prepare the datasets
    if args.transductive == "transductive":
        if args.datasets != "collab":
            if args.datasets in ['cora', 'citeseer', 'pubmed']:
                heart_data = read_data(args.datasets, 'equal')
                new_data = Data(x=heart_data['x'], adj_t = heart_data['train_pos'].T)
                data = new_data.to(device)
                input_size = data.x.size()[1]
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
                input_size = data.x.size()[1]
                #data = data.to(device)
            args.metric = 'Hits@20'
            if args.minibatch:
                data = data.to(mini_batch_device)
                args.node_batch_size = int(data.x.size()[0] / (split_edge['train']['edge'].size()[0] / args.link_batch_size))
            else:
                data = data.to(device)

        elif args.datasets == "collab":
            """
            dataset = PygLinkPropPredDataset(name='ogbl-collab')
            data = dataset[0]
            edge_index = data.edge_index
            data.edge_weight = data.edge_weight.view(-1).to(torch.float)
            data = T.ToSparseTensor()(data)

            split_edge = dataset.get_edge_split()
            input_size = data.num_features
            data.adj_t = edge_index
            """
            args.metric = 'Hits@50'

    else:
        training_data, val_data, inference_data, data, test_edge_bundle, negative_samples = torch.load("../data/" + args.datasets + "_production.pkl")
        input_size = training_data.x.size(1)

        if args.minibatch:
            training_data.to(mini_batch_device)
        else:
            training_data.to(device)
        val_data.to(device)
        inference_data.to(device)

        args.node_batch_size = int(training_data.x.size()[0] / (training_data.edge_index.size(1) / args.link_batch_size))

#    print(data.adj_t.size())
#    torch.save(data.adj_t, 'cora_adj_t.pth')
#    saved_data = torch.load('/home/qinzongyue/HeaRT/benchmarking/exist_setting_small/cora_train_pos.pth')
#    print(saved_data.size())
    #### Prepare the teacher and student model
    model = MLP(args.num_layers, input_size, args.hidden_channels, args.hidden_channels, args.dropout).to(device)

    predictor = LinkPredictor(args.predictor, args.hidden_channels, args.hidden_channels, 1,
                              args.num_layers, args.dropout).to(device)

    """
    pretrained_model = torch.load("../saved-models/" + args.datasets + "-" + args.encoder + "_" + args.transductive + ".pkl", map_location=device)

    teacher_predictor = LinkPredictor(args.predictor, 256, 256, 1, 2, args.dropout)
    teacher_predictor.load_state_dict(pretrained_model['predictor'], strict=True)
    teacher_predictor.to(device)

    t_h = torch.load("../saved-features/" + args.datasets + "-" + args.encoder + "_" + args.transductive + ".pkl", map_location=device)
    t_h = t_h['features']

    for para in teacher_predictor.parameters():
        para.requires_grad=False
    """

    evaluator = Evaluator(name='ogbl-ddi')

    results, h, pos_pred, pos_edge, neg_pred, neg_edge = test_transductive(model, predictor, data, split_edge,
                            evaluator, args.link_batch_size, 'mlp', args.datasets, args, detail = True)
    run = -1
    epoch = -1
    loss = -1
    for key, result in results.items():
        valid_hits, test_hits = result
        print(key)
        print(f'Run: {run + 1:02d}, '
                                f'Epoch: {epoch:02d}, '
                                f'Loss: {loss:.4f}, '
                                f'Valid: {100 * valid_hits:.2f}%, '
                                f'Test: {100 * test_hits:.2f}%')

    
    
    """ 
    all_pos_samples = []
    all_neg_samples = []
    if args.transductive == "transductive":
        for i in range(1):
            pos_samples, neg_samples = get_samples(model, predictor, data, split_edge,
                                args, device)
            all_pos_samples += pos_samples
            all_neg_samples += neg_samples
        all_pos_samples = torch.cat(all_pos_samples, dim=0)
        all_neg_samples = torch.cat(all_neg_samples, dim=0)
        print(all_pos_samples.size(), all_neg_samples.size())
        torch.save((all_pos_samples, all_neg_samples), f'{args.datasets}_edges.pth') 
    return
    """
    
    
    
    pos_distill_edge, neg_distill_edge = torch.load(f'../data/{args.datasets}_edges.pth', map_location=device)
#    pos_distill_edge, neg_distill_edge = torch.load(f'./{args.datasets}_edges.pth')

    pos_distill_pred, neg_distill_pred = torch.load(args.distill_pred_path, map_location=device)
    print(pos_distill_edge.size(), pos_distill_pred.size())

#    return
    if args.transductive == "transductive":
        if args.datasets != "collab":
            loggers = {
                'Hits@10': Logger(args.runs, args),
                'Hits@20': Logger(args.runs, args),
                'Hits@30': Logger(args.runs, args),
                'Hits@50': Logger(args.runs, args),
                'AUC': Logger(args.runs, args),
            }
        elif args.datasets == "collab":
            loggers = {
                'Hits@10': Logger(args.runs, args),
                'Hits@50': Logger(args.runs, args),
                'Hits@100': Logger(args.runs, args),
                'AUC': Logger(args.runs, args),
            }
    else:
        loggers = {
            'Hits@10': ProductionLogger(args.runs, args),
            'Hits@20': ProductionLogger(args.runs, args),
            'Hits@30': ProductionLogger(args.runs, args),
            'Hits@50': ProductionLogger(args.runs, args),
            'AUC': ProductionLogger(args.runs, args),
        }

    best_states = []
    for run in range(args.runs):
        best_state = {}
        torch_geometric.seed.seed_everything(run+1)
        
        model.reset_parameters()
        predictor.reset_parameters()
        optimizer = torch.optim.Adam(
            list(model.parameters()) +
            list(predictor.parameters()), lr=args.lr)

        cnt_wait = 0
        best_val = 0.0
        best_pred = None

        all_pos_samples = []
        all_neg_samples = []


        for epoch in range(1, 1 + args.epochs):
            if args.transductive == "transductive":
                if args.minibatch:
                    loss = train_minibatch(model, predictor, t_h, teacher_predictor, data, split_edge,
                            optimizer, args, device)
                else:
                    loss = train(model, predictor, pos_distill_pred, neg_distill_pred,
                            pos_distill_edge, neg_distill_edge, data, split_edge,
                                optimizer, args, device)
                
                results, h, pos_pred, pos_edge, neg_pred, neg_edge = test_transductive(model, predictor, data, split_edge,
                            evaluator, args.link_batch_size, 'mlp', args.datasets, args, detail = True)
            
            else:
                loss = train(model, predictor, t_h, teacher_predictor, training_data, None,
                            optimizer, args, device)

                results, h = test_production(model, predictor, val_data, inference_data, test_edge_bundle, negative_samples,
                        evaluator, args.link_batch_size, 'mlp', args.datasets)
            
            if results[args.metric][0] >= best_val:
                best_val = results[args.metric][0]
                best_pred = [pos_pred, pos_edge, neg_pred, neg_edge]
                cnt_wait = 0
                best_state['model'] = deepcopy(model.state_dict())
                best_state['predictor'] = deepcopy(predictor.state_dict())
            else:
                cnt_wait +=1

            for key, result in results.items():
                loggers[key].add_result(run, result)

            if epoch % args.log_steps == 0:
                if args.transductive == "transductive":
                    for key, result in results.items():
                        valid_hits, test_hits = result
                        print(key)
                        print(f'Run: {run + 1:02d}, '
                                f'Epoch: {epoch:02d}, '
                                f'Loss: {loss:.4f}, '
                                f'Valid: {100 * valid_hits:.2f}%, '
                                f'Test: {100 * test_hits:.2f}%')
                else: 
                    for key, result in results.items():
                        valid_hits, test_hits, old_old, old_new, new_new = result
                        print(key)
                        print(f'Run: {run + 1:02d}, '
                                f'Epoch: {epoch:02d}, '
                                f'Loss: {loss:.4f}, '
                                f'valid: {100 * valid_hits:.2f}%, '
                                f'test: {100 * test_hits:.2f}%, '
                                f'old_old: {100 * old_old:.2f}%, '
                                f'old_new: {100 * old_new:.2f}%, '
                                f'new_new: {100 * new_new:.2f}%')
                print('---')

            if cnt_wait >= args.patience:
                break

        best_states.append(best_state)
        for key in loggers.keys():
            print(key)
            loggers[key].print_statistics(run)

        #TODO check node-wise
#        analyze(best_pred, data, 20)
#        xxx = input('this run pause')

    file = open(Logger_file, "a")
    file.write(f'All runs:\n')

    return_results = {}
    if args.transductive == "transductive":
        for key in loggers.keys():
            print(key)
            loggers[key].print_statistics()
            return_results[key] = loggers[key].return_statistics()


            file.write(f'{key}:\n')
            best_results = []
            for r in loggers[key].results:
                r = 100 * torch.tensor(r)
                valid = r[:, 0].max().item()
                test1 = r[r[:, 0].argmax(), 1].item()
                best_results.append((valid, test1))

            best_result = torch.tensor(best_results)

            r = best_result[:, 1]
            file.write(f'Test: {r.mean():.4f} ± {r.std():.4f}\n')
    else:
        for key in loggers.keys():
            print(key)
            loggers[key].print_statistics()

            file.write(f'{key}:\n')
            best_results = []
            for r in loggers[key].results:
                r = 100 * torch.tensor(r)
                val = r[r[:, 0].argmax(), 0].item()
                test_r = r[r[:, 0].argmax(), 1].item()
                old_old = r[r[:, 0].argmax(), 2].item()
                old_new = r[r[:, 0].argmax(), 3].item()
                new_new = r[r[:, 0].argmax(), 4].item()
                best_results.append((val, test_r, old_old, old_new, new_new))

            best_result = torch.tensor(best_results)

            r = best_result[:, 0]
            file.write(f'  Final val: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 1]
            file.write(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 2]
            file.write(f'   Final old_old: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 3]
            file.write(f'   Final old_new: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 4]
            file.write(f'   Final new_new: {r.mean():.2f} ± {r.std():.2f}\n')
    file.close()
    return return_results, best_states

args = parse_args()
distill_teacher = args.distill_teacher
dataset = args.datasets
distill_pred_path = f'/local2/qzy_scai/EHDM/HeaRT/benchmarking/exist_setting_small/saved/{dataset}_{distill_teacher}_pred.pth'
args.distill_pred_path = distill_pred_path

_, best_states = main(args)
torch.save(best_states, f'saved_students/{dataset}_{distill_teacher}_MLPs_2.pth')
