import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import scipy.sparse as ssp
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
from models import MLP, GCN, SAGE, LinkPredictor, Gater
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
from get_heuristic import *
from utils import get_metric_score_citation2
from tqdm import tqdm

dir_path  = '/local2/qzy_scai/EHDM/HeaRT/'
@torch.no_grad()
def test_edge(gater, model_list, heurstics, h_list, edges, x, batch_size, args, mrr_mode=False, negative_data=None):

    
    preds = []

    if mrr_mode:
        source = edges.t()[0]
        source = source.view(-1, 1).repeat(1, 1000).view(-1)
        target_neg = negative_data.view(-1)

        for perm in tqdm(DataLoader(range(source.size(0)), batch_size)):
            src, dst_neg = source[perm], target_neg[perm]
            if args.disable_heuristics == True:
                h_value = None
            else:
                h_value = heuristics[perm]
            edge = torch.stack((src, dst_neg), dim=0)
            pred = get_gater_out(gater, model_list, h_value, h_list, None, x, edge, args).squeeze() 
            preds.append(pred)
#            preds += [score_func(h[src], h[dst_neg]).squeeze().cpu()]
        pred_all = torch.cat(preds, dim=0).view(-1, 1000).cpu()

    else:

        for perm  in tqdm(DataLoader(range(edges.size(0)), batch_size)):
            edge = edges[perm].t()
            if args.disable_heuristics == True:
                h_value = None
            else:
                h_value = heuristics[perm]
             
            preds.append(get_gater_out(gater, model_list, h_value, h_list, None, x, edge, args))
   #         preds += [score_func(h[edge[0]], h[edge[1]]).cpu()]
            
        pred_all = torch.cat(preds, dim=0).cpu()


    return pred_all


@torch.no_grad()
def test_citation2(gater, model_list, heuristic_list,  
                        pos_valid_heuristics, neg_valid_heuristics, 
                        pos_test_heuristics, neg_test_heuristics, 
                        data, evaluation_edges, emb, evaluator_hit, evaluator_mrr, batch_size, h_list):
    assert args.disable_heuristics == True
    gater.eval()
    for model in model_list:
        model['model'].eval()
        model['predictor'].eval()

    train_val_edge, pos_valid_edge, neg_valid_edge, pos_test_edge,  neg_test_edge = evaluation_edges
    x = data.x
    A = None

    # print(h[0][:10])
    train_val_edge = train_val_edge.to(x.device)
    pos_valid_edge = pos_valid_edge.to(x.device) 
    neg_valid_edge = neg_valid_edge.to(x.device)
    pos_test_edge = pos_test_edge.to(x.device) 
    neg_test_edge = neg_test_edge.to(x.device)

    neg_valid_pred = test_edge(gater, model_list, neg_valid_heuristics, h_list, pos_valid_edge, x, batch_size, args, mrr_mode=True, negative_data=neg_valid_edge)

    pos_valid_pred = test_edge(gater, model_list, pos_valid_heuristics, h_list, pos_valid_edge, x, batch_size, args)

    start_time = time.perf_counter()
    pos_test_pred = test_edge(gater, model_list, pos_test_heuristics, h_list, pos_test_edge, x, batch_size, args)

    neg_test_pred = test_edge(gater, model_list, pos_test_heuristics, h_list, pos_test_edge, x, batch_size, args, mrr_mode = True, negative_data=neg_test_edge)
    end_time = time.perf_counter()
    infer_time = end_time - start_time

    pos_train_pred = test_edge(gater, model_list, None, h_list, train_val_edge, x, batch_size, args)

        
    pos_valid_pred = pos_valid_pred.view(-1)
    pos_test_pred =pos_test_pred.view(-1)
    pos_train_pred = pos_valid_pred.view(-1)
    
#    print('train valid_pos valid_neg test_pos test_neg', pos_train_pred.size(), pos_valid_pred.size(), neg_valid_pred.size(), pos_test_pred.size(), neg_test_pred.size())
    
    result = get_metric_score_citation2(evaluator_hit, evaluator_mrr, pos_train_pred, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred)
    
#    score_emb = [pos_valid_pred.cpu(),neg_valid_pred.cpu(), pos_test_pred.cpu(), neg_test_pred.cpu(), x1.cpu(), x2.cpu()]
    score_emb = None

    return result, infer_time

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
    data['A'] = A
    data['train_pos'] = train_pos_tensor
    data['train_val'] = train_val

    data['valid_pos'] = valid_pos
    data['valid_neg'] = valid_neg
    data['test_pos'] = test_pos
    data['test_neg'] = test_neg

    data['x'] = feature_embeddings

    return data

def get_gater_out(gater, mlp_list, heuristic_values, h_list, A, x, edge, args, return_weight=False):
#    heuristic_val_list = []
#    for heuristic in heuristic_list:
#        val = eval(heuristic)(A, edge.cpu()).squeeze()
#        heuristic_val_list.append(val.to(x.device))
#    heuristic_values = torch.stack(heuristic_val_list, dim=1)

    device = x.device
    if len(mlp_list) > 0:
        out_list = []
        for h, model in zip(h_list, mlp_list):
            out = model['predictor'](h[edge[0]].to(device), h[edge[1]].to(device)).squeeze()
            out_list.append(out)
        predictions = torch.stack(out_list, dim=1)
        if args.disable_heuristics:
            heuristic_values = None
        else:
            heuristic_values = heuristic_values.to(x.device)

        if args.add_heuristics:
            predictions = torch.cat((predictions, heuristic_values.to(x.device)), dim=1)
    else:
        predictions = heuristic_values.to(device)

    #heuristic_values = heuristic_values.to(device)
    out, weight = gater(x[edge[0]], x[edge[1]], heuristic_values, predictions, return_weight=True)
    if return_weight == False:
        return out.squeeze()
    else:
        return out.squeeze(), weight

@torch.no_grad()
def get_score(gater, model_list, pos_edge, neg_edge, 
              pos_heuristics, neg_heuristics, data, 
              batch_size, args):
    gater.eval()
    for model in model_list:
        model['model'].eval()
        model['predictor'].eval()

    h_list = [model['model'](data.x) for model in model_list]
    x = data.x
    A = data['A']
    pos_edge = pos_edge.to(x.device)
    neg_edge = neg_edge.to(x.device)

    pos_preds = []
#    xxx = input("pause")
    for perm in DataLoader(range(pos_edge.size(1)), batch_size):
        edge = pos_edge[:,perm]
        heuristics = pos_heuristics[perm]
        outs = get_gater_out(gater, model_list, heuristics, h_list, A, x, edge, args)
        pos_preds += [outs.squeeze().cpu()]
    pos_pred = torch.cat(pos_preds, dim=0)

    neg_preds = []
    for perm in DataLoader(range(neg_edge.size(1)), batch_size):
        edge = neg_edge[:,perm]
        heuristics = neg_heuristics[perm]
        outs = get_gater_out(gater, model_list, heuristics, h_list, A, x, edge, args)
        neg_preds += [outs.squeeze().cpu()]
    neg_pred = torch.cat(neg_preds, dim=0)

    return pos_pred, neg_pred



@torch.no_grad()
def test_gater_transductive(gater, mlp_list, heuristic_list, 
        pos_valid_heuristics, neg_valid_heuristics,
        pos_test_heuristics, neg_test_heuristics,
        data, split_edge, evaluator, batch_size, dataset, args, h_list, detail=False):
    gater.eval()
    for model in mlp_list:
        model['model'].eval()
        model['predictor'].eval()

#    h_list = [model['model'](data.x) for model in mlp_list]
    x = data.x
    #A = data['A']
    A = None

    pos_valid_edge = split_edge['valid']['edge'].to(x.device)
    neg_valid_edge = split_edge['valid']['edge_neg'].to(x.device)
    pos_test_edge = split_edge['test']['edge'].to(x.device)
    neg_test_edge = split_edge['test']['edge_neg'].to(x.device)

    pos_valid_preds = []
    for perm in DataLoader(range(pos_valid_edge.size(0)), batch_size):
        edge = pos_valid_edge[perm].t()
        if args.disable_heuristics == True:
            heuristics = None
        else:
            heuristics = pos_valid_heuristics[perm]
        outs = get_gater_out(gater, mlp_list, heuristics, h_list, A, x, edge, args)
        pos_valid_preds += [outs.squeeze().cpu()]
    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

    neg_valid_preds = []
    for perm in DataLoader(range(neg_valid_edge.size(0)), batch_size):
        edge = neg_valid_edge[perm].t()
        if args.disable_heuristics == True:
            heuristics = None
        else:
            heuristics = neg_valid_heuristics[perm]
        outs = get_gater_out(gater, mlp_list, heuristics, h_list, A, x, edge, args)
        neg_valid_preds += [outs.squeeze().cpu()]
    neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

    start_time = time.perf_counter() 
    pos_test_preds = []
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
        edge = pos_test_edge[perm].t()
        if args.disable_heuristics == True:
            heuristics = None
        else:
            heuristics = pos_test_heuristics[perm]
        outs = get_gater_out(gater, mlp_list, heuristics, h_list, A, x, edge, args)
        pos_test_preds += [outs.squeeze().cpu()]
    pos_test_pred = torch.cat(pos_test_preds, dim=0)

    neg_test_preds = []
    for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
        edge = neg_test_edge[perm].t()
        if args.disable_heuristics == True:
            heuristics = None
        else:
            heuristics = neg_test_heuristics[perm]
        outs = get_gater_out(gater, mlp_list, heuristics, h_list, A, x, edge, args)
        neg_test_preds += [outs.squeeze().cpu()]
    neg_test_pred = torch.cat(neg_test_preds, dim=0)
    end_time = time.perf_counter() 
    infer_time = end_time - start_time

    results = {}
    if dataset not in ["collab", "igb"]:
        for K in [10, 20, 30, 50]:
            evaluator.K = K
            valid_hits = evaluator.eval({
                'y_pred_pos': pos_valid_pred,
                'y_pred_neg': neg_valid_pred,
            })[f'hits@{K}']
            test_hits = evaluator.eval({
                'y_pred_pos': pos_test_pred,
                'y_pred_neg': neg_test_pred,
            })[f'hits@{K}']

            results[f'Hits@{K}'] = (valid_hits, test_hits)
    else:
        for K in [10, 50, 100]:
            evaluator.K = K
 
            valid_hits = evaluator.eval({
                'y_pred_pos': pos_valid_pred,
                'y_pred_neg': neg_valid_pred,
            })[f'hits@{K}']
            test_hits = evaluator.eval({
                'y_pred_pos': pos_test_pred,
                'y_pred_neg': neg_test_pred,
            })[f'hits@{K}']

            results[f'Hits@{K}'] = (valid_hits, test_hits)

    valid_result = torch.cat((torch.ones(pos_valid_pred.size()), torch.zeros(neg_valid_pred.size())), dim=0)
    valid_pred = torch.cat((pos_valid_pred, neg_valid_pred), dim=0)

    test_result = torch.cat((torch.ones(pos_test_pred.size()), torch.zeros(neg_test_pred.size())), dim=0)
    test_pred = torch.cat((pos_test_pred, neg_test_pred), dim=0)

    results['AUC'] = (roc_auc_score(valid_result.cpu().numpy(),valid_pred.cpu().numpy()),roc_auc_score(test_result.cpu().numpy(),test_pred.cpu().numpy()))

    if detail == False:
        return results, infer_time
    else:
        pos_pred = torch.cat([pos_valid_pred, pos_test_pred], dim=0)
        pos_edge = torch.cat([pos_valid_edge, pos_test_edge], dim=0)
        neg_pred = torch.cat([neg_valid_pred, neg_test_pred], dim=0)
        neg_edge = torch.cat([neg_valid_edge, neg_test_edge], dim=0)
        return results, pos_pred, pos_edge, neg_pred, neg_edge

def train(ensemble_gater, mlp_list, heuristic_list, pos_heuristic_vals, neg_heuristic_vals,
        data, split_edge, optimizer, batch_size, args, transductive, h_list):
    return 0
    if transductive == "transductive":
        row, col = data.adj_t
#        pos_train_edge = split_edge['train']['edge'].to(data.x.device)
#        neg_train_edge = split_edge['train']['edge_neg'].to(data.x.device)
        pos_train_edge = split_edge['train']['edge'].to('cpu')
        try:
            neg_train_edge = split_edge['train']['edge_neg'].to('cpu')
        except:
            neg_train_edge = None

#        print(neg_train_edge.size())
    else:
        row, col = data.edge_index
        pos_train_edge = data.edge_index.t()        
        neg_train_edge = split_edge['train']['edge_neg'].to(data.x.device)

    edge_index = torch.stack([col, row], dim=0)
    x = data['x']

    ensemble_gater.train()
    for model in mlp_list:
        model['model'].eval()
        model['predictor'].eval()

    bce_loss = nn.BCELoss()
    total_loss = total_examples = 0
    
    if neg_train_edge is not None:
        neg_edge_loader = iter(DataLoader(range(neg_train_edge.size(1)), args.link_batch_size, shuffle=True))
   # print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
   # print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
   # print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))

    for link_perm in tqdm(DataLoader(range(pos_train_edge.size(0)), args.link_batch_size, shuffle=True)):
        optimizer.zero_grad()


#        h = model(data.x)

        edge = pos_train_edge[link_perm].t()#.to(data.x.device)
#        print(pos_heuristic_vals.size())
#        print(link_perm.max())
        if pos_heuristic_vals is not None:
            pos_heuristics = pos_heuristic_vals[link_perm].to(data.x.device)
        else:
            pos_heuristics = None

        if neg_heuristic_vals is not None:
            neg_link_perm = next(neg_edge_loader)
#        print(neg_link_perm.max())
#        print(neg_train_edge.size())
            neg_edge = neg_train_edge[:,neg_link_perm]
            neg_heuristics = neg_heuristic_vals[neg_link_perm].to(data.x.device)
        else:
            neg_edge = torch.randint(0, data.x.size()[0], edge.size(), dtype=torch.long)
#                             device=data.x.device)
            neg_heuristics = None#pos_heuristics * 0


        optimizer.zero_grad()



        ### calculate the true_label loss
        train_edges = torch.cat((edge, neg_edge), dim=-1)
        if pos_heuristics is not None:
            heuristic_vals = torch.cat((pos_heuristics, neg_heuristics), dim=0)
        else:
            heuristic_vals = None
   #     print(3)
   #     print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
   #     print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
   #     print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))

        train_label = torch.cat((torch.ones(edge.size()[1]), torch.zeros(neg_edge.size()[1])), dim=0).to(x.device)

   #     print(4)
   #     print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
   #     print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
   #     print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))

        out, weight = get_gater_out(ensemble_gater, mlp_list, heuristic_vals, h_list, None, x, train_edges, args, return_weight=True)
   #     print(5)
   #     print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
   #     print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
   #     print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))


#        print(out.size(), train_label.size())
#        print(out.isnan().any(), out.isinf().any())
#        print(out.min(), out.max())
        l1_loss = torch.sum(weight.abs(), dim=1).mean()
        loss = bce_loss(out, train_label) + args.l1 * l1_loss

        loss.backward(retain_graph=True)

        torch.nn.utils.clip_grad_norm_(data.x, 1.0)
        torch.nn.utils.clip_grad_norm_(ensemble_gater.parameters(), 1.0)

        optimizer.step()

        num_examples = edge.size(1)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples

def get_heuristic_values(split_edge, data, heuristic_list, pos_distill_edge, neg_distill_edge, args):
    pos_heuristic_vals = []
    neg_heuristic_vals = []
    pos_test_heuristics = []
    neg_test_heuristics = []
    pos_valid_heuristics = []
    neg_valid_heuristics = []
    pos_distill_heuristics = []
    neg_distill_heuristics = []

    for heuristic in heuristic_list:
        val = eval(heuristic)(data.A, split_edge['train']['edge'].T.cpu()).squeeze()
#        pos_heuristic_vals.append(val.to(data.x.device))
        pos_heuristic_vals.append(val)

        val = eval(heuristic)(data.A, split_edge['train']['edge_neg'].cpu()).squeeze()
#        neg_heuristic_vals.append(val.to(data.x.device))
        neg_heuristic_vals.append(val)

        val = eval(heuristic)(data.A, split_edge['test']['edge'].T).squeeze()
#        pos_test_heuristics.append(val.to(device))
        pos_test_heuristics.append(val)

        val = eval(heuristic)(data.A, split_edge['test']['edge_neg'].T).squeeze()
        neg_test_heuristics.append(val)

        val = eval(heuristic)(data.A, split_edge['valid']['edge'].T).squeeze()
        pos_valid_heuristics.append(val)
        val = eval(heuristic)(data.A, split_edge['valid']['edge_neg'].T).squeeze()
        neg_valid_heuristics.append(val)

        if args.edge_path_file != 'None':
            val = eval(heuristic)(data.A, pos_distill_edge).squeeze()
            pos_distill_heuristics.append(val)
            val = eval(heuristic)(data.A, neg_distill_edge).squeeze()
            neg_distill_heuristics.append(val)


    pos_heuristic_values = torch.stack(pos_heuristic_vals, dim=1)
    neg_heuristic_values = torch.stack(neg_heuristic_vals, dim=1)
    h_values = torch.cat((pos_heuristic_values, neg_heuristic_values), dim=0)
    # compute max and min of each column of h_values
    # normalize all the heuristic values based on min and max
    # Compute max and min of each column of h_values
    max_vals, _ = h_values.max(dim=0, keepdim=True)  # Shape: (1, num_columns)
    min_vals, _ = h_values.min(dim=0, keepdim=True)  # Shape: (1, num_columns)

    # Normalize h_values based on min and max
    pos_heuristic_values = (pos_heuristic_values - min_vals) / (max_vals - min_vals + 1e-8)
    neg_heuristic_values = (neg_heuristic_values - min_vals) / (max_vals - min_vals + 1e-8)

    pos_test_heuristics = torch.stack(pos_test_heuristics, dim=1)
    neg_test_heuristics = torch.stack(neg_test_heuristics, dim=1)
    pos_valid_heuristics = torch.stack(pos_valid_heuristics, dim=1)
    neg_valid_heuristics = torch.stack(neg_valid_heuristics, dim=1)

    # Normalize test and validation heuristics using the same min and max
    pos_test_heuristics = (pos_test_heuristics - min_vals) / (max_vals - min_vals + 1e-8)
    neg_test_heuristics = (neg_test_heuristics - min_vals) / (max_vals - min_vals + 1e-8)
    pos_valid_heuristics = (pos_valid_heuristics - min_vals) / (max_vals - min_vals + 1e-8)
    neg_valid_heuristics = (neg_valid_heuristics - min_vals) / (max_vals - min_vals + 1e-8)

    if args.edge_path_file != 'None':
        pos_distill_heuristics = torch.stack(pos_distill_heuristics, dim=1)
        neg_distill_heuristics = torch.stack(neg_distill_heuristics, dim=1)
        pos_distill_heuristics = (pos_distill_heuristics - min_vals) / (max_vals - min_vals + 1e-8)
        neg_distill_heuristics = (neg_distill_heuristics - min_vals) / (max_vals - min_vals + 1e-8)
    else:
        pos_distill_heuristics = None
        neg_distill_heuristics = None
    return pos_heuristic_values, neg_heuristic_values, pos_valid_heuristics, neg_valid_heuristics,\
        pos_test_heuristics, neg_test_heuristics, pos_distill_heuristics, neg_distill_heuristics 


def parse_args():
    parser = argparse.ArgumentParser(description='OGBL-DDI (GNN)')

    parser.add_argument('--label', type=str, default='ensembled_mlp', help='label of this experiment')
    parser.add_argument('--l1', type=float, default=0, help='l1 regularization loss for gater weight')
    parser.add_argument('--edge_path_file', type=str, default='None')
    parser.add_argument('--add_heuristics', action='store_true')
    parser.add_argument('--add_prediction', action='store_true')
    parser.add_argument('--disable_heuristics', action='store_true')
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
    parser.add_argument('--patience', type=int, default=10, help='number of patience steps for early stopping')
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


def main(teacher_list, heuristic_list, args):
#    args = parse_args()

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
    #mini_batch_device = 'cpu'

    ### Prepare the datasets
    if args.transductive == "transductive":
        if args.datasets == 'igb':
            from igb.dataloader import IGB260M
#            from torch_geometric.data import Data
            igb_data = IGB260M('/local2/qzy_scai/IGB-Datasets/', size='small', in_memory=1, classes=19, synthetic=0)
            data = Data(x = torch.Tensor(igb_data.paper_feat.copy()),
                    edge_index = torch.Tensor(igb_data.paper_edge.copy()).T.long(),
                    num_nodes = igb_data.num_nodes(),)
            edge_index = data.edge_index
            data = T.ToSparseTensor()(data)
            path_name = '/local2/qzy_scai/IGB-Datasets/small-split-edges.pkl'
            split_edge = torch.load(path_name)
            node_num = data.num_nodes
            input_size = data.x.size()[1]
            data.adj_t = edge_index
            args.metric = 'Hits@100'

        elif args.datasets not in ["collab", "citation2"]:
            if args.datasets in ['cora', 'citeseer', 'pubmed']:
                heart_data = read_data(args.datasets, 'equal')
                data = Data(x=heart_data['x'], adj_t = heart_data['train_pos'].T, A=heart_data['A'])
                #data = new_data.to(device)
                input_size = data.x.size()[1]
                split_edge = {}
                split_edge['train'] = {'edge': heart_data['train_pos']}
                split_edge['valid'] = {'edge': heart_data['valid_pos'], 'edge_neg': heart_data['valid_neg']}
                split_edge['test'] = {'edge': heart_data['test_pos'], 'edge_neg': heart_data['test_neg']}
                if args.datasets != "collab":
                    neg_edge = negative_sampling(data.adj_t, num_nodes=data.x.size(0),
                             num_neg_samples=split_edge['train']['edge'].numel()*4, method='dense')
                elif args.datasets == "collab":
                    edge = heart_data['train_pos']
                    neg_edge = torch.randint(0, data.x.size()[0], [edge.size(0)*4, edge.size(1)*4], dtype=torch.long, device=x.device)
                split_edge['train']['edge_neg'] = neg_edge

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

                edge_weight = torch.ones(edge_index.size(1))
                num_nodes = data.x.size(0)
                data.A = ssp.csr_matrix((edge_weight.view(-1), (edge_index[0], edge_index[1])), shape=(num_nodes, num_nodes)) 
                neg_edge = negative_sampling(data.adj_t, num_nodes=data.x.size(0),
                             num_neg_samples=split_edge['train']['edge'].numel()*4, method='dense')
                split_edge['train']['edge_neg'] = neg_edge

                #data = data.to(device)
            args.metric = 'Hits@20'
            if args.minibatch:
                data = data.to(mini_batch_device)
                args.node_batch_size = int(data.x.size()[0] / (split_edge['train']['edge'].size()[0] / args.link_batch_size))
            else:
                data = data.to(device)

        else: # collab or citation2
            dataset = PygLinkPropPredDataset(name=f'ogbl-{args.datasets}')
            data = dataset[0]
            edge_index = data.edge_index
            if hasattr(data, "edge_weight") and data.edge_weight is not None:
                data.edge_weight = data.edge_weight.view(-1).to(torch.float)
            data = T.ToSparseTensor()(data)

            split_edge = dataset.get_edge_split()
            input_size = data.num_features
            data.adj_t = edge_index
            
            if args.datasets == 'collab':
                args.metric = 'Hits@50'
            else:
                args.metric = 'MRR'

        if args.use_valedges_as_input:
            val_edge_index = split_edge['valid']['edge'].t()
            full_edge_index = torch.cat([edge_index, val_edge_index], dim=-1)
            if args.datasets not in ["collab", "citation2"]:
                data.full_adj_t = full_edge_index
            else:
                data.full_adj_t = SparseTensor.from_edge_index(full_edge_index).t()
                data.full_adj_t = data.full_adj_t.to_symmetric()
        else:
            data.full_adj_t = data.adj_t

        if args.minibatch:
            data = data.to(mini_batch_device)
        else:
            data = data.to(device)

        if args.datasets == 'citation2':
            source_edge, target_edge = split_edge['train']['source_node'], split_edge['train']['target_node']
            pos_train_edge = torch.cat([source_edge.unsqueeze(1), target_edge.unsqueeze(1)], dim=-1)
            split_edge['train']['edge'] = pos_train_edge

            source, target = split_edge['valid']['source_node'],  split_edge['valid']['target_node']
            pos_valid_edge = torch.cat([source.unsqueeze(1), target.unsqueeze(1)], dim=-1)
            neg_valid_edge = split_edge['valid']['target_node_neg'] 

            source, target = split_edge['test']['source_node'],  split_edge['test']['target_node']
            pos_test_edge = torch.cat([source.unsqueeze(1), target.unsqueeze(1)], dim=-1)
            neg_test_edge = split_edge['test']['target_node_neg']

            idx = torch.randperm(pos_train_edge.size(0))[:pos_valid_edge.size(0)]
            train_val_edge = pos_train_edge[idx]

            evaluation_edges = [train_val_edge, pos_valid_edge, neg_valid_edge, pos_test_edge,  neg_test_edge]

        args.node_batch_size = int(data.x.size()[0] / (split_edge['train']['edge'].size()[0] / args.link_batch_size))

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


    if args.edge_path_file != 'None':
        print(f'loading from {args.edge_path_file}')
        edges = torch.load(args.edge_path_file)
        pos_edge, neg_edge = edges[0], edges[1]

        node = pos_edge[:,0]
        pos_nb = pos_edge[:,1:]
        neg_nb = neg_edge

        src_nodes = node.repeat_interleave(pos_nb.size(1))
        pos_distill_edge = torch.stack((src_nodes, pos_nb.flatten()), dim=0)
        src_nodes = node.repeat_interleave(neg_nb.size(1))
        neg_distill_edge = torch.stack((src_nodes, neg_nb.flatten()), dim=0)
    else:
        pos_distill_edge = neg_distill_edge = None


    if args.disable_heuristics == True:
        pos_heuristic_values = None#torch.zeros(split_edge['train']['edge'].size(0), 1)
        try:
            neg_heuristic_values = None#torch.zeros(split_edge['train']['edge_neg'].size(1), 1)
        except:
            neg_heuristic_values = None
        pos_valid_heuristics = None#torch.zeros(split_edge['valid']['edge'].size(0), 1)
        neg_valid_heuristics = None#torch.zeros(split_edge['valid']['edge_neg'].size(0), 1)
        pos_test_heuristics = None#torch.zeros(split_edge['test']['edge'].size(0), 1)
        neg_test_heuristics = None#torch.zeros(split_edge['test']['edge_neg'].size(0), 1)
        if args.edge_path_file != 'None':
            pos_distill_heuristics = None#torch.zeros(pos_distill_edge.size(1), 1)
            neg_distill_heuristics = None#torch.zeros(neg_distill_edge.size(1), 1)
    else:
        pos_heuristic_values, neg_heuristic_values, pos_valid_heuristics, neg_valid_heuristics,\
        pos_test_heuristics, neg_test_heuristics, pos_distill_heuristics, neg_distill_heuristics = get_heuristic_values(split_edge, data, heuristic_list, pos_distill_edge, neg_distill_edge, args)

#    print(data.adj_t.size())
#    torch.save(data.adj_t, 'cora_adj_t.pth')
#    saved_data = torch.load('/home/qinzongyue/HeaRT/benchmarking/exist_setting_small/cora_train_pos.pth')
#    print(saved_data.size())
    #### Prepare the teacher and student model
    states_list = []
    model_list = []
    runs = 5
    dataset = args.datasets
    load_num_layers = args.num_layers
    load_hidden_channels = args.hidden_channels
    for distill_teacher in teacher_list: 
        if distill_teacher == 'none':
            if dataset == 'igb':
                states = torch.load(f'saved_students/{dataset}_mlp_model_256.pth', map_location='cpu')

            else:
                states = torch.load(f'saved_students/{dataset}_mlp_model.pth', map_location='cpu')
            new_states = []
            for state in states:
                new_state = {}
                for key, value in state['model'].items():
                    # Rename keys: replace "lins." with "layers."
                   if key.startswith("lins"):
                        new_key = key.replace("lins", "layers")
                   else:
                        new_key = key
                   new_state[new_key] = value
                state['model'] = new_state

                new_state = {}
                for key, value in state['predictor'].items():
                    # Rename keys: replace "lins." with "layers."
                   if key.startswith("lins") and False:
                        new_key = key.replace("lins", "layers")
                   else:
                        new_key = key
                   new_state[new_key] = value
                state['predictor'] = new_state

                new_states.append(state)
#                print(new_states)
#                xxx = input("pause")
            states = new_states
        else:
            try:
                states = torch.load(f'saved_students/{dataset}_{distill_teacher}_MLPs.pth',
                               map_location='cpu')
            except:
                states = torch.load(f'saved_students/{dataset}_{distill_teacher}_MLPs_2.pth',
                               map_location='cpu')

        print(len(states))
        if len(states) < runs:
            runs = len(states)
        states_list.append(states)
        if distill_teacher == 'none' and dataset == 'citation2':
            load_hidden_channels = 128
        else:
            load_hidden_channels = args.hidden_channels
        model = MLP(load_num_layers, input_size, load_hidden_channels, load_hidden_channels, 0).to(device)

        print(model.state_dict().keys())
        predictor = LinkPredictor(args.predictor, load_hidden_channels, load_hidden_channels, 1,
                              load_num_layers, 0).to(device)
        model_list.append({'model': model, 'predictor': predictor})

    print(f"total run: {runs}")
    args.runs = runs



    evaluator = Evaluator(name='ogbl-ddi')
    if args.datasets == 'citation2':
        evaluator_hit = Evaluator(name='ogbl-collab')
        evaluator_mrr = Evaluator(name='ogbl-citation2')

    if args.transductive == "transductive":
        if args.datasets not in ["collab", "citation2", "igb"]:
            loggers = {
                'Hits@10': Logger(args.runs, args),
                'Hits@20': Logger(args.runs, args),
                'Hits@30': Logger(args.runs, args),
                'Hits@50': Logger(args.runs, args),
                'AUC': Logger(args.runs, args),
            }
        elif args.datasets == "collab" or args.datasets == "igb":
            loggers = {
                'Hits@10': Logger(args.runs, args),
                'Hits@50': Logger(args.runs, args),
                'Hits@100': Logger(args.runs, args),
                'AUC': Logger(args.runs, args),
            }
        elif args.datasets == 'citation2':
            loggers = {
              'MRR': Logger(args.runs),
              'mrr_hit20':  Logger(args.runs),
              'mrr_hit50':  Logger(args.runs),
              'mrr_hit100':  Logger(args.runs),
              'mrr_hit200':  Logger(args.runs),
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

    gater = Gater(
            2,
            len(teacher_list) if args.disable_heuristics else len(heuristic_list),
            input_size,
            args.hidden_channels,
            len(teacher_list) + len(heuristic_list) if args.add_heuristics else len(teacher_list),
            args.dropout,
            add_prediction = args.add_prediction,
            disable_heuristics = args.disable_heuristics,
            ).to(device)
   # print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
   # print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
   # print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))

    infer_time_list = []
    train_time_list = []
    for run in range(runs):
        best_state = {}
        train_start_time = time.perf_counter()
        torch_geometric.seed.seed_everything(run+1)
        
        gater.reset_parameters()
        optimizer = torch.optim.Adam(gater.parameters(),
            lr=args.lr)

        cnt_wait = 0
        best_val = 0.0
        best_pred = None

        all_pos_samples = []
        all_neg_samples = []
        for model, states in zip(model_list, states_list):
            model['model'].load_state_dict(states[run]['model'])
            model['predictor'].load_state_dict(states[run]['predictor'])
            model['model'].to(device)
            model['predictor'].to(device)

        with torch.no_grad():
            h_list = [model['model'](data.x) for model in model_list]


        for epoch in range(1, 1 + args.epochs):
            if args.transductive == "transductive":
                print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
                print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
                print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))

                loss = train(gater, model_list, heuristic_list, pos_heuristic_values, neg_heuristic_values,
                        data, split_edge, optimizer, args.link_batch_size, args, args.transductive, h_list)
                #TODO modify test_transductive
                if args.datasets == 'citation2':
                    results, infer_time = test_citation2(gater, model_list, heuristic_list,  
                        pos_valid_heuristics, neg_valid_heuristics, 
                        pos_test_heuristics, neg_test_heuristics, 
                        data, evaluation_edges, None, evaluator_hit, evaluator_mrr, args.link_batch_size, h_list)
                else:

                    results, infer_time = test_gater_transductive(gater, model_list, heuristic_list, 
                        pos_valid_heuristics, neg_valid_heuristics, 
                        pos_test_heuristics, neg_test_heuristics, data, 
                        split_edge, evaluator, args.link_batch_size, dataset, args, h_list, detail=False)

                infer_time_list.append(infer_time)
            
            else:
                loss = train(gater, model_list, heuristic_list, data, split_edge, optimizer, batch_size, args, transductive)
                #TODO modify test_transductive
                results, h = test_gater_production(model, predictor, val_data, inference_data, test_edge_bundle, negative_samples,
                        evaluator, args.link_batch_size, 'mlp', args.datasets)
            
            if results[args.metric][0] > best_val:
                if args.edge_path_file != 'None':
                    pos_pred, neg_pred = get_score(gater, model_list, pos_distill_edge, neg_distill_edge, 
                                                    pos_distill_heuristics, neg_distill_heuristics, 
                                                    data, args.link_batch_size, args)
                    pos_pred = pos_pred.flatten()
                    neg_pred = neg_pred.flatten()
        
                    pos_distill_pred = pos_pred.reshape(node.size(0), -1)
                    neg_distill_pred = neg_pred.reshape(node.size(0), -1)

                best_val = results[args.metric][0]
                #best_pred = [pos_pred, pos_edge, neg_pred, neg_edge]
                cnt_wait = 0
                best_state['gater'] = deepcopy(gater.state_dict())
                best_state['experts'] = [states[run] for states in states_list] 
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

        train_time = time.perf_counter() - train_start_time
        train_time_list.append(train_time)
        best_states.append(best_state)
        for key in loggers.keys():
            print(key)
            loggers[key].print_statistics(run)

    print(f'train time {np.mean(train_time_list):.4f} ({np.std(train_time_list):.4f})')
    print(f'infer time {np.mean(infer_time_list):.4f} ({np.std(infer_time_list):.4f})')

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
    torch.save(best_states, f'saved_students/{args.datasets}_mlp_gater.pth')
    if args.edge_path_file != 'None':
        torch.save((pos_distill_pred, neg_distill_pred), f'EH_{len(heuristic_list)}_{args.datasets}_{args.label}_pred.pth')
    return return_results, best_states 

teacher_list = ['CN', 'AA']
#teacher_list = [] 

heuristic_list = ['AA', 'RA', 'CN', 'capped_shortest_path']
#heuristic_list = ['CN']
#main(teacher_list, heuristic_list)
select_key = 'Hits@20' 

args = parse_args()
for dataset in [args.datasets]:
    grid_search_file = open(f'grid_search_results/average_{len(heuristic_list)}_{dataset}_{args.add_prediction}.txt', 'w')
    best_results = None
    best_param = None
    best_valid_perf = -100
    best_states = None
    if dataset == 'collab':
        select_key = 'Hits@50'
    elif dataset == 'igb':
        select_key = 'Hits@100'
    elif dataset == 'citation2':
        select_key = 'MRR'
    else:
        select_key = 'Hits@20'


    for num_layers in [2]:
        for l1 in [0.1]:
            for dropout in [0.]:
                for lr in [0.001]:
                    args.num_layers = num_layers
                    args.dropout = dropout
                    args.lr = lr
                    args.l1 = l1
                    args.disable_heuristics = True
                    return_results, states = main(teacher_list, heuristic_list, args)
                    val_mean, val_std, test_mean, test_std = return_results[select_key]
                    if val_mean > best_valid_perf:
                        best_valid_perf = val_mean
                        best_results = return_results
                        best_param = (num_layers, l1, dropout, lr)
                        best_states = states 

    # Writing results to the file
    torch.save(best_states, f'saved_students/{dataset}_ensemble_MLPs.pth')
    grid_search_file.write(f'Dataset: {dataset}\n')
    grid_search_file.write('Best Parameters:\n')
    grid_search_file.write(
                f'  num_layers={best_param[0]}, l1={best_param[1]}, '
                f'dropout={best_param[2]}, lr={best_param[3]}\n'
            )

    grid_search_file.write(f'Best Validation Performance: {best_valid_perf:.4f}\n')
    for key in best_results.keys():
        test_mean, test_std = best_results[key][2], best_results[key][3]
        grid_search_file.write(f'Best Test {key}: {test_mean:.4f} ± {test_std:.4f}\n')

       
#torch.save(best_states, f'saved_students/{dataset}_{distill_teacher}_MLPs.pth')
