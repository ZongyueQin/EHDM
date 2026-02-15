import random
import math
import torch
import numpy as np
import subprocess
from torch_geometric import datasets
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import (negative_sampling, add_self_loops, train_test_split_edges)
from torch_geometric.transforms import NormalizeFeatures, Compose, BaseTransform, ToDevice, RandomLinkSplit
from torch_geometric.utils import degree
from evalutors import evaluate_hits, evaluate_auc, evaluate_mrr
import scipy.sparse as ssp
from torch_sparse import SparseTensor

def read_data(data_name, neg_mode):
    data_name = data_name
    dir_path  = '/home/qinzongyue/HeaRT/'
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

def get_metric_score_citation2(evaluator_hit, evaluator_mrr, pos_train_pred, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred):
    
    k_list = [20, 50, 100, 200]
    result = {}

    result_mrr_train = evaluate_mrr( evaluator_mrr,  pos_train_pred, neg_val_pred)
    result_mrr_val = evaluate_mrr( evaluator_mrr, pos_val_pred, neg_val_pred )
    result_mrr_test = evaluate_mrr( evaluator_mrr, pos_test_pred, neg_test_pred )
    
   
#    result['MRR'] = (result_mrr_train['MRR'], result_mrr_val['MRR'], result_mrr_test['MRR'])
    result['MRR'] = (result_mrr_val['MRR'], result_mrr_test['MRR'])

    for K in k_list:
#        result[f'mrr_hit{K}'] = (result_mrr_train[f'mrr_hit{K}'], result_mrr_val[f'mrr_hit{K}'], result_mrr_test[f'mrr_hit{K}'])
        result[f'mrr_hit{K}'] = (result_mrr_val[f'mrr_hit{K}'], result_mrr_test[f'mrr_hit{K}'])

    return result


def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map

def get_dataset(root, name: str):
    if name.startswith('ogbl-'):
        dataset = PygLinkPropPredDataset(name=name, root=root, transform=transform)
        return dataset

    pyg_dataset_dict = {
        'cora': (datasets.Planetoid, 'Cora'),
        'citeseer': (datasets.Planetoid, 'Citeseer'),
        'pubmed': (datasets.Planetoid, 'Pubmed'),
        'coauthor-cs': (datasets.Coauthor, 'CS'),
        'coauthor-physics': (datasets.Coauthor, 'physics'),
        'amazon-computers': (datasets.Amazon, 'Computers'),
        'amazon-photos': (datasets.Amazon, 'Photo')
    }

    assert name in pyg_dataset_dict, "Dataset must be in {}".format(list(pyg_dataset_dict.keys()))

    dataset_class, name = pyg_dataset_dict[name]
    dataset = dataset_class(root, name=name)

    return dataset

def set_random_seeds(random_seed=0):
    r"""Sets the seed for generating random numbers."""
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

# From the OGB implementation of SEAL
def do_edge_split(dataset, fast_split=False, val_ratio=0.05, test_ratio=0.1):
    data = dataset[0]
    random.seed(234)
    torch.manual_seed(234)

    if not fast_split:
        data = train_test_split_edges(data, val_ratio, test_ratio)
        edge_index, _ = add_self_loops(data.train_pos_edge_index)
        data.train_neg_edge_index = negative_sampling(
            edge_index, num_nodes=data.num_nodes,
            num_neg_samples=data.train_pos_edge_index.size(1))
    else:
        num_nodes = data.num_nodes
        row, col = data.edge_index
        # Return upper triangular portion.
        mask = row < col
        row, col = row[mask], col[mask]
        n_v = int(math.floor(val_ratio * row.size(0)))
        n_t = int(math.floor(test_ratio * row.size(0)))
        # Positive edges.
        perm = torch.randperm(row.size(0))
        row, col = row[perm], col[perm]
        r, c = row[:n_v], col[:n_v]
        data.val_pos_edge_index = torch.stack([r, c], dim=0)
        r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]
        data.test_pos_edge_index = torch.stack([r, c], dim=0)
        r, c = row[n_v + n_t:], col[n_v + n_t:]
        data.train_pos_edge_index = torch.stack([r, c], dim=0)
        # Negative edges (cannot guarantee (i,j) and (j,i) won't both appear)
        neg_edge_index = negative_sampling(
            data.edge_index, num_nodes=num_nodes,
            num_neg_samples=row.size(0))
        data.val_neg_edge_index = neg_edge_index[:, :n_v]
        data.test_neg_edge_index = neg_edge_index[:, n_v:n_v + n_t]
        data.train_neg_edge_index = neg_edge_index[:, n_v + n_t:]

    split_edge = {'train': {}, 'valid': {}, 'test': {}}
    split_edge['train']['edge'] = data.train_pos_edge_index.t()
    split_edge['train']['edge_neg'] = data.train_neg_edge_index.t()
    split_edge['valid']['edge'] = data.val_pos_edge_index.t()
    split_edge['valid']['edge_neg'] = data.val_neg_edge_index.t()
    split_edge['test']['edge'] = data.test_pos_edge_index.t()
    split_edge['test']['edge_neg'] = data.test_neg_edge_index.t()
    return split_edge

def randomly_drop_nodes_citation2(dataset, keep_rate):
    data = dataset[0]
    from torch_sparse import SparseTensor    
    row, col, _ = data.adj_t.t().coo()
    edge_index = torch.stack([row, col], dim=0)
    import numpy as np; from torch_geometric.utils import k_hop_subgraph
    keep_nodes, this_edge_index = k_hop_subgraph(torch.arange(30), 4, edge_index)[:2]
    from torch_geometric.utils import subgraph
    from torch_geometric.utils import to_undirected
    edge_index = to_undirected(this_edge_index, data.x.size(0))

    split_edge = dataset.get_edge_split()
    train_edge_index = torch.stack([split_edge['train']['source_node'], split_edge['train']['target_node']], dim=0)
    train_edge_index = subgraph(keep_nodes, train_edge_index, num_nodes = data.x.size(0))[0]
    row, col = train_edge_index
    split_edge['train']['source_node'] = row
    split_edge['train']['target_node'] = col

    keep_nodes_set = keep_nodes
    indexes = torch.isin(split_edge['valid']['target_node'], keep_nodes_set).int() + torch.isin(split_edge['valid']['source_node'], keep_nodes_set).int()
    this_index = [i for i in range(indexes.size(0)) if int(indexes[i]) == 2]
    split_edge['valid']['target_node_neg'] = split_edge['valid']['target_node_neg'][this_index]
    valid_edge_index = torch.stack([split_edge['valid']['source_node'], split_edge['valid']['target_node']], dim=0)
    valid_edge_index = subgraph(keep_nodes, valid_edge_index, num_nodes = data.x.size(0))[0]
    row, col = valid_edge_index
    split_edge['valid']['source_node'] = row
    split_edge['valid']['target_node'] = col

    indexes = torch.isin(split_edge['test']['target_node'], keep_nodes_set).int() + torch.isin(split_edge['test']['source_node'], keep_nodes_set).int()
    this_index = [i for i in range(indexes.size(0)) if int(indexes[i]) == 2]
    split_edge['test']['target_node_neg'] = split_edge['test']['target_node_neg'][this_index]
    valid_edge_index = torch.stack([split_edge['test']['source_node'], split_edge['test']['target_node']], dim=0)
    valid_edge_index = subgraph(keep_nodes, valid_edge_index, num_nodes = data.x.size(0))[0]
    row, col = valid_edge_index
    split_edge['test']['source_node'] = row
    split_edge['test']['target_node'] = col

    torch.save((data, split_edge, keep_nodes.tolist()), "data/citation2_small.pkl")

    return (data, split_edge, keep_nodes)

def preprocess_igb(size):
    ### size: tiny, small, medium ####
    size = 'medium'
    #### download datasets (https://github.com/IllinoisGraphBenchmark/IGB-Datasets) #####
    # from igb import download
    # download.download_dataset(path='./', dataset_type='homogeneous', dataset_size=size) 

    from igb.dataloader import IGB260MDGLDataset
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='./', help='path containing the datasets')
    parser.add_argument('--dataset_size', type=str, default=size, choices=['tiny', 'small', 'medium', 'large', 'full'], help='size of the datasets')
    parser.add_argument('--num_classes', type=int, default=19, choices=[19, 2983], help='number of classes')
    parser.add_argument('--in_memory', type=int, default=0, choices=[0, 1], help='0:read only mmap_mode=r, 1:load into memory')
    parser.add_argument('--synthetic', type=int, default=0, choices=[0, 1], help='0:nlp-node embeddings, 1:random')
    args = parser.parse_args()

    dataset = IGB260MDGLDataset(args)
    import torch
    edge_index = torch.stack((dataset[0].edges()[0], dataset[0].edges()[1]), dim=0)
    from torch_geometric.data import Data
    data = Data(dataset[0].ndata['feat'], edge_index)

    from torch_geometric.transforms import RandomLinkSplit

    import random
    random.seed(234)
    import torch
    torch.manual_seed(234)

    splitter = RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True)
    training_data, val_data, test_data = splitter(data)

    split_edge = {'train': {}, 'valid': {}, 'test': {}}
    split_edge['train']['edge'] = training_data.edge_label_index[:,:int(training_data.edge_label.sum())].t()
    split_edge['train']['edge_neg'] = training_data.edge_label_index[:,int(training_data.edge_label.sum()):].t()
    split_edge['valid']['edge'] = val_data.edge_label_index[:,:int(val_data.edge_label.sum())].t()
    split_edge['valid']['edge_neg'] = val_data.edge_label_index[:,int(val_data.edge_label.sum()):].t()
    split_edge['test']['edge'] = test_data.edge_label_index[:,:int(test_data.edge_label.sum())].t()
    split_edge['test']['edge_neg'] = test_data.edge_label_index[:,int(test_data.edge_label.sum()):].t()

    data.adj_t = training_data.edge_index
    data.edge_index = training_data.edge_index

    torch.save((data, split_edge), "data/igb_" + size + ".pkl")

def get_node_wise_HITS(pos_pred, pos_edge, neg_pred, neg_edge, K):
    kth_score_in_negative_edges = torch.topk(neg_pred, K)[0][-1]
    hitsK = (pos_pred > kth_score_in_negative_edges).float()
    nodes = torch.cat([pos_edge[:,0], pos_edge[:,1]], dim=0)
    values = torch.cat([hitsK, hitsK], dim=0).to(nodes.device)
    unique_keys, inverse_indices, node_counts = torch.unique(nodes, return_inverse=True, return_counts=True)


    # Sum values for each unique key
    sums = torch.zeros_like(unique_keys, dtype=torch.float, device=pos_edge.device).scatter_add(0, inverse_indices, values.float())
    # Count occurrences of each key
    counts = torch.zeros_like(unique_keys, dtype=torch.float, device=pos_edge.device).scatter_add(0, inverse_indices, torch.ones_like(values, dtype=torch.float, device=pos_edge.device))

    # Calculate the average values
    average_values = sums / counts
    return average_values



def analyze(pred_results, data, K, count_thres=5):
    pos_pred, pos_edge, neg_pred, neg_edge= pred_results
    edge_index = data.adj_t
    node_degree = degree(torch.cat([edge_index[0], edge_index[1]], dim=0), data.x.size(0))  
    degree_thres = torch.quantile(node_degree, 0.8)
    print('degree thres', degree_thres)

    #TODO compute node-wise HIT@K
    if len(neg_pred) < K:
        print('K too small')
        return

    if isinstance(pos_pred, torch.Tensor):
        kth_score_in_negative_edges = torch.topk(neg_pred, K)[0][-1]
        hitsK = (pos_pred > kth_score_in_negative_edges).float()
#        src, dst = pos_edge
        nodes = torch.cat([pos_edge[:,0], pos_edge[:,1]], dim=0)
        values = torch.cat([hitsK, hitsK], dim=0).to(nodes.device)
        unique_keys, inverse_indices, node_counts = torch.unique(nodes, return_inverse=True, return_counts=True)


        # Sum values for each unique key
        sums = torch.zeros_like(unique_keys, dtype=torch.float, device=pos_edge.device).scatter_add(0, inverse_indices, values.float())

        # Count occurrences of each key
        counts = torch.zeros_like(unique_keys, dtype=torch.float, device=pos_edge.device).scatter_add(0, inverse_indices, torch.ones_like(values, dtype=torch.float, device=pos_edge.device))

        # Calculate the average values
        average_values = sums / counts

        # Display results
        print("Unique keys:", unique_keys)
        print("Average values:", average_values)
        print("Unique keys frequent:", unique_keys[node_counts>=count_thres])
        print("Average values frequent:", average_values[node_counts>=count_thres])

        test_degree = node_degree[unique_keys]
        mask = test_degree < degree_thres
        print('Avg values low degree:', average_values[mask].mean().item())
        print('Avg values high degree:', average_values[~mask].mean().item())

        print(hitsK.mean().item())

        # type_info is numpy
        
    else:
        kth_score_in_negative_edges = np.sort(neg_pred)[-self.K]
        hitsK = pos_pred > kth_score_in_negative_edges

