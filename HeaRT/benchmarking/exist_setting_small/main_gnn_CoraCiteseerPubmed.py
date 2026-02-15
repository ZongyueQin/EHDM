import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import sys
sys.path.append("..") 

from os.path import exists

import torch
import argparse
import scipy.sparse as ssp
from gnn_model import *
from utils import *
from scoring import mlp_score
from sklearn.metrics import *

from torch.utils.data import DataLoader
from torch_sparse import SparseTensor


from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from evalutors import evaluate_hits, evaluate_mrr, evaluate_auc
from copy import deepcopy
from utils import get_dataset, do_edge_split 


dir_path  = get_root_dir()
log_print = get_logger('testrun', 'log', get_config_dir())


def read_data(data_name, neg_mode):
    data_name = data_name
    if 'coauthor' in data_name or 'amazon' in data_name:
        dataset = get_dataset('../data/', data_name)
        data = dataset[0]

        if exists("../data/" + data_name + ".pkl"):
            split_edge = torch.load("../data/" + data_name + ".pkl")
        else:
            split_edge = do_edge_split(dataset)
            torch.save(split_edge, "../data/" + data_name + ".pkl")
            
        edge_index = split_edge['train']['edge'].t()
        data.adj_t = edge_index

        edge_weight = torch.ones(edge_index.size(1))
        num_nodes = data.x.size(0)
        A = ssp.csr_matrix((edge_weight.view(-1), (edge_index[0], edge_index[1])), shape=(num_nodes, num_nodes)) 

        adj = SparseTensor.from_edge_index(edge_index, edge_weight, [num_nodes, num_nodes])
        train_pos_tensor = split_edge['train']['edge']
        valid_pos = split_edge['valid']['edge'] 
        valid_neg = split_edge['valid']['edge_neg'] 

        test_pos =  split_edge['test']['edge'] 
        test_neg =  split_edge['test']['edge_neg'] 

        idx = torch.randperm(train_pos_tensor.size(0))
        idx = idx[:valid_pos.size(0)]
        train_val = train_pos_tensor[idx]

        ret_data = {}
        ret_data['A'] = A
        ret_data['adj'] = adj
        ret_data['train_pos'] = split_edge['train']['edge'] 
        ret_data['train_val'] = train_val

        ret_data['valid_pos'] = valid_pos
        ret_data['valid_neg'] = valid_neg
        ret_data['test_pos'] = test_pos
        ret_data['test_neg'] = test_neg

        ret_data['x'] = data.x
        return ret_data

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


def get_metric_score(evaluator_hit, evaluator_mrr, pos_train_pred, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred):

    
    # result_hit = evaluate_hits(evaluator_hit, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred)
    result = {}
    k_list = [1, 3, 10, 20, 50, 100]
    result_hit_train = evaluate_hits(evaluator_hit, pos_train_pred, neg_val_pred, k_list)
    result_hit_val = evaluate_hits(evaluator_hit, pos_val_pred, neg_val_pred, k_list)
    result_hit_test = evaluate_hits(evaluator_hit, pos_test_pred, neg_test_pred, k_list)

    # result_hit = {}
    for K in [1, 3, 10, 20, 50, 100]:
        result[f'Hits@{K}'] = (result_hit_train[f'Hits@{K}'], result_hit_val[f'Hits@{K}'], result_hit_test[f'Hits@{K}'])


    result_mrr_train = evaluate_mrr(evaluator_mrr, pos_train_pred, neg_val_pred.repeat(pos_train_pred.size(0), 1))
    result_mrr_val = evaluate_mrr(evaluator_mrr, pos_val_pred, neg_val_pred.repeat(pos_val_pred.size(0), 1) )
    result_mrr_test = evaluate_mrr(evaluator_mrr, pos_test_pred, neg_test_pred.repeat(pos_test_pred.size(0), 1) )
    
    # result_mrr = {}
    result['MRR'] = (result_mrr_train['MRR'], result_mrr_val['MRR'], result_mrr_test['MRR'])
    # for K in [1,3,10, 100]:
    #     result[f'mrr_hit{K}'] = (result_mrr_train[f'mrr_hit{K}'], result_mrr_val[f'mrr_hit{K}'], result_mrr_test[f'mrr_hit{K}'])

   
    train_pred = torch.cat([pos_train_pred, neg_val_pred])
    train_true = torch.cat([torch.ones(pos_train_pred.size(0), dtype=int), 
                            torch.zeros(neg_val_pred.size(0), dtype=int)])

    val_pred = torch.cat([pos_val_pred, neg_val_pred])
    val_true = torch.cat([torch.ones(pos_val_pred.size(0), dtype=int), 
                            torch.zeros(neg_val_pred.size(0), dtype=int)])
    test_pred = torch.cat([pos_test_pred, neg_test_pred])
    test_true = torch.cat([torch.ones(pos_test_pred.size(0), dtype=int), 
                            torch.zeros(neg_test_pred.size(0), dtype=int)])

    result_auc_train = evaluate_auc(train_pred, train_true)
    result_auc_val = evaluate_auc(val_pred, val_true)
    result_auc_test = evaluate_auc(test_pred, test_true)

    # result_auc = {}
    result['AUC'] = (result_auc_train['AUC'], result_auc_val['AUC'], result_auc_test['AUC'])
    result['AP'] = (result_auc_train['AP'], result_auc_val['AP'], result_auc_test['AP'])

    
    return result

        

def train(model, score_func, train_pos, x, optimizer, batch_size, remove_target=True):
    model.train()
    score_func.train()

    # train_pos = train_pos.transpose(1, 0)
    total_loss = total_examples = 0
    num_nodes = x.size(0)
    if remove_target == False:
        train_edge_mask = train_pos.transpose(1,0)
        # edge_weight_mask = torch.cat((edge_weight_mask, edge_weight_mask), dim=0).to(torch.float)
        edge_weight_mask = torch.ones(train_edge_mask.size(1)).to(torch.float).to(train_pos.device)
        
        adj = SparseTensor.from_edge_index(train_edge_mask, edge_weight_mask, [num_nodes, num_nodes]).to(train_pos.device)


    for perm in DataLoader(range(train_pos.size(0)), batch_size,
                           shuffle=True):
        optimizer.zero_grad()



        ######################### remove loss edges from the aggregation
        if remove_target == True:
            mask = torch.ones(train_pos.size(0), dtype=torch.bool).to(train_pos.device)
            mask[perm] = 0
    
            train_edge_mask = train_pos[mask].transpose(1,0)

            # train_edge_mask = to_undirected(train_edge_mask)
            train_edge_mask = torch.cat((train_edge_mask, train_edge_mask[[1,0]]),dim=1)
            # edge_weight_mask = torch.cat((edge_weight_mask, edge_weight_mask), dim=0).to(torch.float)
            edge_weight_mask = torch.ones(train_edge_mask.size(1)).to(torch.float).to(train_pos.device)
        
            adj = SparseTensor.from_edge_index(train_edge_mask, edge_weight_mask, [num_nodes, num_nodes]).to(train_pos.device)
            
        ###################
        # print(adj)

        h = model(x, adj)

        edge = train_pos[perm].t()

        pos_out = score_func(h[edge[0]], h[edge[1]])
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        # Just do some trivial random sampling.
        edge = torch.randint(0, num_nodes, edge.size(), dtype=torch.long,
                             device=h.device)
        neg_out = score_func(h[edge[0]], h[edge[1]])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(score_func.parameters(), 1.0)

        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples



@torch.no_grad()
def test_edge(score_func, input_data, h, batch_size):

    # input_data  = input_data.transpose(1, 0)
    # with torch.no_grad():
    preds = []
    for perm  in DataLoader(range(input_data.size(0)), batch_size):
        edge = input_data[perm].t()
    
        preds += [score_func(h[edge[0]], h[edge[1]]).cpu()]
        
    pred_all = torch.cat(preds, dim=0)

    return pred_all

@torch.no_grad()
def get_score(model, score_func, pos_edge, neg_edge, data, x, evaluator_hit, evaluator_mrr, batch_size):
    model.eval()
    score_func.eval()

    # adj_t = adj_t.transpose(1,0)
    
    
    h = model(x, data['adj'].to(x.device))
    # print(h[0][:10])
    x = h

    #reshape pos edge neg edge
    node = pos_edge[:,0]

    pos_nb = pos_edge[:,1:]
    neg_nb = neg_edge

    src_nodes = node.repeat_interleave(pos_nb.size(1))
    pos_edge = torch.stack((src_nodes, pos_nb.flatten()), dim=1)
    src_nodes = node.repeat_interleave(neg_nb.size(1))
    neg_edge = torch.stack((src_nodes, neg_nb.flatten()), dim=1)

#    print(pos_edge.size(), neg_edge.size(), data['train_pos'].size())
    """
    def edge_tensor_to_set(edge_index):
        edges = set((edge_index[i,0].item(), edge_index[i,1].item()) for i in range(edge_index.shape[0]))
        return edges
    set1 = edge_tensor_to_set(pos_edge)
    set2 = edge_tensor_to_set(neg_edge)
    set3 = edge_tensor_to_set(data['train_pos'])
    pos = set1.issubset(set3)
    neg = set2.isdisjoint(set3)
    print(pos, neg)
    xxx = input("pause")
    """

    pos_pred = test_edge(score_func, pos_edge, h, batch_size).flatten()

    neg_pred = test_edge(score_func, neg_edge, h, batch_size).flatten()

    result = get_metric_score(evaluator_hit, evaluator_mrr, pos_pred[:1000], pos_pred[:1000], neg_pred[:1000], pos_pred[:1000], neg_pred[:1000])
    #TODO reshape pos pred neg pred
    pos_pred = pos_pred.reshape(node.size(0), -1)
    neg_pred = neg_pred.reshape(node.size(0), -1)

    return result, pos_pred, neg_pred



@torch.no_grad()
def test(model, score_func, data, x, evaluator_hit, evaluator_mrr, batch_size):
    model.eval()
    score_func.eval()

    # adj_t = adj_t.transpose(1,0)
    
    
    h = model(x, data['adj'].to(x.device))
    # print(h[0][:10])
    x = h

    pos_train_pred = test_edge(score_func, data['train_val'], h, batch_size)

    neg_valid_pred = test_edge(score_func, data['valid_neg'], h, batch_size)

    pos_valid_pred = test_edge(score_func, data['valid_pos'], h, batch_size)

    pos_test_pred = test_edge(score_func, data['test_pos'], h, batch_size)

    neg_test_pred = test_edge(score_func, data['test_neg'], h, batch_size)

    pos_train_pred = torch.flatten(pos_train_pred)
    neg_valid_pred, pos_valid_pred = torch.flatten(neg_valid_pred),  torch.flatten(pos_valid_pred)
    pos_test_pred, neg_test_pred = torch.flatten(pos_test_pred), torch.flatten(neg_test_pred)


    print('train valid_pos valid_neg test_pos test_neg', pos_train_pred.size(), pos_valid_pred.size(), neg_valid_pred.size(), pos_test_pred.size(), neg_test_pred.size())


    
    result = get_metric_score(evaluator_hit, evaluator_mrr, pos_train_pred, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred)
    

    score_emb = [pos_valid_pred.cpu(),neg_valid_pred.cpu(), pos_test_pred.cpu(), neg_test_pred.cpu(), x.cpu()]

    return result, score_emb

@torch.no_grad()
def test_production(model, score_func, val_data, inference_data, test_edge_bundle, negative_samples, batch_size, evaluator, device):
    model.eval()
    score_func.eval()

    val_data = val_data.to(device)
    inference_data = inference_data.to(device)
    h = model(val_data.x, val_data.edge_index)

    saved_h = h

    negative_edges = negative_samples.t().to(h.device)


    val_edges = val_data.edge_label_index.t()
    val_pos_edges = val_edges[val_data.edge_label.bool()]
    val_neg_edges = val_edges[(torch.tensor(1)-val_data.edge_label).bool()] 
    old_old_edges = test_edge_bundle[0].t().to(h.device)
    old_new_edges = test_edge_bundle[1].t().to(h.device)
    new_new_edges = test_edge_bundle[2].t().to(h.device)
    test_edges = test_edge_bundle[3].t().to(h.device)

    pos_valid_pred = test_edge(score_func, val_pos_edges, h, batch_size).squeeze()
    neg_valid_pred = test_edge(score_func, val_neg_edges, h, batch_size).squeeze()

    h = model(inference_data.x, inference_data.edge_index)
    pos_test_pred = test_edge(score_func, test_edges, h, batch_size).squeeze()
    neg_test_pred = test_edge(score_func, negative_edges, h, batch_size).squeeze()

    old_old_pos_test_pred = test_edge(score_func, old_old_edges, h, batch_size).squeeze()
    old_new_pos_test_pred = test_edge(score_func, old_new_edges, h, batch_size).squeeze()
    new_new_pos_test_pred = test_edge(score_func, new_new_edges, h, batch_size).squeeze()

    results = {}
    for K in [10, 20, 30, 50]:
        evaluator.K = K
        val_hits = evaluator.eval({
            'y_pred_pos': pos_valid_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']
        old_old_test_hits = evaluator.eval({
            'y_pred_pos': old_old_pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']
        old_new_test_hits = evaluator.eval({
            'y_pred_pos': old_new_pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']
        new_new_test_hits = evaluator.eval({
            'y_pred_pos': new_new_pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (val_hits, test_hits, old_old_test_hits, old_new_test_hits, new_new_test_hits)

    valid_result = torch.cat((torch.ones(pos_valid_pred.size()), torch.zeros(neg_valid_pred.size())), dim=0)
    valid_pred = torch.cat((pos_valid_pred, neg_valid_pred), dim=0)

    test_result = torch.cat((torch.ones(pos_test_pred.size()), torch.zeros(neg_test_pred.size())), dim=0)
    test_pred = torch.cat((pos_test_pred, neg_test_pred), dim=0)

    old_old_result = torch.cat((torch.ones(old_old_pos_test_pred.size()), torch.zeros(neg_test_pred.size())), dim=0)
    old_old_pred = torch.cat((old_old_pos_test_pred, neg_test_pred), dim=0)

    old_new_result = torch.cat((torch.ones(old_new_pos_test_pred.size()), torch.zeros(neg_test_pred.size())), dim=0)
    old_new_pred = torch.cat((old_new_pos_test_pred, neg_test_pred), dim=0)

    new_new_result = torch.cat((torch.ones(new_new_pos_test_pred.size()), torch.zeros(neg_test_pred.size())), dim=0)
    new_new_pred = torch.cat((new_new_pos_test_pred, neg_test_pred), dim=0)

    results['AUC'] = (roc_auc_score(valid_result.cpu().numpy(),valid_pred.cpu().numpy()), roc_auc_score(test_result.cpu().numpy(),test_pred.cpu().numpy()),roc_auc_score(old_old_result.cpu().numpy(),old_old_pred.cpu().numpy()),roc_auc_score(old_new_result.cpu().numpy(),old_new_pred.cpu().numpy()),roc_auc_score(new_new_result.cpu().numpy(),new_new_pred.cpu().numpy()))

    return results, saved_h


def main():
    parser = argparse.ArgumentParser(description='homo')
    parser.add_argument('--edge_path_file', type=str, default='None')
    parser.add_argument('--production', action='store_true', default=False)

    parser.add_argument('--data_name', type=str, default='cora')
    parser.add_argument('--neg_mode', type=str, default='equal')
    parser.add_argument('--gnn_model', type=str, default='GCN')
    parser.add_argument('--score_model', type=str, default='mlp_score')

    ##gnn setting
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--num_layers_predictor', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.0)


    ### train setting
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=9999)
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--kill_cnt',           dest='kill_cnt',      default=10,    type=int,       help='early stopping')
    parser.add_argument('--output_dir', type=str, default='output_test')
    parser.add_argument('--l2',		type=float,             default=0.0,			help='L2 Regularization for Optimizer')
    parser.add_argument('--seed', type=int, default=999)
    
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--use_saved_model', action='store_true', default=False)
    parser.add_argument('--metric', type=str, default='MRR')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    
    ####### gin
    parser.add_argument('--gin_mlp_layer', type=int, default=2)

    ######gat
    parser.add_argument('--gat_head', type=int, default=1)

    ######mf
    parser.add_argument('--cat_node_feat_mf', default=False, action='store_true')

    ###### n2v
    parser.add_argument('--cat_n2v_feat', default=False, action='store_true')
    
    args = parser.parse_args()
   

    print('cat_node_feat_mf: ', args.cat_node_feat_mf)
    print('cat_n2v_feat: ', args.cat_n2v_feat)
    print(args)

    init_seed(args.seed)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    # dataset = Planetoid('.', 'cora')

    if args.production == False:
        data = read_data(args.data_name, args.neg_mode)
        node_num = data['x'].size(0)
        x = data['x']
        train_pos = data['train_pos'].to(device)
    else:
#        data = read_data_production(args.data_name)
        training_data, val_data, inference_data, _, test_edge_bundle, negative_samples = torch.load("/home/qinzongyue/linkless-link-prediction/data_production/" + args.data_name + "_production.pkl")
        x = training_data.x
        node_num = x.size(0)
        train_pos = training_data.edge_index.T
        train_pos = train_pos.to(device)
        edge_index = training_data.edge_index
        edge_weight = torch.ones(training_data.edge_index.size(1))
        num_nodes = node_num 
        adj = SparseTensor.from_edge_index(edge_index, edge_weight, [num_nodes, num_nodes])

   

    if args.cat_n2v_feat:
        print('cat n2v embedding!!')
        n2v_emb = torch.load(os.path.join(get_root_dir(), 'dataset', args.data_name+'-n2v-embedding.pt'))
        x = torch.cat((x, n2v_emb), dim=-1)

    x = x.to(device)
#    torch.save(train_pos,  'cora_train_pos.pth')
#    return

    input_channel = x.size(1)
    model = eval(args.gnn_model)(input_channel, args.hidden_channels,
                    args.hidden_channels, args.num_layers, args.dropout, args.gin_mlp_layer, args.gat_head, node_num, args.cat_node_feat_mf).to(device)
    
    score_func = eval(args.score_model)(args.hidden_channels, args.hidden_channels,
                    1, args.num_layers_predictor, args.dropout).to(device)
   
    
    eval_metric = args.metric
    if args.production:
        eval_metric = 'Hits@20'
    evaluator_hit = Evaluator(name='ogbl-collab')
    evaluator_mrr = Evaluator(name='ogbl-citation2')

    if args.production == False:
        loggers = {
          'Hits@1': Logger(args.runs),
          'Hits@3': Logger(args.runs),
          'Hits@10': Logger(args.runs),
          'Hits@20': Logger(args.runs),
          'Hits@50': Logger(args.runs),
          'Hits@100': Logger(args.runs),
          'MRR': Logger(args.runs),
          'AUC':Logger(args.runs),
          'AP':Logger(args.runs)
        }

    else:
        loggers = {
            'Hits@10': ProductionLogger(args.runs, args),
            'Hits@20': ProductionLogger(args.runs, args),
            'Hits@30': ProductionLogger(args.runs, args),
            'Hits@50': ProductionLogger(args.runs, args),
            'AUC': ProductionLogger(args.runs, args),
        }


    if args.edge_path_file != 'None':
        print(f'loading from {args.edge_path_file}')
        edges = torch.load(args.edge_path_file)
        pos_edge, neg_edge = edges[0], edges[1]
        #pos_edge, neg_edge = data['valid_pos'], data['valid_neg']
#        print(pos_edge, neg_edge)
    if 'coauthor' in args.data_name or 'amazon' in args.data_name:
        remove_target = False
    else:
        remove_target = True

    best_states = []
    best_ratios = []
    for run in range(args.runs):

        print('#################################          ', run, '          #################################')
        best_state = {}
        
        if args.runs == 1:
            seed = args.seed
        else:
            seed = run
        print('seed: ', seed)

        init_seed(seed)
        
        save_path = args.output_dir+'/lr'+str(args.lr) + '_drop' + str(args.dropout) + '_l2'+ str(args.l2) + '_numlayer' + str(args.num_layers)+ '_numPredlay' + str(args.num_layers_predictor) + '_numGinMlplayer' + str(args.gin_mlp_layer)+'_dim'+str(args.hidden_channels) + '_'+ 'best_run_'+str(seed)

        model.reset_parameters()
        score_func.reset_parameters()

        optimizer = torch.optim.Adam(
                list(model.parameters()) + list(score_func.parameters()),lr=args.lr, weight_decay=args.l2)

        best_valid = 0
        kill_cnt = 0
#        print(data['valid_pos'])
#        print(data['valid_neg'])
#        print(data['valid_pos'].size())
#        print(data['valid_neg'].size())
#        xx = input("pause")

        for epoch in range(1, 1 + args.epochs):
            loss = train(model, score_func, train_pos, x, optimizer, args.batch_size, remove_target)
            # print(model.convs[0].att_src[0][0][:10])
            
            if epoch % args.eval_steps == 0:
                if args.production == False:
                    results_rank, score_emb = test(model, score_func, data, x, evaluator_hit, evaluator_mrr, args.batch_size)
                else:
                    results_rank, score_emb = test_production(model, score_func, val_data, inference_data, test_edge_bundle, negative_samples, args.batch_size, evaluator_hit, device)


                for key, result in results_rank.items():
                    loggers[key].add_result(run, result)

                if epoch % args.log_steps == 0:
                    for key, result in results_rank.items():
                        
                        print(key)
                        
                        if args.production == False:
                            train_hits, valid_hits, test_hits = result
                            log_print.info(
                                f'Run: {run + 1:02d}, '
                                  f'Epoch: {epoch:02d}, '
                                  f'Loss: {loss:.4f}, '
                                  f'Train: {100 * train_hits:.2f}%, '
                                  f'Valid: {100 * valid_hits:.2f}%, '
                                  f'Test: {100 * test_hits:.2f}%')
                        else:
                            valid_hits, test_hits, old_old, old_new, new_new = result
                            print(f'Run: {run + 1:02d}, '
                              f'Epoch: {epoch:02d}, '
                              f'Loss: {loss:.4f}, '
                              f'valid: {100 * valid_hits:.2f}%, '
                              f'test: {100 * test_hits:.2f}%, '
                              f'old_old: {100 * old_old:.2f}%, '
                              f'old_new: {100 * old_new:.2f}%, '
                              f'new_new: {100 * new_new:.2f}%')

                    print('---')

                best_valid_current = torch.tensor(loggers[eval_metric].results[run])[:, 1].max()

                if best_valid_current > best_valid:
                    pos_test_pred = score_emb[2]
                    neg_test_pred = score_emb[3]
                    true_pos_pred = (pos_test_pred>0.05).float().sum()
                    false_pos_pred = (neg_test_pred>0.05).float().sum()
                    best_ratio = true_pos_pred/(true_pos_pred+false_pos_pred)


                    best_valid = best_valid_current
                    kill_cnt = 0

                    if args.save:

                        save_emb(score_emb, save_path)
                    if args.edge_path_file != 'None' and run == 0:
                        if args.production == False:
                            results_distill, pos_pred, neg_pred = get_score(model, score_func, pos_edge, neg_edge, data, x, evaluator_hit, evaluator_mrr, args.batch_size)
                        else:
                            results_distill, pos_pred, neg_pred = get_score(model, score_func, pos_edge, neg_edge, {'adj':adj}, x, evaluator_hit, evaluator_mrr, args.batch_size)


                        print('distill edge result')
                        for key, result in results_distill.items():
                        
                            print(key)
                        
                            train_hits, valid_hits, test_hits = result


                            log_print.info(
                              f'Run: {run + 1:02d}, '
                                f'Epoch: {epoch:02d}, '
                                f'Loss: {loss:.4f}, '
                                f'Train: {100 * train_hits:.2f}%, '
                                f'Valid: {100 * valid_hits:.2f}%, '
                                f'Test: {100 * test_hits:.2f}%')
                    print('****************')
                    best_state['model'] = deepcopy(model.state_dict())
                    best_state['predictor'] = deepcopy(score_func.state_dict())


                
                else:
                    kill_cnt += 1
                    
                    if kill_cnt > args.kill_cnt: 
                        print("Early Stopping!!")
                        break
        
        for key in loggers.keys():
            print(key)
            loggers[key].print_statistics(run)
        best_states.append(best_state)
        best_ratios.append(best_ratio)

    if args.production == False:
        if args.edge_path_file != 'None':
            torch.save((pos_pred, neg_pred), f'{args.data_name}_{args.gnn_model}_pred.pth')
        torch.save(best_states, f'/home/qinzongyue/linkless-link-prediction/src/saved_students/{args.data_name}_{args.gnn_model}.pth')
    else:
        if args.edge_path_file != 'None':
            torch.save((pos_pred, neg_pred), f'{args.data_name}_{args.gnn_model}_pred_production.pth')
        torch.save(best_states, f'/home/qinzongyue/linkless-link-prediction/src/saved_students/{args.data_name}_{args.gnn_model}_production.pth')

    print(np.mean(best_ratios))
   
    if args.production == False:
        result_all_run = {}
        for key in loggers.keys():
            print(key)
        
            best_metric,  best_valid_mean, mean_list, var_list = loggers[key].print_statistics()
   
            if key == eval_metric:
                best_metric_valid_str = best_metric
                best_valid_mean_metric = best_valid_mean


            
            if key == 'AUC':
                best_auc_valid_str = best_metric
                best_auc_metric = best_valid_mean

            result_all_run[key] = [mean_list, var_list]
        print(best_metric_valid_str +' ' +best_auc_valid_str)

        return best_valid_mean_metric, best_auc_metric, result_all_run
    else:
        for key in loggers.keys():
            print(key)
        
            loggers[key].print_statistics()
        return

        
    




if __name__ == "__main__":
    main()

   
