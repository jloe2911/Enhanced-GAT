import pandas as pd
import numpy as np
import operator
import gzip
import networkx as nx
import random
random.seed(10)

import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F

import torch_geometric
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, Linear, to_hetero
from torch_geometric.utils import negative_sampling, structured_negative_sampling

from sklearn.metrics import precision_score, recall_score, f1_score
    
class GCN(torch.nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = GCNConv(-1, hidden_dim)
        self.conv2 = GCNConv(-1, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x
    
class GraphSAGE(torch.nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_dim)
        self.conv2 = SAGEConv((-1, -1), output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x
    
class GAT(torch.nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = GATConv((-1, -1), hidden_dim, add_self_loops=False)
        self.lin1 = Linear(-1, hidden_dim)
        self.conv2 = GATConv((-1, -1), output_dim, add_self_loops=False)
        self.lin2 = Linear(-1, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index) + self.lin1(x)
        x = x.relu()
        x = self.conv2(x, edge_index) + self.lin2(x)
        return x
    
class Two_Hop_GAT(torch.nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = GATConv((-1, -1), hidden_dim, add_self_loops=False)
        self.lin1 = Linear(-1, hidden_dim)
        self.conv2 = GATConv((-1, -1), output_dim, add_self_loops=False)
        self.lin2 = Linear(-1, output_dim)

    def forward(self, x, edge_index, edge_index_2_hop):
        x = self.conv1(x, edge_index) + self.conv1(x, edge_index_2_hop) 
        x = self.lin1(x)
        x = x.relu()
        
        x = self.conv2(x, edge_index) + self.conv2(x, edge_index_2_hop) 
        x = self.lin2(x)
        return x
    
class GNN():
    def __init__(self):
        self.model = None
        self.epochs = 800
        self.node_embed_size = 200
        self.hidden_dim = 200
        self.output_dim = 200
        self.seed = 10
        torch.manual_seed(self.seed)
    
    def _train(self, device, GNN_variant, g_train, g_subclass_filter = None, g_assertion_filter = None):
        self.device = device
        
        adj = nx.to_scipy_sparse_array(g_train)
        pos_edge_index = torch_geometric.utils.from_scipy_sparse_matrix(adj)[0]
        neg_edge_index = negative_sampling(pos_edge_index)
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
        num_nodes = g_train.number_of_nodes()
        
        self.node_embeds = torch.rand(num_nodes, self.node_embed_size).to(self.device)
        
        if GNN_variant == 'GCN':
            self.model = GCN(self.hidden_dim, self.output_dim).to(self.device)
        elif GNN_variant == 'GraphSAGE':
            self.model = GraphSAGE(self.hidden_dim, self.output_dim).to(self.device)
        elif GNN_variant == 'GAT':
            self.model = GAT(self.hidden_dim, self.output_dim).to(self.device)
        elif GNN_variant == '2-Hop GAT':
            self.model = Two_Hop_GAT(self.hidden_dim, self.output_dim).to(self.device)
            adj_2hop = adj.dot(adj)
            edge_index_2hop = torch_geometric.utils.from_scipy_sparse_matrix(adj_2hop)[0].to(self.device)
        elif GNN_variant == 'GAT Reasoner':
            self.model = Two_Hop_GAT(self.hidden_dim, self.output_dim).to(self.device)
            adj_assertion = nx.to_scipy_sparse_array(g_assertion_filter)
            adj_subclass = nx.to_scipy_sparse_array(g_subclass_filter)
            adj_subclass_2hop = adj_subclass.dot(adj_subclass)
            
            edge_index_assertion = torch_geometric.utils.from_scipy_sparse_matrix(adj_assertion)[0].to(self.device)
            edge_index_subclass = torch_geometric.utils.from_scipy_sparse_matrix(adj_subclass)[0].to(self.device)
            edge_index_subclass_2hop = torch_geometric.utils.from_scipy_sparse_matrix(adj_subclass_2hop)[0].to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005, weight_decay=5e-4)    
        targets = torch.cat([torch.ones(pos_edge_index.shape[1]), torch.zeros(neg_edge_index.shape[1])])
        edge_index, targets = shuffle_predictions_targets(edge_index, targets, self.device)
        
        for i in range(self.epochs+1):
            self.model.train()
            optimizer.zero_grad()
            
            if GNN_variant == '2-Hop GAT':
                embeds = self.model(self.node_embeds, edge_index, edge_index_2hop).to(self.device)
            elif GNN_variant == 'GAT Reasoner':
                embeds_assertion = self.model(self.node_embeds, edge_index_assertion, edge_index_subclass_2hop).to(self.device)
                embeds_subclass = self.model(self.node_embeds, edge_index_subclass, edge_index_subclass_2hop).to(self.device)
                embeds_1hop = self.model(self.node_embeds, edge_index).to(self.device)
                embeds = torch.concat([embeds_assertion, embeds_subclass, embeds_1hop])
            else:
                embeds = self.model(self.node_embeds, edge_index).to(self.device)
                
            u = torch.index_select(embeds, 0, edge_index[0, :])
            v = torch.index_select(embeds, 0, edge_index[1, :])
            pred = torch.sum(u * v, dim=-1)
            pred = (pred - pred.min()) / (pred.max() - pred.min())
            
            loss = mse_loss(pred, targets)
            loss.backward()
            optimizer.step()
            
            if i % 400 == 0:
                print(f'Epoch: {i}, Loss: {loss:.4f}')

    def _eval(self, max_num, GNN_variant, g_test, g_subclass_filter = None, g_assertion_filter = None):
         with torch.no_grad():
                self.model.eval()
                
                adj = nx.to_scipy_sparse_array(g_test)
                pos_edge_index = torch_geometric.utils.from_scipy_sparse_matrix(adj)[0]
                neg_edge_index = negative_sampling(pos_edge_index)
                edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1).to(self.device)
                
                if GNN_variant == '2-Hop GAT':
                    adj_2hop = adj.dot(adj)
                    edge_index_2hop = torch_geometric.utils.from_scipy_sparse_matrix(adj_2hop)[0].to(self.device)
                
                if GNN_variant == 'GAT Reasoner':
                    adj_assertion = nx.to_scipy_sparse_array(g_assertion_filter)
                    adj_subclass = nx.to_scipy_sparse_array(g_subclass_filter)
                    adj_subclass_2hop = adj_subclass.dot(adj_subclass)

                    edge_index_assertion = torch_geometric.utils.from_scipy_sparse_matrix(adj_assertion)[0].to(self.device)
                    edge_index_subclass = torch_geometric.utils.from_scipy_sparse_matrix(adj_subclass)[0].to(self.device)
                    edge_index_subclass_2hop = torch_geometric.utils.from_scipy_sparse_matrix(adj_subclass_2hop)[0].to(self.device)
                
                if GNN_variant == '2-Hop GAT':
                    output = self.model(self.node_embeds, edge_index, edge_index_2hop).to(self.device)
                elif GNN_variant == 'GAT Reasoner':
                    embeds_assertion = self.model(self.node_embeds, edge_index_assertion, edge_index_subclass_2hop).to(self.device)
                    embeds_subclass = self.model(self.node_embeds, edge_index_subclass, edge_index_subclass_2hop).to(self.device)
                    embeds_1hop = self.model(self.node_embeds, edge_index).to(self.device)
                    output = torch.concat([embeds_assertion, embeds_subclass, embeds_1hop])
                else:
                    output = self.model(self.node_embeds, edge_index).to(self.device)
                
                ###Model as Binary Classification Problem###
                #u = torch.index_select(output, 0, edge_index[0, :])
                #v = torch.index_select(output, 0, edge_index[1, :])
                #pred = torch.sum(u * v, dim=-1)
                #pred = (pred - pred.min()) / (pred.max() - pred.min())
                
                #pred = pred.detach().numpy()
                #pred = np.where(pred >= 0.5, 1, 0)
                #targets = torch.cat([torch.ones(pos_edge_index.shape[1]), torch.zeros(neg_edge_index.shape[1])]).detach().numpy()
                
                #precision = precision_score(targets, pred)
                #recall = recall_score(targets, pred)
                #f1 = f1_score(targets, pred)
                
                #print(f'Precision: {precision:.3f}, Recall: {recall:.3f}, F1-Score: {f1:.3f}')
                #print()
                ######
                
                hits1, hits10 = eval_hits(tail_pred=1, 
                                          g_test=g_test, 
                                          pos_edge_index=pos_edge_index.to(self.device), 
                                          output=output, 
                                          max_num=max_num, 
                                          device=self.device)
                print(f'hits@1: {hits1:.3f}, hits@10: {hits10:.3f}')
                print('--------')
                
###HELPER FUNCIONS### 

def mse_loss(pred, target):
    return (pred - target.to(pred.dtype)).pow(2).mean()

def shuffle_predictions_targets(edge_index, targets, device):
    shuffle = list(zip(edge_index[0], edge_index[1], targets))
    random.shuffle(shuffle)
    u, v, targets = zip(*shuffle)
    return torch.tensor([list(u), list(v)]).to(device), torch.tensor(list(targets)).to(device)

def eval_hits(tail_pred, g_test, pos_edge_index, output, max_num, device):    
    top1 = 0
    top10 = 0
    n = pos_edge_index.size(1)

    for idx in range(n):

        if tail_pred == 1:
            x = torch.index_select(output, 0, pos_edge_index[0, idx])
        else:
            x = torch.index_select(output, 0, pos_edge_index[1, idx])
        
        candidates, candidates_embeds = sample_negative_edges_idx(idx=idx,
                                                                  tail_pred=tail_pred,
                                                                  g_test=g_test,
                                                                  pos_edge_index=pos_edge_index,
                                                                  output=output,
                                                                  max_num=max_num, 
                                                                  device=device)

        distances = torch.cdist(candidates_embeds, x, p=2)
        dist_dict = {cand: dist for cand, dist in zip(candidates, distances)} 

        sorted_dict = dict(sorted(dist_dict.items(), key=operator.itemgetter(1), reverse=True))
        sorted_keys = list(sorted_dict.keys())

        ranks_dict = {sorted_keys[i]: i for i in range(0, len(sorted_keys))}
        rank = ranks_dict[pos_edge_index[1, idx].item()]
        
        if rank <= 1:
            top1 += 1
        if rank <= 10:
            top10 += 1
    return top1/n, top10/n

def sample_negative_edges_idx(idx, tail_pred, g_test, pos_edge_index, output, max_num, device):
    num_neg_samples = 0
    candidates = []
    nodes = list(range(g_test.number_of_nodes()))
    random.shuffle(nodes)
    
    while num_neg_samples < max_num:    
        if tail_pred == 1:
            t = nodes[num_neg_samples]
            h = pos_edge_index[0, idx].item()
            if (h,t) not in g_test.edges():
                candidates.append(t)
        else: 
            t = pos_edge_index[1, idx].item()
            h = nodes[num_neg_samples]
            if (h,t) not in g_test.edges():
                candidates.append(h)
        num_neg_samples += 1
    candidates_embeds = torch.index_select(output, 0, torch.tensor(candidates).to(device))
    
    if tail_pred == 1:
        true_tail = pos_edge_index[1, idx]
        candidates.append(true_tail.item())
        candidates_embeds = torch.concat([candidates_embeds, torch.index_select(output, 0, true_tail)])
    else:
        true_head = pos_edge_index[0, idx]
        candidates.append(true_head.item())
        candidates_embeds = torch.concat([candidates_embeds, torch.index_select(output, 0, true_head)])
    return candidates, candidates_embeds.to(device)