import os.path as osp
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
import torch_geometric.nn as pygnn
from torch_geometric.nn import global_add_pool, MessagePassing, HeteroConv, to_hetero, GINConv, GCNConv
from torch_geometric.utils import softmax, add_self_loops
from torch_scatter import scatter_add
import torch_geometric.transforms as T


import numpy as np
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
import sys
import torch_scatter


class Explainer(torch.nn.Module):
    def __init__(self, num_features, dim, num_gc_layers):
        super(Explainer, self).__init__()

        self.num_gc_layers = num_gc_layers

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        for i in range(num_gc_layers):

            if i and i != num_gc_layers - 1:
                nn = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
                bn = torch.nn.BatchNorm1d(dim)
                gcns = GCNConv(dim, dim)
            elif i == num_gc_layers - 1:
                nn = Sequential(Linear(dim, dim), ReLU(), Linear(dim, 1))
                bn = torch.nn.BatchNorm1d(1)
                gcns = GCNConv(dim, 1)
            else:
                nn = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
                bn = torch.nn.BatchNorm1d(dim)
                gcns = GCNConv(num_features, dim)

            conv = GINConv(nn)
            #conv = gcns

            self.convs.append(conv)
            self.bns.append(bn)

    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        else:
            raise ValueError("unmatched number of arguments.")

        if x is None:
            x = torch.ones((batch.shape[0], 1)).to(edge_index.device)

        xs = []

        for i in range(self.num_gc_layers):

            if i != self.num_gc_layers - 1:
                x = F.relu(self.bns[i](self.convs[i](x, edge_index)))
            else:
                x = self.bns[i](self.convs[i](x, edge_index))
            xs.append(x)

        node_prob = xs[-1]
        #node_prob = softmax(node_prob - node_prob.max(), batch)
        node_prob = softmax(node_prob/5.0, batch)
        # _, num_nodes = torch.unique(batch, return_counts=True)
        # num_nodes = torch.unsqueeze(num_nodes[batch], 1)
        # node_prob = node_prob * num_nodes

        return node_prob


class linears(MessagePassing):
    def __init__(self, in_dim, out_dim, aggr = 'add', coef = 0.1, header = 1):
        super(linears, self).__init__()
        self.out_dim = out_dim * header
        self.mlp = Linear(in_dim, self.out_dim)
        self.aggr = aggr
        self.coef = coef

    def forward(self, x_dict, edge_index_dict):

        return self.propagate(edge_index_dict, x=x_dict)
    

    def update(self, aggr_out):
        return self.mlp(aggr_out) * self.coef


class HGNN(torch.nn.Module):
    def __init__(self, num_layer, emb_dim, seq_len, attr_len, JK = "last", drop_ratio=0, gnn_type = "gin"):
        super(HGNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.attr_len = attr_len
        self.JK = JK
        self.node_type = ['0','1']
        self.edge_type = [('1', '0', '1'), ('1', '1', '0'), ('0', '2', '1'), ('0', '3', '0')]
        self.meta_data = (self.node_type, self.edge_type)
        self.gnn_type = gnn_type
        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ###List of MLPs
        self.gnns = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        
        for layer in range(num_layer):
            conv_dict = {}
            if layer:
                nn = Sequential(Linear(emb_dim, emb_dim), ReLU(), Linear(emb_dim, emb_dim))
                hlinear = linears(emb_dim, emb_dim)
                gcns = GCNConv(emb_dim, emb_dim)
            else:
                nn = Sequential(Linear(seq_len, emb_dim), ReLU(), Linear(emb_dim, emb_dim))
                hlinear = linears(seq_len, emb_dim)
                gcns = GCNConv(seq_len, emb_dim)
            for i in self.edge_type:
                if i == ('1','0','1'):
                    if gnn_type == 'gin':
                        conv_dict[i] = GINConv(nn)
                    elif gnn_type == 'gcn':
                        conv_dict[i] = gcns
                else:
                    conv_dict[i] = hlinear
                    #conv_dict[i] = GINConv(nn)
            conv = HeteroConv(conv_dict, aggr='mean')
            bn = torch.nn.BatchNorm1d(emb_dim)

            self.gnns.append(conv)
            self.batch_norms.append(bn)


    def forward(self, batch):
        x_dict, edge_index_dict, batch_dict = batch.x_dict, batch.edge_index_dict, batch.batch_dict 

        h_dict_list = [x_dict]

        for layer in range(self.num_layer):
            h_dict = self.gnns[layer](h_dict_list[layer], edge_index_dict)
            h_dict = {key:F.relu(x) for key, x in h_dict.items()}
            h_dict = {key:self.batch_norms[layer](x) for key, x in h_dict.items()}
            #h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            # if layer == self.num_layer - 1:
            #     #remove relu for the last layer
            #     h_dict = {key:F.dropout(x, self.drop_ratio, training = self.training) for key, x in h_dict.items()}
            # else:
            #     h_dict = {key:F.dropout(F.relu(x), self.drop_ratio, training = self.training) for key, x in h_dict.items()}
            h_dict_list.append(h_dict)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_dict_list[-1]
        else:
            node_representation = {}
            tensor0 = []
            tensor1 = []
            for i in h_dict_list[1:]:
                tensor0.append(i['0'])
                tensor1.append(i['1'])
            node_representation['0'] = torch.cat(tensor0, 1)
            node_representation['1'] = torch.cat(tensor1, 1)

        batch._node_store_dict['0']['x'] = node_representation['0']
        batch._node_store_dict['1']['x'] = node_representation['1']
        return batch.to_homogeneous().x
    





class HEncoder(torch.nn.Module):
    def __init__(self, num_features, dim, num_gc_layers, pooling):
        super(HEncoder, self).__init__()

        self.num_gc_layers = num_gc_layers
        self.pooling = pooling

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.dim = dim

        for i in range(num_gc_layers):
            if i:
                nn = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
            else:
                nn = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
            conv = GINConv(nn)
            bn = torch.nn.BatchNorm1d(dim)

            self.convs.append(conv)
            self.bns.append(bn)

    def forward(self, x, edge_index, batch, node_imp):

        # mapping node_imp to [0.9,1.1]
        if node_imp is not None:
            out, _ = torch_scatter.scatter_max(torch.reshape(node_imp, (1, -1)), batch)
            out = out.reshape(-1, 1)
            out = out[batch]
            node_imp /= (out*10)
            node_imp += 0.9
            node_imp = node_imp.expand(-1, self.dim)

        if x is None:
            x = torch.ones((batch.shape[0], 1)).to(device)

        xs = []
        for i in range(self.num_gc_layers):

            x = F.relu(self.convs[i](x, edge_index))
            x = self.bns[i](x)

            if node_imp is not None:
                x_imp = x * node_imp
            else:
                x_imp = x

            xs.append(x_imp)

        if self.pooling == 'last':
            x = global_add_pool(xs[-1], batch)
        else:
            xpool = [global_add_pool(x, batch) for x in xs]
            x = torch.cat(xpool, 1)

        return x, torch.cat(xs, 1)

    def get_embeddings(self, loader):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ret = []
        y = []
        with torch.no_grad():
            for data in loader:

                data = data[0]
                data.to(device)
                x, edge_index, batch = data.x, data.edge_index, data.batch
                if x is None:
                    x = torch.ones((batch.shape[0], 1)).to(device)
                x, _ = self.forward(x, edge_index, batch, None)

                ret.append(x.cpu().numpy())
                y.append(data.y.cpu().numpy())
        ret = np.concatenate(ret, 0)
        y = np.concatenate(y, 0)
        return ret, y

    def get_embeddings_v(self, loader):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ret = []
        y = []
        with torch.no_grad():
            for n, data in enumerate(loader):
                data.to(device)
                x, edge_index, batch = data.x, data.edge_index, data.batch
                if x is None:
                    x = torch.ones((batch.shape[0],1)).to(device)
                x_g, x = self.forward(x, edge_index, batch, None)
                x_g = x_g.cpu().numpy()
                ret = x.cpu().numpy()
                y = data.edge_index.cpu().numpy()
                print(data.y)
                if n == 1:
                   break

        return x_g, ret, y


