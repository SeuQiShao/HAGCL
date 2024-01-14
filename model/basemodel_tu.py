import os.path as osp
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_add_pool, MessagePassing, HeteroConv, to_hetero, Linear
from torch_geometric.utils import softmax, add_self_loops
from torch_scatter import scatter_add
import torch_geometric.transforms as T


import numpy as np
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
import sys
import torch_scatter

class linears(MessagePassing):
    def __init__(self, in_dim, out_dim, aggr = 'add', coef = 0.1, header = 1):
        super(linears, self).__init__()
        self.out_dim = out_dim * header
        self.mlp = Linear(in_dim, self.out_dim)
        self.aggr = aggr
        self.coef = coef

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        if edge_attr_dict is None:
            edge_attr_dict = torch.zeros(1)

        return self.propagate(edge_index_dict, x=x_dict, edge_attr=edge_attr_dict)

    def message(self, x_j, edge_attr):
        if x_j.shape[0] == edge_attr.shape[0]:
            return x_j + edge_attr
        else:
            return x_j

    def update(self, aggr_out):
        return self.mlp(aggr_out) * self.coef


class GINConv(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.

    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        embed_input (bool): whether to embed input or not. 
        

    See https://arxiv.org/abs/1810.00826
    """
    def __init__(self, emb_dim, aggr="add"):
        super(GINConv, self).__init__()
        #multi-layer perceptron
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, emb_dim))
        # self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        # self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        # torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        # torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        self.eps = 0.1
        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        # #add self loops in the edge space
        # edge_index = add_self_loops(edge_index, num_nodes = x.size(0))

        # #add features corresponding to self-loop edges.
        # self_loop_attr = torch.zeros(x.size(0), 2)
        # self_loop_attr[:,0] = 4 #bond type for self-loop edge
        # self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        # edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        # edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])
        if edge_attr is None:
            edge_attr = torch.zeros(1)

        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        if x_j.shape[0] == edge_attr.shape[0]:
            return x_j + edge_attr
        else:
            return x_j

    def update(self, aggr_out, x):
        return self.mlp(aggr_out + (1 + self.eps) * x)

class GCNConv(MessagePassing):

    def __init__(self, in_dim, out_dim,edge_dim, aggr="add"):
        super(GCNConv, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.edge_dim = edge_dim
        self.linear = torch.nn.Linear(in_dim, out_dim)
        # self.edge_embedding1 = torch.nn.Embedding(num_bond_type, out_dim)
        # self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, out_dim)
        self.edge_embeddings = torch.nn.Linear(edge_dim, out_dim)
        # torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        # torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embeddings.weight.data)

        self.aggr = aggr

    def norm(self, edge_index, num_nodes, dtype):
        ### assuming that self-loops have been already added in edge_index
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype, device=edge_index.device)
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_attr):
        #add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes = x.size(0))
        if edge_attr is not None:
            #add features corresponding to self-loop edges.
            self_loop_attr = torch.ones(x.size(0), self.edge_dim)
            #self_loop_attr[:, 0] = 4 #bond type for self-loop edge
            self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)

            edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

            #edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])
            edge_embeddings = torch.sigmoid(self.edge_embeddings(edge_attr))
        else:
            edge_embeddings = torch.zeros(1)
        norm = self.norm(edge_index[0], x.size(0), x.dtype)

        x = self.linear(x)

        return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings, norm=norm)

    def message(self, x_j, edge_attr, norm):
        if x_j.shape[0] == edge_attr.shape[0]:
            return norm.view(-1, 1) * (x_j + edge_attr)
        else:
            return norm.view(-1, 1) * (x_j)




class GNN_imp_estimator(torch.nn.Module):
    """

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat

    Output:
        node representations

    """

    def __init__(self, num_layer, seq_len, attr_len, emb_dim, JK="last", drop_ratio=0):
        super(GNN_imp_estimator, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.seq_len = seq_len
        self.attr_len = attr_len

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        # self.x_embedding1 = torch.nn.Embedding(num_atom_type, emb_dim)
        # self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, emb_dim)
        self.x_embeddings = torch.nn.Linear(self.seq_len, emb_dim)
        


        torch.nn.init.xavier_uniform_(self.x_embeddings.weight.data)
        #torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        ###List of MLPs
        self.gnns = torch.nn.ModuleList()
        self.gnns.append(GCNConv(emb_dim, 128, self.attr_len))
        self.gnns.append(GCNConv(128, 64, self.attr_len))
        self.gnns.append(GCNConv(64, 32, self.attr_len))

        ###List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        self.batch_norms.append(torch.nn.BatchNorm1d(128))
        self.batch_norms.append(torch.nn.BatchNorm1d(64))
        self.batch_norms.append(torch.nn.BatchNorm1d(32))

        self.linear = torch.nn.Linear(32, 1)

    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        else:
            raise ValueError("unmatched number of arguments.")

        x = self.x_embeddings(x)

        h_list = [x]
        for layer in range(len(self.gnns)):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == len(self.gnns) - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)

        node_representation = h_list[-1]
        node_representation = self.linear(node_representation)
        node_representation = softmax(node_representation - node_representation.max(), batch)

        return node_representation


class HGNN(torch.nn.Module):
    def __init__(self, num_layer, emb_dim, seq_len, attr_len, JK = "last", drop_ratio=0, gnn_type = "gin", add_loop = True, headers = 1):
        super(HGNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.attr_len = attr_len
        self.JK = JK
        self.headers = headers
        self.add_loop = add_loop
        self.node_type = ['0','1']
        self.edge_type = [('1', '0', '1'), ('1', '1', '0'), ('0', '2', '1'), ('0', '3', '0')]
        self.meta_data = (self.node_type, self.edge_type)
        self.gnn_type = gnn_type
        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        # self.x_embedding1 = torch.nn.Embedding(num_atom_type, emb_dim)
        # self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, emb_dim)
        # self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        # self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)
        # torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        # torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)
        # torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        # torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        self.x_embeddings = torch.nn.Linear(seq_len, emb_dim)
        self.edge_embeddings = torch.nn.Linear(attr_len, emb_dim)
        torch.nn.init.xavier_uniform_(self.x_embeddings.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embeddings.weight.data)
        # define model:
        if gnn_type == "gin":
            define_model = GINConv(emb_dim)

        ###List of MLPs
        self.header_mlp = torch.nn.ModuleList()
        self.gnns = torch.nn.ModuleList()
        conv_dict = {}
        for i in self.edge_type:
            if i == ('1','0','1'):
                conv_dict[i] = define_model
            else:
                conv_dict[i] = linears(emb_dim, emb_dim, header= self.headers)
        conv = HeteroConv(conv_dict, aggr='mean')
        for layer in range(num_layer):
            self.gnns.append(conv)
            self.header_mlp.append(Linear(self.headers * emb_dim, emb_dim))
        ###List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))




    def forward(self, batch):
        # batch: hete graph
        # batch0 = deepcopy(batch)
        if self.add_loop == True:
            T.AddSelfLoops(attr='edge_attr', fill_value = torch.tensor([1]*self.attr_len))(batch)
        x_dict, edge_index_dict, edge_attr_dict, batch_dict = batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict, batch.batch_dict 

        #embeddings
        x_dict = {key: self.x_embeddings(x) for key, x in x_dict.items()}
        if len(edge_attr_dict) == 0:
            edge_attr_dict = {key:None for key, edge_attr in edge_index_dict.items()}
        else:
            edge_attr_dict = {key:self.edge_embeddings(edge_attr) for key, edge_attr in edge_attr_dict.items()}
        h_dict_list = [x_dict]

        for layer in range(self.num_layer):
            h_dict = self.gnns[layer](h_dict_list[layer], edge_index_dict, edge_attr_dict)
            h_dict = {key:self.batch_norms[layer](x) for key, x in h_dict.items()}
            #h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h_dict = {key:F.dropout(x, self.drop_ratio, training = self.training) for key, x in h_dict.items()}
            else:
                h_dict = {key:F.dropout(F.relu(x), self.drop_ratio, training = self.training) for key, x in h_dict.items()}
            h_dict_list.append(h_dict)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_dict_list[-1]

        batch._node_store_dict['0']['x'] = node_representation['0']
        batch._node_store_dict['1']['x'] = node_representation['1']
        return batch.to_homogeneous().x
    





class Encoder(torch.nn.Module):
    def __init__(self, num_features, dim, num_gc_layers, pooling):
        super(Encoder, self).__init__()

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



class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        try:
            num_features = dataset.num_features
        except:
            num_features = 1
        dim = 32

        self.encoder = Encoder(num_features, dim)

        self.fc1 = Linear(dim*5, dim)
        self.fc2 = Linear(dim, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        if x is None:
            x = torch.ones(batch.shape[0]).to(device)

        x, _ = self.encoder(x, edge_index, batch)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)


def train(epoch):
    model.train()

    if epoch == 51:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.5 * param_group['lr']

    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.x, data.edge_index, data.batch)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()

    return loss_all / len(train_dataset)

def test(loader):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        output = model(data.x, data.edge_index, data.batch)
        pred = output.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)

if __name__ == '__main__':

    for percentage in [ 1.]:
        for DS in [sys.argv[1]]:
            if 'REDDIT' in DS:
                epochs = 200
            else:
                epochs = 100
            path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', DS)
            accuracies = [[] for i in range(epochs)]
            #kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
            dataset = TUDataset(path, name=DS) #.shuffle()
            num_graphs = len(dataset)
            print('Number of graphs', len(dataset))
            dataset = dataset[:int(num_graphs * percentage)]
            dataset = dataset.shuffle()

            kf = KFold(n_splits=10, shuffle=True, random_state=None)
            for train_index, test_index in kf.split(dataset):

                # x_train, x_test = x[train_index], x[test_index]
                # y_train, y_test = y[train_index], y[test_index]
                train_dataset = [dataset[int(i)] for i in list(train_index)]
                test_dataset = [dataset[int(i)] for i in list(test_index)]
                print('len(train_dataset)', len(train_dataset))
                print('len(test_dataset)', len(test_dataset))

                train_loader = DataLoader(train_dataset, batch_size=128)
                test_loader = DataLoader(test_dataset, batch_size=128)
                # print('train', len(train_loader))
                # print('test', len(test_loader))

                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = Net().to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

                for epoch in range(1, epochs+1):
                    train_loss = train(epoch)
                    train_acc = test(train_loader)
                    test_acc = test(test_loader)
                    accuracies[epoch-1].append(test_acc)
                    tqdm.write('Epoch: {:03d}, Train Loss: {:.7f}, '
                          'Train Acc: {:.7f}, Test Acc: {:.7f}'.format(epoch, train_loss,
                                                                       train_acc, test_acc))
            tmp = np.mean(accuracies, axis=1)
            print(percentage, DS, np.argmax(tmp), np.max(tmp), np.std(accuracies[np.argmax(tmp)]))
            input()
