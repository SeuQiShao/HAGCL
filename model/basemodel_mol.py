import torch
from torch_geometric.nn import MessagePassing, HeteroConv, to_hetero, Linear
from torch_geometric.utils import add_self_loops, degree, softmax
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.nn.aggr import AttentionalAggregation
from torch_geometric.data import HeteroData, Batch
import torch_geometric.transforms as T
from copy import deepcopy
from model.model_utils import hete_cat
from utils.data_loader_mol import hete_nodes_prob

num_atom_type = 120 #including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 6 #including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3 


class linears(MessagePassing):
    def __init__(self, in_dim, out_dim, aggr = 'add', coef = 0.1, header = 1):
        super(linears, self).__init__()
        self.out_dim = out_dim * header
        self.mlp = Linear(in_dim, self.out_dim)
        self.aggr = aggr
        self.coef = coef

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):


        return self.propagate(edge_index_dict, x=x_dict, edge_attr=edge_attr_dict)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

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

        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out, x):
        return self.mlp(aggr_out + (1 + self.eps) * x)


class GCNConv(MessagePassing):

    def __init__(self, in_dim, out_dim, aggr="add"):
        super(GCNConv, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = torch.nn.Linear(in_dim, out_dim)
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, out_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, out_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

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

        #add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])

        norm = self.norm(edge_index[0], x.size(0), x.dtype)

        x = self.linear(x)

        return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings, norm=norm)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * (x_j + edge_attr)


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

    def __init__(self, num_layer, emb_dim, JK="last", drop_ratio=0):
        super(GNN_imp_estimator, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.x_embedding1 = torch.nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, emb_dim)

        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        ###List of MLPs
        self.gnns = torch.nn.ModuleList()
        self.gnns.append(GCNConv(emb_dim, 128))
        self.gnns.append(GCNConv(128, 64))
        self.gnns.append(GCNConv(64, 32))

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

        x = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])

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
    def __init__(self, num_layer, emb_dim, JK = "last", drop_ratio=0, gnn_type = "gin", add_loop = True, headers = 1):
        super(HGNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.headers = headers
        self.add_loop = add_loop
        self.node_type = ['0','1']
        self.edge_type = [('1', '0', '1'), ('1', '1', '0'), ('0', '2', '1'), ('0', '3', '0')]
        self.meta_data = (self.node_type, self.edge_type)
        self.gnn_type = gnn_type
        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.x_embedding1 = torch.nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, emb_dim)
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)
        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        # define model:
        if gnn_type == "gin":
            define_model = GINConv(emb_dim)
        elif gnn_type == "gcn":
            define_model = GCNConv(emb_dim, emb_dim)


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
            T.AddSelfLoops(attr='edge_attr', fill_value = torch.tensor([4,0]))(batch)
        x_dict, edge_index_dict, edge_attr_dict, batch_dict = batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict, batch.batch_dict 

        #embeddings
        x_dict = {key:self.x_embedding1(x[:,0]) + self.x_embedding2(x[:,1]) for key, x in x_dict.items()}
        edge_attr_dict = {key:self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1]) for key, edge_attr in edge_attr_dict.items()}
        h_dict_list = [x_dict]
        for layer in range(self.num_layer):
            h_dict = self.gnns[layer](h_dict_list[layer], edge_index_dict, edge_attr_dict)
            h_dict = {key:self.batch_norms[layer](x) for key, x in h_dict.items()}
            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h_dict = {key:F.dropout(x, self.drop_ratio, training = self.training) for key, x in h_dict.items()}
            else:
                h_dict = {key:F.dropout(F.relu(x), self.drop_ratio, training = self.training) for key, x in h_dict.items()}
            h_dict_list.append(h_dict)

        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = hete_cat(h_dict_list) # todo
        elif self.JK == "last":
            node_representation = h_dict_list[-1]

        batch._node_store_dict['0']['x'] = node_representation['0']
        batch._node_store_dict['1']['x'] = node_representation['1']
        return batch.to_homogeneous().x
    


if __name__ == "__main__":
    pass

