from torch_geometric.nn import  global_add_pool, global_mean_pool, global_max_pool
import torch_scatter
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
import tqdm
from utils.data_loader_mol import hete_nodes

class SpatialHeteroModel(nn.Module):
    '''Spatial heterogeneity modeling by using a soft-clustering paradigm.
    '''
    def __init__(self, c_in, nmb_prototype, batch_size, tau=0.5):
        super(SpatialHeteroModel, self).__init__()
        self.l2norm = lambda x: F.normalize(x, dim=1, p=2)
        self.prototypes = nn.Linear(c_in, nmb_prototype, bias=False)
        
        self.tau = tau
        self.d_model = c_in # 64
        self.batch_size = batch_size

        for m in self.modules():
            self.weights_init(m)
    
    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, z1, z2):
        """Compute the contrastive loss of batched data.
        :param z1, z2 (tensor): shape nlvc
        :param loss: contrastive loss
        """
        with torch.no_grad():
            w = self.prototypes.weight.data.clone()
            w = self.l2norm(w) # (6,64)
            self.prototypes.weight.copy_(w)
        
        # l2norm avoids nan of Q in sinkhorn
        # self.prototypes in:64 out:6
        zc1 = self.prototypes(self.l2norm(z1)) # nd -> nk, assignment q, embedding z z1(32,1,128,64)
        zc2 = self.prototypes(self.l2norm(z2)) # nd -> nk
        with torch.no_grad():
            q1 = sinkhorn(zc1.detach())
            q2 = sinkhorn(zc2.detach())
        l1 = - torch.mean(torch.sum(q1 * F.log_softmax(zc2 / self.tau, dim=1), dim=1))
        l2 = - torch.mean(torch.sum(q2 * F.log_softmax(zc1 / self.tau, dim=1), dim=1))
        return l1 + l2
    
@torch.no_grad()
def sinkhorn(out, epsilon=0.05, sinkhorn_iterations=3):
    Q = torch.exp(out / epsilon).t() # Q is K-by-B for consistency with notations from our paper
    B = Q.shape[1] # number of samples to assign
    K = Q.shape[0] # how many prototypes

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    Q /= sum_Q
    
    for it in range(sinkhorn_iterations):
        # normalize each row: total weight per prototype must be 1/K
        Q /= torch.sum(Q, dim=1, keepdim=True)
        Q /= K
        
        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B # the colomns must sum to 1 so that Q is an assignment
    return Q.t()



class graphcl(nn.Module):
    def __init__(self, args, gnn, node_imp_estimator):
        super(graphcl, self).__init__()
        self.gnn = gnn
        self.args = args
        self.node_imp_estimator = node_imp_estimator
        graph_pooling = self.args.graph_pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        self.emb_dim = args.emb_dim
        if args.JK == 'last':
            emb = args.emb_dim
        elif args.JK == 'all':
            emb = args.emb_dim * args.num_layer + args.dataset_num_features
        else:
            emb = args.emb_dim * args.num_layer
        self.projection_head = nn.Sequential(nn.Linear(emb, args.emb_dim), nn.ReLU(inplace=True), nn.Linear(args.emb_dim, args.emb_dim))
        self.prototype_loss = SpatialHeteroModel(emb, args.nmb_prototype, args.batch_size)
        self.prototype_loss2 = SpatialHeteroModel(emb, args.nmb_prototype, args.batch_size)


    def forward(self,batch_):
        imp_batch = batch_.to_homogeneous()
        node_imp = self.node_imp_estimator(imp_batch)
        x = self.gnn(batch_)
        batch = imp_batch.batch
        #cal score
        out, _ = torch_scatter.scatter_max(torch.reshape(node_imp, (1, -1)), batch)
        out = out.reshape(-1, 1)
        out = out[batch]    
        node_imp /= (out * 10)
        node_imp += 0.9
        #node_imp = node_imp.expand(-1, self.emb_dim)

        x = x * node_imp
        x_graph = self.pool(x, batch)
        x_graph = self.projection_head(x_graph)
        return x_graph, x

    def loss_cl(self, x1, x2, temp, task = 'graph'):
        T = temp
        if task == 'node':
            x1 = self.projection_head(x1)
            x2 = self.projection_head(x2)
        batch_size, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()
        return loss

    def loss_infonce(self, x1, x2, temp = 0.1, task = 'graph'):
        T = temp
        if task == 'node':
            x1 = self.projection_head(x1)
            x2 = self.projection_head(x2)
        batch_size, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / sim_matrix.sum(dim=1)
        loss = - torch.log(loss).mean()
        return loss
    
 

    def loss_ra(self, x1, x2, x3, temp, lamda):
        batch_size, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)
        x3_abs = x3.norm(dim=1)

        cp_sim_matrix = torch.einsum('ik,jk->ij', x1, x3) / torch.einsum('i,j->ij', x1_abs, x3_abs)
        cp_sim_matrix = torch.exp(cp_sim_matrix / temp)

        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / temp)

        pos_sim = sim_matrix[range(batch_size), range(batch_size)]

        ra_loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        ra_loss = - torch.log(ra_loss.abs() + 1e-5).mean()

        cp_loss = pos_sim / (cp_sim_matrix.sum(dim=1) + pos_sim)
        cp_loss = - torch.log(cp_loss.abs() + 1e-5).mean()

        loss = ra_loss + lamda * cp_loss

        return ra_loss, cp_loss, loss
    
    def loss_pro(self, z1,z2):
        loss = self.prototype_loss(z1, z2)
        return loss
    
    def loss_pro2(self, z1,z2):
        loss = self.prototype_loss2(z1, z2)
        return loss
    

    def get_embeddings(self, loader):
        print('start generate embeddings....')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ret = []
        y = []
        with torch.no_grad():
            for step, batch in tqdm.tqdm(enumerate(loader)):
                batch.to(device)
                if self.args.use_imp:
                    imp = self.node_imp_estimator(batch)
                    hete_batch = hete_nodes(batch.cpu(),self.args.aug_ratio,imp.cpu())
                    hete_batch.to(device)
                    batch.to(device)
                else:
                    batch.node_type = torch.ones(batch.x.shape[0]).long()
                    batch.edge_type = torch.zeros(batch.edge_index.shape[1]).long()
                    hete_batch = batch.to_heterogeneous(node_type_names = ['0','1'], edge_type_names = [('1', '0', '1'), ('1', '1', '0'), ('0', '2', '1'), ('0', '3', '0')])
                x = self.gnn(hete_batch)
                x = self.pool(x, batch.batch)    
                ret.append(x.cpu().numpy())
                y.append(batch.y.cpu().numpy())
                if self.args.debug:
                    if step > 20:
                        break
        ret = np.concatenate(ret, 0)
        y = np.concatenate(y, 0)
        return ret, y
    
    def get_embeddings_v(self, loader):
        print('start generate embeddings....')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ret = []
        y = []
        with torch.no_grad():
            for step, batch in enumerate(loader):
                batch.to(device)
                batch.node_type = torch.ones(batch.x.shape[0]).long()
                batch.edge_type = torch.zeros(batch.edge_index.shape[1]).long()
                hete_batch = batch.to_heterogeneous(node_type_names = ['0','1'], edge_type_names = [('1', '0', '1'), ('1', '1', '0'), ('0', '2', '1'), ('0', '3', '0')])
                x = self.gnn(hete_batch)
                if self.args.use_imp:
                    imp = self.node_imp_estimator(batch)
                    imp = imp/10 + 0.9
                    x = x * imp
        return x, batch.y



class HGNN_graphpred(torch.nn.Module):
    """
    Extension of GIN to incorporate edge information by concatenation.

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        num_tasks (int): number of tasks in multi-task learning scenario
        drop_ratio (float): dropout rate
        JK (str): last, concat, max or sum.
        graph_pooling (str): sum, mean, max, attention, set2set
        gnn_type: gin, gcn, graphsage, gat
        
    See https://arxiv.org/abs/1810.00826
    JK-net: https://arxiv.org/abs/1806.03536
    """
    def __init__(self, args, imp, gnn):
        super(HGNN_graphpred, self).__init__()
        self.rate = 0.2
        graph_pooling = args.graph_pooling
        self.JK = args.JK
        self.num_tasks = args.num_tasks
        self.emb_dim = args.emb_dim
        self.num_layer = args.num_layer
        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn = gnn
        self.node_imp_estimator = imp

        #Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        else:
            raise ValueError("Invalid graph pooling type.")

        
        if self.JK == "concat":
            self.graph_pred_linear = torch.nn.Linear((self.num_layer + 1) * self.emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_tasks)

    def from_pretrained(self, model_file, device, model_epo):
        # self.gnn.load_state_dict(torch.load(model_file + "_gnn.pth", map_location=lambda storage, loc: storage))
        # self.gnn.load_state_dict(torch.load("model_gin/contextpred.pth", map_location=lambda storage, loc: storage))
        if not model_file == "":
            self.gnn.load_state_dict(torch.load(model_file + "/gnn_{}.pt".format(model_epo)))
            self.gnn.to(device)
            self.node_imp_estimator.load_state_dict(torch.load(model_file + "/imp_{}.pt".format(model_epo)))
            self.node_imp_estimator.to(device)
        print('finish loading model')

    def forward(self, *argv):
        if len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        else:
            raise ValueError("Input a batch data.")
        #node_imp = self.node_imp_estimator(x, edge_index, edge_attr, batch)
        # hete_list = []
        # score_index = 0
        # for i in data.to_data_list():
        #     temp_data = hete_nodes_prob(i, self.rate, node_imp[score_index:score_index + i.x.shape[0]][:,0].cpu().detach(), GPU = True)
        #     temp_data.id = i.id
        #     if len(temp_data['0']) == 2:
        #         del temp_data['0']['id']
        #     if len(temp_data['1']) == 2:
        #         del temp_data['1']['id']  
        #     hete_list.append(temp_data)
        #     score_index = score_index + i.x.shape[0]
        # hete_batch = Batch.from_data_list(hete_list, exclude_keys = ['y'])
        data.node_type = torch.ones(data.x.shape[0]).long()
        data.edge_type = torch.zeros(data.edge_index.shape[1]).long()

        hete_batch = data.to_heterogeneous(node_type_names = ['0','1'], edge_type_names = [('1', '0', '1'), ('1', '1', '0'), ('0', '2', '1'), ('0', '3', '0')])

        node_representation = self.gnn(hete_batch)
        # node_imp = self.node_imp_estimator(x, edge_index, edge_attr, batch)
        #node_representation = torch.mul(node_representation, node_imp)

        return self.graph_pred_linear(self.pool(node_representation, batch))