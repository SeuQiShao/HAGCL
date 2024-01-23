from typing import Callable, List, Optional
import numpy as np
import torch
from itertools import chain
import sys
import json
sys.path.append("..") 
sys.path.append(".") 
from model.model_utils import *
from sklearn.preprocessing import MinMaxScaler
from utils.data_loader_mol import hete_nodes_prob
import pickle
import pandas as pd
import networkx as nx
import os
import os.path as osp

from copy import deepcopy
import pdb
from torch_geometric.data import InMemoryDataset, download_url, Data
from torch_geometric.io import read_planetoid_data, read_npz
from torch_geometric.utils import to_undirected

class Planetoid_aug(InMemoryDataset):
    r"""The citation network datasets "Cora", "CiteSeer" and "PubMed" from the
    `"Revisiting Semi-Supervised Learning with Graph Embeddings"
    <https://arxiv.org/abs/1603.08861>`_ paper.
    Nodes represent documents and edges represent citation links.
    Training, validation and test splits are given by binary masks.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"Cora"`,
            :obj:`"CiteSeer"`, :obj:`"PubMed"`).
        split (string): The type of dataset split
            (:obj:`"public"`, :obj:`"full"`, :obj:`"geom-gcn"`,
            :obj:`"random"`).
            If set to :obj:`"public"`, the split will be the public fixed split
            from the `"Revisiting Semi-Supervised Learning with Graph
            Embeddings" <https://arxiv.org/abs/1603.08861>`_ paper.
            If set to :obj:`"full"`, all nodes except those in the validation
            and test sets will be used for training (as in the
            `"FastGCN: Fast Learning with Graph Convolutional Networks via
            Importance Sampling" <https://arxiv.org/abs/1801.10247>`_ paper).
            If set to :obj:`"geom-gcn"`, the 10 public fixed splits from the
            `"Geom-GCN: Geometric Graph Convolutional Networks"
            <https://openreview.net/forum?id=S1e2agrFvS>`_ paper are given.
            If set to :obj:`"random"`, train, validation, and test sets will be
            randomly generated, according to :obj:`num_train_per_class`,
            :obj:`num_val` and :obj:`num_test`. (default: :obj:`"public"`)
        num_train_per_class (int, optional): The number of training samples
            per class in case of :obj:`"random"` split. (default: :obj:`20`)
        num_val (int, optional): The number of validation samples in case of
            :obj:`"random"` split. (default: :obj:`500`)
        num_test (int, optional): The number of test samples in case of
            :obj:`"random"` split. (default: :obj:`1000`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)

    Stats:
        .. list-table::
            :widths: 10 10 10 10 10
            :header-rows: 1

            * - Name
              - #nodes
              - #edges
              - #features
              - #classes
            * - Cora
              - 2,708
              - 10,556
              - 1,433
              - 7
            * - CiteSeer
              - 3,327
              - 9,104
              - 3,703
              - 6
            * - PubMed
              - 19,717
              - 88,648
              - 500
              - 3
    """

    url = 'https://github.com/kimiyoung/planetoid/raw/master/data'
    geom_gcn_url = ('https://raw.githubusercontent.com/graphdml-uiuc-jlu/'
                    'geom-gcn/master')

    def __init__(self, root: str, name: str, split: str = "public",
                 num_train_per_class: int = 20, num_val: int = 500,
                 num_test: int = 1000, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None, aug=None, aug_ratio=0.9):
        self.name = name
        self.pd_datset = ['Cora', 'Citeseer', 'Pubmed']
        self.am_datset = ['amc', 'amp', 'coc']
        self.wiki_dataset = ['wikics']
        self.split = split.lower()
        self.is_undirected = True
        assert self.split in ['public', 'full', 'geom-gcn', 'random']

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

        if split == 'full':
            data = self.get(0)
            data.train_mask.fill_(True)
            data.train_mask[data.val_mask | data.test_mask] = False
            self.data, self.slices = self.collate([data])

        elif split == 'random':
            data = self.get(0)
            data.train_mask.fill_(False)
            for c in range(self.num_classes):
                idx = (data.y == c).nonzero(as_tuple=False).view(-1)
                idx = idx[torch.randperm(idx.size(0))[:num_train_per_class]]
                data.train_mask[idx] = True

            remaining = (~data.train_mask).nonzero(as_tuple=False).view(-1)
            remaining = remaining[torch.randperm(remaining.size(0))]

            data.val_mask.fill_(False)
            data.val_mask[remaining[:num_val]] = True

            data.test_mask.fill_(False)
            data.test_mask[remaining[num_val:num_val + num_test]] = True

            self.data, self.slices = self.collate([data])

        self.node_score = torch.zeros(self.data['x'].shape[0], dtype=torch.half)
        self.aug = aug
        self.aug_ratio = aug_ratio

    @property
    def raw_dir(self) -> str:
        if self.split == 'geom-gcn':
            return osp.join(self.root, self.name, 'geom-gcn', 'raw')
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        if self.split == 'geom-gcn':
            return osp.join(self.root, self.name, 'geom-gcn', 'processed')
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        if self.name in self.pd_datset:
            names = ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']
            return [f'ind.{self.name.lower()}.{name}' for name in names]
        elif self.name in self.am_datset:
            if self.name == 'amc':
                return ['amazon_electronics_computers.npz']
            elif self.name == 'amp':
                return ['amazon_electronics_photo.npz']
            elif self.name == 'coc':
                return ['ms_academic_cs.npz']
            else:
                print('No data!')
        elif self.name in self.wiki_dataset:
            return ['data.json']





    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    # def download(self):
    #     for name in self.raw_file_names:
    #         download_url(f'{self.url}/{name}', self.raw_dir)
    #     if self.split == 'geom-gcn':
    #         for i in range(10):
    #             url = f'{self.geom_gcn_url}/splits/{self.name.lower()}'
    #             download_url(f'{url}_split_0.6_0.2_{i}.npz', self.raw_dir)

    def process(self):
        #data = read_npz(self.raw_paths[0], to_undirected=True)
        if self.name in self.pd_datset:
            data = read_planetoid_data(self.raw_dir, self.name)

            if self.split == 'geom-gcn':
                train_masks, val_masks, test_masks = [], [], []
                for i in range(10):
                    name = f'{self.name.lower()}_split_0.6_0.2_{i}.npz'
                    splits = np.load(osp.join(self.raw_dir, name))
                    train_masks.append(torch.from_numpy(splits['train_mask']))
                    val_masks.append(torch.from_numpy(splits['val_mask']))
                    test_masks.append(torch.from_numpy(splits['test_mask']))
                data.train_mask = torch.stack(train_masks, dim=1)
                data.val_mask = torch.stack(val_masks, dim=1)
                data.test_mask = torch.stack(test_masks, dim=1)
            data = data if self.pre_transform is None else self.pre_transform(data)
            torch.save(self.collate([data]), self.processed_paths[0])
        elif self.name in self.am_datset:
            data = read_npz(self.raw_paths[0])
            data = data if self.pre_transform is None else self.pre_transform(data)
            data, slices = self.collate([data])
            torch.save((data, slices), self.processed_paths[0])
        elif self.name in self.wiki_dataset:
            with open(self.raw_paths[0], 'r') as f:
                data = json.load(f)

            x = torch.tensor(data['features'], dtype=torch.float)
            y = torch.tensor(data['labels'], dtype=torch.long)

            edges = [[(i, j) for j in js] for i, js in enumerate(data['links'])]
            edges = list(chain(*edges))
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            if self.is_undirected:
                edge_index = to_undirected(edge_index, num_nodes=x.size(0))

            train_mask = torch.tensor(data['train_masks'], dtype=torch.bool)
            train_mask = train_mask.t().contiguous()

            val_mask = torch.tensor(data['val_masks'], dtype=torch.bool)
            val_mask = val_mask.t().contiguous()

            test_mask = torch.tensor(data['test_mask'], dtype=torch.bool)

            stopping_mask = torch.tensor(data['stopping_masks'], dtype=torch.bool)
            stopping_mask = stopping_mask.t().contiguous()

            data = Data(x=x, y=y, edge_index=edge_index, train_mask=train_mask,
                        val_mask=val_mask, test_mask=test_mask,
                        stopping_mask=stopping_mask)

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(self.collate([data]), self.processed_paths[0])



    def __repr__(self) -> str:
        return f'{self.name}()'

    def get_num_feature(self, idx = 0):
        data = self.data
        _, num_feature = data.x.size()
        if 'edge_attr' in self.data.keys:
            _, num_attr = data.edge_attr.size()
        else:
            num_attr = 1
        return num_feature, num_attr

    def drop_feature(self, x: torch.Tensor, drop_prob: float) -> torch.Tensor:
        device = x.device
        drop_mask = torch.empty((x.size(1),), dtype=torch.float32).uniform_(0, 1) < drop_prob
        drop_mask = drop_mask.to(device)
        x = x.clone()
        x[:, drop_mask] = 0
        return x


    def get(self, idx):
        data = self.data
        """
        edge_index = data.edge_index
        node_num = data.x.size()[0]
        edge_num = data.edge_index.size()[1]
        data.edge_index = torch.tensor([[edge_index[0, n], edge_index[1, n]] for n in range(edge_num) if edge_index[0, n] < node_num and edge_index[1, n] < node_num] + [[n, n] for n in range(node_num)], dtype=torch.int64).t()
        """

        node_num = data.edge_index.max()
        sl = torch.tensor([[n, n] for n in range(node_num)]).t()
        #data.edge_index = torch.cat((data.edge_index, sl), dim=1)
        if self.aug == 'dropN':
            nodes_score = self.node_score
            data_aug = hete_nodes_prob(data, self.aug_ratio, nodes_score.cpu())
        elif self.aug == 'dropN_cp':
            nodes_score = self.node_score
            data_aug = hete_nodes_prob(data, self.aug_ratio, nodes_score.cpu(), cp = True)
        elif self.aug == 'dropN_node':
            x = self.drop_feature(data.x, self.aug_ratio)
            new_data = Data(x = x, y = data.y, edge_index = data.edge_index)
            # new_data.node_type = torch.ones(new_data.x.shape[0]).long()
            # new_data.edge_type = torch.zeros(new_data.edge_index.shape[1]).long()
            # data_aug = new_data.to_heterogeneous(node_type_names = ['0','1'], edge_type_names = [('1', '0', '1'), ('1', '1', '0'), ('0', '2', '1'), ('0', '3', '0')])
            nodes_score = self.node_score
            data_aug = hete_nodes_prob(new_data, self.aug_ratio, nodes_score.cpu())

        elif self.aug == 'none':
            data_aug = deepcopy(data)
            # data_aug.x = torch.ones((data.edge_index.max() + 1, 1))
            # data_cp = deepcopy(data)
            # data_cp.x = torch.ones((data.edge_index.max() + 1, 1))
        elif self.aug == 'hete_none':
            batch = deepcopy(data)
            batch.node_type = torch.ones(batch.x.shape[0]).long()
            batch.edge_type = torch.zeros(batch.edge_index.shape[1]).long()
            data_aug = batch.to_heterogeneous(node_type_names = ['0','1'], edge_type_names = [('1', '0', '1'), ('1', '1', '0'), ('0', '2', '1'), ('0', '3', '0')])

        else:
            print('augmentation error')
            assert False

        return data_aug
