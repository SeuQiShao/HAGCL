import numpy as np
import torch
import yaml
import argparse
import torch.nn.functional as F
import torch.distributions as tdist
from torch_geometric.loader import DataLoader
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score
from collections import defaultdict
from scipy.spatial.transform import Rotation as R
import networkx as nx
import scipy.sparse as sp
import pandas as pd
from networkx.algorithms.community import greedy_modularity_communities
from utils.splitters import scaffold_split, random_split, random_scaffold_split

def hete_cat(dict_list):
    keys = dict_list[0].keys() 
    cat_list = [len(dict_list[0].keys() ) * []]
    for dict in dict_list:
        k = 0
        for key in keys:
            cat_list[k].append(dict[key])
            k = k + 1
    h = {key: torch.cat(i,1) for i in cat_list}
    return h



def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def dict_to_namespace(d):
    namespace = argparse.Namespace()
    for key, value in d.items():
        setattr(namespace, key, value)
    return namespace

def append_losses(losses_list, losses):
    for loss, value in losses.items():
        if type(value) == float:
            losses_list[loss].append(value)
        elif type(value) == defaultdict:
            if losses_list[loss] == []:
                losses_list[loss] = defaultdict(list)
            for idx, elem in value.items():
                losses_list[loss][idx].append(elem)
        else:
            losses_list[loss].append(value.item())
    return losses_list


def average_listdict(listdict, num_atoms):
    average_list = [None] * num_atoms
    for k, v in listdict.items():
        average_list[k] = sum(v) / len(v)
    return average_list


def find_max_index(list):
    index = []
    max_list = np.max(list)
    for i in range(len(list)):
        if list[i] == max_list:
            index.append(i)
    return max_list, index

def find_min_index(list):
    index = []
    min_list = np.min(list)
    for i in range(len(list)):
        if list[i] == min_list:
            index.append(i)
    return min_list, index

def Frobenius(x, y):
    return torch.sqrt(torch.sum((x-y)**2))

def get_mol_dataloader(args, dataset):
    if args.split == "scaffold":
        smiles_list = pd.read_csv(args.data_path + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1)
        print("scaffold")
    elif args.split == "random":
        train_dataset, valid_dataset, test_dataset = random_split(dataset, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = args.seed)
        print("random")
    elif args.split == "random_scaffold":
        smiles_list = pd.read_csv(args.data_path + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = random_scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = args.seed)
        print("random scaffold")
    else:
        raise ValueError("Invalid split option.")

    print(train_dataset[0])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return train_loader, val_loader, test_loader



if __name__ == '__main__':
    print('Hello World')