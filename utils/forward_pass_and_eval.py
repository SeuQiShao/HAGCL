from collections import defaultdict
import time
import torch
from torch_geometric.loader import DataLoader
from copy import deepcopy
import tqdm
import numpy as np
import gc
from model.modules import *


def forward_imp(args, model, device, dataset):
    # generate graph augmentations
    dataset.aug = "none"
    imp_batch_size = 32
    
    loader = DataLoader(dataset, batch_size=imp_batch_size, num_workers=args.num_workers, shuffle=False)
    for step, batch in tqdm.tqdm(enumerate(loader)):
        node_index_start = step*imp_batch_size
        node_index_end = min(node_index_start + imp_batch_size - 1, len(dataset)-1)
        batch = batch.to(device)
        node_imp = model.node_imp_estimator(batch.x, batch.edge_index, batch.edge_attr, batch.batch).detach()
        dataset.node_score[dataset.slices['x'][node_index_start]:dataset.slices['x'][node_index_end + 1]] = torch.squeeze(node_imp.half())
        if args.debug:
            break

    dataset1 = deepcopy(dataset)
    #dataset1 = dataset1.shuffle()
    dataset2 = deepcopy(dataset1)
    #dataset3 = deepcopy(dataset1)

    dataset1.aug, dataset1.aug_ratio = args.aug1, args.aug_ratio
    dataset2.aug, dataset2.aug_ratio = args.aug2, args.aug_ratio
    #dataset3.aug, dataset3.aug_ratio = args.aug2 + '_cp', args.aug_ratio

    loader1 = DataLoader(dataset1, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    loader2 = DataLoader(dataset2, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    #loader3 = DataLoader(dataset3, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    #del dataset1, dataset2, dataset3
    del dataset1, dataset2
    gc.collect()
    return loader1, loader2

def forward_cl(args, model, device, batch):
    start = time.time()
    losses = defaultdict(lambda: torch.zeros((), device=args.device.type))
    ################# loader ###############
    batch1, batch2 = batch
    batch1 = batch1.to(device)
    batch1.cp = False
    batch2 = batch2.to(device)
    batch2.cp = False
    # batch3 = batch3.to(device)
    # batch3.cp = True
    ################### CL ####################
    x1, z1 = model(batch1)
    x2, z2 = model(batch2)
    #x3 = model(batch3)

    #ra_loss, cp_loss, loss = model.loss_ra(x1, x2, x3, args.loss_temp, args.lamda)
    loss_inf = model.loss_cl(x1, x2, args.loss_temp)
    loss_pro = model.loss_pro(z1,z2)


    #################### MAIN LOSSES ####################
    ## KL loss ##
    losses["loss_inf"] = loss_inf
    losses["loss_pro"] = loss_pro
    losses["loss"] = loss_inf + loss_pro
    losses["inference time"] = time.time() - start

    return losses


