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
    imp_batch_size = args.batch_size
    
    loader = DataLoader(dataset, batch_size=imp_batch_size, num_workers=args.num_workers, shuffle=False)
    for step, batch in tqdm.tqdm(enumerate(loader)):
        node_index_start = step*imp_batch_size
        node_index_end = min(node_index_start + imp_batch_size - 1, len(dataset)-1)
        batch = batch.to(device)
        node_imp = model.node_imp_estimator(batch.x, batch.edge_index, batch.edge_attr, batch.batch).detach()
        if node_index_end:
            dataset.node_score[dataset.slices['x'][node_index_start]:dataset.slices['x'][node_index_end + 1]] = torch.squeeze(node_imp.half())
        else:
            dataset.node_score = torch.squeeze(node_imp.half())
        if args.debug:
            break

    dataset1 = deepcopy(dataset)
    dataset1 = dataset1.shuffle()
    dataset2 = deepcopy(dataset1)
    dataset3 = deepcopy(dataset1)

    dataset1.aug, dataset1.aug_ratio = args.aug0, args.aug_ratio
    dataset2.aug, dataset2.aug_ratio = args.aug1, args.aug_ratio
    dataset3.aug, dataset3.aug_ratio = args.aug2, args.aug_ratio

    loader1 = DataLoader(dataset1, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    loader2 = DataLoader(dataset2, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    loader3 = DataLoader(dataset3, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    #del dataset1, dataset2, dataset3
    del dataset1, dataset2, dataset3
    gc.collect()
    return loader1, loader2, loader3

def forward_cl(args, model, device, batch):
    start = time.time()
    losses = defaultdict(lambda: torch.zeros((), device=args.device.type))
    ################# loader ###############
    batch1, batch2, batch3 = batch
    batch1 = batch1.to(device)
    batch1.cp = False
    batch2 = batch2.to(device)
    batch2.cp = False
    batch3 = batch3.to(device)
    batch3.cp = False
    ################### CL ####################
    x1, z1 = model(batch1)
    x2, z2 = model(batch2)
    x3, z3 = model(batch3)

    #ra_loss, cp_loss, loss = model.loss_ra(x1, x2, x3, args.loss_temp, args.lamda)

    if args.task == 'pd':
        loss_inf1 = model.loss_cl(z1,z2,args.loss_temp, task = 'node')
        loss_inf2 = model.loss_cl(z3,z2,args.loss_temp, task = 'node')
    else:
        loss_inf1 = model.loss_cl(x1, x2, args.loss_temp)
        loss_inf2 = model.loss_cl(x3, x2, args.loss_temp)
    loss_pro1 = model.loss_pro(z1,z2)
    loss_pro2 = model.loss_pro2(z3,z2)


    #################### MAIN LOSSES ####################
    ## KL loss ##
    losses["loss_inf"] = args.lamda1 * loss_inf1 + args.lamda2 * loss_inf2
    losses["loss_pro"] = args.lamda3 * loss_pro1 + args.lamda4 * loss_pro2
    losses["loss"] = args.lamda1 * loss_inf1 + args.lamda2 * loss_inf2 + args.lamda3 * loss_pro1 + args.lamda4 * loss_pro2
    #losses["loss"] = loss_pro1
    losses["inference time"] = time.time() - start

    return losses


