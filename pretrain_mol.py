from collections import defaultdict
import time
import datetime
import numpy as np
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
import torch

from model.modules import *
from utils import logger, forward_pass_and_eval
from utils.data_loader_mol import MoleculeDataset_aug_rgcl
from model import model_utils, model_loader 
import tqdm





def pretrain(args, model, device, dataset, optimizer):

    for epoch in range(args.epochs):
        t_epoch = time.time()
        train_losses = defaultdict(list) 
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        model.eval()
        torch.set_grad_enabled(False)
        print('-----init imp-----')
        loader1, loader2 = forward_pass_and_eval.forward_imp(args, model, device, dataset)
        torch.set_grad_enabled(True)
        model.train()


        print('----start CL---------')
        for step, batch in tqdm.tqdm(enumerate(zip(loader1, loader2))):
            #Loss & back
            losses = forward_pass_and_eval.forward_cl(args, model, device, batch)
            optimizer.zero_grad()
            loss = losses["loss"] 
            loss.backward()
            optimizer.step()
            train_losses = model_utils.append_losses(train_losses, losses)
            if args.debug:
                if step > 2:
                    break

        string = logs.result_string("train", epoch, train_losses, t=t_epoch)
        logs.write_to_log_file(string)
        logs.append_train_loss(train_losses)
        logs.draw_loss_curves()
        if (epoch + 1) % 5 == 0:
            logs.create_log(
                args,
                specifier = epoch + 1,
                imp=model.node_imp_estimator,
                gnn=model.gnn,
                optimizer=optimizer,
            )



if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')
    config_path = '/home/shaoqi/code/HAGCL/configs/pretrain_mol.yaml'
    config = model_utils.load_config(config_path)
    args = model_utils.dict_to_namespace(config['pretrain'])
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.time = datetime.datetime.now().isoformat()
    dataset = MoleculeDataset_aug_rgcl(args.data_path + args.dataset, dataset=args.dataset)
    logs = logger.Logger(args)
    logs.write_to_log_file(dataset)
    if args.device is not None:
        logs.write_to_log_file("Using GPU #" + str(args.device))

    model, optimizer, scheduler = model_loader.load_model(args)
    model.to(args.device)
    logs.write_to_log_file(model)


    ##Train model
    pretrain(args, model, args.device, dataset, optimizer)

