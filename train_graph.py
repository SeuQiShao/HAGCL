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
from utils.data_loader_tu import TUDataset_aug
from torch_geometric.loader import DataLoader
from utils.evaluate_embedding import evaluate_embedding
from model import model_utils, model_loader 
import tqdm





def train(args, model, device, dataset, optimizer):
    accuracies = {'val': [], 'test': []}
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
        if (epoch + 1) % 1== 0:
            logs.create_log(
                args,
                specifier = epoch + 1,
                imp=model.node_imp_estimator,
                gnn=model.gnn,
                optimizer=optimizer,
            )

            model.eval()
            emb, y = model.get_embeddings(dataloader_eval)
            acc_val, acc = evaluate_embedding(emb, y)
            accuracies['val'].append(acc_val)
            accuracies['test'].append(acc)
            un_string = 'val: '+ str(acc_val) + ' test: '+ str(acc)
            logs.write_to_log_file(un_string)
    logs.finetune_result(args,accuracies['test'],accuracies['val'])




if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')
    config_path = '/home/shaoqi/code/HAGCL/configs/train_graph.yaml'
    config = model_utils.load_config(config_path)
    args = model_utils.dict_to_namespace(config)
    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.time = datetime.datetime.now().isoformat()
    dataset = TUDataset_aug(args.data_path + args.dataset, name=args.dataset)
    dataset_eval = TUDataset_aug(args.data_path + args.dataset, name=args.dataset, aug='none')
    dataloader_eval = DataLoader(dataset_eval, batch_size=args.batch_size, num_workers=args.num_workers)
    print(len(dataset))
    print(dataset.get_num_feature())
    try:
        args.dataset_num_features, args.dataset_num_attr = dataset.get_num_feature()
    except:
        args.dataset_num_features = 1
        args.dataset_num_attr = 1

    logs = logger.Logger(args)
    logs.write_to_log_file(dataset)
    if args.device is not None:
        logs.write_to_log_file("Using GPU #" + str(args.device))

    model, optimizer, scheduler = model_loader.load_model(args)
    model.to(args.device)
    logs.write_to_log_file(model)


    ##Train model
    train(args, model, args.device, dataset, optimizer)

