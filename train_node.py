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
from utils.data_loader_pd import Planetoid_aug
from utils.Evaluator import get_split, LREvaluator
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
        loader1, loader2, loader3 = forward_pass_and_eval.forward_imp(args, model, device, dataset)
        torch.set_grad_enabled(True)
        model.train()


        print('----start CL---------')
        for step, batch in tqdm.tqdm(enumerate(zip(loader1, loader2, loader3))):
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
        if (epoch + 1) % args.val_lag== 0:
            logs.create_log(
                args,
                specifier = epoch + 1,
                imp=model.node_imp_estimator,
                gnn=model.gnn,
                optimizer=optimizer,
            )
            start = time.time() 
            model.eval()
            emb, y = model.get_embeddings_v(dataloader_eval)
            best_result = test(emb, y)
            accuracies['val'].append(best_result['accuracy_val'])
            accuracies['test'].append(best_result['accuracy'])
            un_string = 'val: '+ str(best_result['accuracy_val']) + ' test: '+ str(best_result['accuracy'])
            logs.write_to_log_file(un_string)
            print('LR_time: ', time.time() - start)
    logs.finetune_result(args,accuracies['test'],accuracies['val'])


def test(z, y):
    split = get_split(num_samples=z.shape[0], train_ratio=0.1, test_ratio=0.8)
    best_result = {
        'accuracy': 0,
        'micro_f1': 0,
        'macro_f1': 0,
        'accuracy_val': 0,
        'micro_f1_val': 0,
        'macro_f1_val': 0
    }
    for decay in [0.0, 0.001, 0.005]:
        result = LREvaluator(weight_decay=decay)(z, y, split)
        if result['accuracy_val'] > best_result['accuracy_val']:
            best_result = result
    return best_result



if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')
    config_path = '/home/shaoqi/code/HAGCL_test/configs/train_node.yaml'
    config = model_utils.load_config(config_path)
    args = model_utils.dict_to_namespace(config)
    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.time = datetime.datetime.now().isoformat()
    if args.dataset == 'Citeseer':
        args.lamda2 = 0.1
    dataset = Planetoid_aug(args.data_path + args.dataset, name=args.dataset)
    dataset_eval = Planetoid_aug(args.data_path + args.dataset, name=args.dataset, aug='none')
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

