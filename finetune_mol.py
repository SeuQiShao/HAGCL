import datetime
from utils.data_loader_mol import MoleculeDataset
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import model_utils, model_loader     
from tqdm import tqdm
import numpy as np
from utils import logger
from sklearn.metrics import roc_auc_score


criterion = nn.BCEWithLogitsLoss(reduction="none")

def train(model, device, loader, optimizer):
    model.train()

    for step, batch in enumerate(loader):
        batch = batch.to(device)
        pred = model(batch)
        y = batch.y.view(pred.shape).to(torch.float64)

        #Whether y is non-null or not.
        is_valid = y**2 > 0
        #Loss matrix
        loss_mat = criterion(pred.double(), (y+1)/2)
        #loss matrix after removing null target
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
            
        optimizer.zero_grad()
        loss = torch.sum(loss_mat)/torch.sum(is_valid)
        loss.backward()

        optimizer.step()


def eval(model, device, loader):
    model.eval()
    y_true = []
    y_scores = []

    for step, batch in enumerate(loader):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)

        y_true.append(batch.y.view(pred.shape))
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim = 0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim = 0).cpu().numpy()

    roc_list = []
    for i in range(y_true.shape[1]):
        #AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == -1) > 0:
            is_valid = y_true[:,i]**2 > 0
            roc_list.append(roc_auc_score((y_true[is_valid,i] + 1)/2, y_scores[is_valid,i]))

    if len(roc_list) < y_true.shape[1]:
        print("Some target is missing!")
        print("Missing ratio: %f" %(1 - float(len(roc_list))/y_true.shape[1]))

    return sum(roc_list)/len(roc_list) #y_true.shape[1]


def main(args, model, dataset, epoch_num):
    train_loader, val_loader, test_loader = model_utils.get_mol_dataloader(args, dataset)

    train_acc_list = []
    val_acc_list = []
    test_acc_list = []

    for epoch in range(1, epoch_num+1):
        print("====epoch " + str(epoch))
        
        train(model, args.device, train_loader, optimizer)

        print("====Evaluation")
        if args.eval_train:
            train_acc = eval(model, args.device, train_loader)
        else:
            print("omit the training accuracy computation")
            train_acc = 0
        val_acc = eval(model, args.device, val_loader)
        test_acc = eval(model, args.device, test_loader)

        val_acc_list.append(val_acc)
        test_acc_list.append(test_acc)
        train_acc_list.append(train_acc)

        logs.write_to_log_file("train: %.4f val: %.4f test: %.4f" %
              (train_acc, val_acc, test_acc))

        print("")
    logs.finetune_result(args, test_acc_list, val_acc_list)

if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')
    config_path = '/home/shaoqi/code/HAGCL/configs/pretrain_mol.yaml'
    config = model_utils.load_config(config_path)
    args = model_utils.dict_to_namespace(config['finetune'])
    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.time = datetime.datetime.now().isoformat()
    logs = logger.Logger(args)
    if args.dataset == "tox21":#75.20-76.7
        args.num_tasks = 12
        epoch_num = 100
    elif args.dataset == "hiv":#77.90-78.47
        args.num_tasks = 1
        epoch_num = 100
    elif args.dataset == "pcba":#
        args.num_tasks = 128
        epoch_num = 100
    elif args.dataset == "muv":#76.66
        args.num_tasks = 17
        epoch_num = 100
    elif args.dataset == "bace":#76.03-82.6
        args.num_tasks = 1
        epoch_num = 100
    elif args.dataset == "bbbp":#71.42
        args.num_tasks = 1
        epoch_num = 100
    elif args.dataset == "toxcast":#63.33-64.2
        args.num_tasks = 617
        epoch_num = 100
    elif args.dataset == "sider":#61.38
        args.num_tasks = 27
        epoch_num = 300
    elif args.dataset == "clintox":#83.38
        args.num_tasks = 2
        epoch_num = 500
    else:
        raise ValueError("Invalid dataset name.")

    dataset = MoleculeDataset(args.data_path + args.dataset, dataset=args.dataset)

    print(args.dataset)



    #set up model
    model, optimizer, scheduler = model_loader.load_model(args)
    if not args.input_model_file == "":
        model.from_pretrained(args.input_model_file, args.device, args.model_epo)
    model.to(args.device)
    main(args, model, dataset, epoch_num)
