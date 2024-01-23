import os
import torch
import torch.optim as optim
from torch.optim import lr_scheduler

from model import basemodel_mol, model_CL, basemodel_tu, basemodel_tu2, basemodel_pd
from model import model_utils



def load_imp(args):

    if args.task == "mol_class_pre" or args.task == "mol_class_fine":
        Imp = basemodel_mol.GNN_imp_estimator(
            args.num_layer,
            args.emb_dim,
            args.JK
        )
    elif args.task == 'tu':
        # Imp = basemodel_tu2.GNN_imp_estimator(
        #     args.num_layer,
        #     args.dataset_num_features,
        #     args.dataset_num_attr,
        #     args.emb_dim
        # )
        Imp = basemodel_tu.Explainer(
            args.dataset_num_features,
            args.emb_dim,
            args.num_layer,
        )
    elif args.task == 'pd':
        # Imp = basemodel_tu2.GNN_imp_estimator(
        #     args.num_layer,
        #     args.dataset_num_features,
        #     args.dataset_num_attr,
        #     args.emb_dim
        # )
        Imp = basemodel_pd.Explainer(
            args.dataset_num_features,
            args.emb_dim,
            args.num_layer,
        )


    if args.load_folder:
        print("Loading model file")
        args.imp_file = os.path.join(args.load_folder, "Imp_{}.pt".format(args.pre_model_epo))
        Imp.load_state_dict(torch.load(args.imp_file, map_location=args.device))

    return Imp


def load_gnn(args):

    if args.task == "mol_class_pre" or args.task == "mol_class_fine":
        gnn = basemodel_mol.HGNN(
            args.num_layer,
            args.emb_dim,
            args.JK,
            args.dropout_ratio,
            args.gnn_type,
            args.add_loop,
            args.headers   
        )

    elif args.task == 'tu':
        gnn = basemodel_tu.HGNN(
            args.num_layer,
            args.emb_dim,
            args.dataset_num_features,
            args.dataset_num_attr,
            args.JK,
            args.dropout_ratio,
            args.gnn_type,
        )
        # gnn = basemodel_tu2.HGNN(
        #     args.num_layer,
        #     args.emb_dim,
        #     args.dataset_num_features,
        #     args.dataset_num_attr,
        #     args.JK,
        #     args.dropout_ratio,
        #     args.gnn_type,
        # )
    elif args.task == 'pd':
        gnn = basemodel_pd.HGNN(
            args.num_layer,
            args.emb_dim,
            args.dataset_num_features,
            args.dataset_num_attr,
            args.JK,
            args.dropout_ratio,
            args.gnn_type,
        )


    if args.load_folder:
        print("Loading model file")
        args.gnn_file = os.path.join(args.load_folder, "gnn_{}.pt".format(args.pre_model_epo))
        gnn.load_state_dict(torch.load(args.gnn_file, map_location=args.device))

    return gnn




def load_model(args):
    Imp = load_imp(args)  
    gnn = load_gnn(args)
    if args.task == 'mol_class_pre' or args.task == 'tu' or args.task == 'pd':
        model = model_CL.graphcl(args, gnn, Imp)
        optimizer = optim.Adam(
            list(model.parameters()),
            lr=args.lr
        )
        scheduler = lr_scheduler.StepLR(
            optimizer,
            step_size=args.lr_decay
            )
    elif args.task == 'mol_class_fine':
        model = model_CL.HGNN_graphpred(args, Imp, gnn)
        model_param_group = []
        model_param_group.append({"params": model.gnn.parameters()})
        model_param_group.append({"params": model.node_imp_estimator.parameters()})
        model_param_group.append({"params": model.graph_pred_linear.parameters(), "lr": args.lr*args.lr_scale})
        optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.lr_decay)
        print(optimizer)
        scheduler = lr_scheduler.StepLR(
            optimizer,
            step_size=args.lr_decay
            )



    return (
        model,
        optimizer,
        scheduler
    )
