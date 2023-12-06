import argparse
import time

from loader import MoleculeDataset
from torch_geometric.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from model import GNN, GNN_graphpred
from sklearn.metrics import roc_auc_score

from splitters import scaffold_split, random_split, random_split_for_self_training
import pandas as pd

import os 
import shutil

from tensorboardX import SummaryWriter

import time
import copy
import csv

criterion = nn.BCEWithLogitsLoss(reduction = "none")

# def train(args, model, device, loader, optimizer):
#     model.train()

#     for step, batch in enumerate(tqdm(loader, desc="Iteration")):
#         batch = batch.to(device)
#         pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
#         y = batch.y.view(pred.shape).to(torch.float64)

#         #Whether y is non-null or not.
#         is_valid = y**2 > 0
#         #Loss matrix
#         loss_mat = criterion(pred.double(), (y+1)/2)
#         #loss matrix after removing null target
#         loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
            
#         optimizer.zero_grad()
#         loss = torch.sum(loss_mat)/torch.sum(is_valid)
#         loss.backward()

#         optimizer.step()

def train(args, model, device, loader, optimizer, n_iter, writer):
    model.train()

    loss_sum = 0
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
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

        n_iter += 1
        loss_sum += loss
        # writer.add_scalar('data/train loss', loss, n_iter)
    loss_avg = loss_sum/step

    return n_iter, loss_avg



def eval(args, model, device, loader):
    model.eval()
    y_true = []
    y_scores = []
    smiles = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        # print('batch', batch)

        batch = batch.to(device)
        batch_smiles = batch.smiles
        # print('batch_smiles', batch_smiles)
        # # input()
        # for i in batch_smiles:
        #     print('i', i)
        #     input()

        with torch.no_grad():
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        y_true.append(batch.y.view(pred.shape))
        y_scores.append(pred)
        # smiles.append(batch.smiles)
        # smiles.append(i for i in batch_smiles)
        for i in batch_smiles:
            smiles.append(i)
        # print('smiles', smiles)
        # input()

    # print('torch.cat(y_scores, dim = 0)0', torch.cat(y_scores, dim = 0))
    # hard_preds_1 = torch.max(torch.cat(y_scores, dim = 0), 1)[1]
    y_true = torch.cat(y_true, dim = 0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim = 0).cpu().numpy()
    # smiles = torch.cat(torch.tensor(smiles), dim = 0).cpu()

    roc_list = []
    for i in range(y_true.shape[1]):
        #AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == -1) > 0:
            is_valid = y_true[:,i]**2 > 0
            roc_list.append(roc_auc_score((y_true[is_valid,i] + 1)/2, y_scores[is_valid,i]))

    if len(roc_list) < y_true.shape[1]:
        print("Some target is missing!")
        print("Missing ratio: %f" %(1 - float(len(roc_list))/y_true.shape[1]))

    y_preds = [i for k in y_scores for i in k]
    y_label = [i for k in y_true for i in k]
    # print('y_scores', y_scores.shape)
    # print('y_true', y_true.shape)
    # print('smiles', len(smiles))
    # print('y_preds', len(y_preds))
    # print('y_pred', y_preds)
    # print('smiles', smiles)
    # print('y_label', y_label)
    # # input()
    hard_preds = [1 if p > 0.5 else 0 for p in y_preds]

    # print('hard_preds', hard_preds)

    smiles_preds = list(zip(smiles, hard_preds))
    # print('smiles_preds', smiles_preds)

    if args.self_training:
        with open(os.path.join('tmp', args.dataset + '_preds.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['smiles', 'HIV_active'])
            for i in smiles_preds:
                writer.writerow(i)

    return sum(roc_list)/len(roc_list) #y_true.shape[1]



def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--lr_scale', type=float, default=1,
                        help='relative learning rate for the feature extraction layer (default: 1)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--dataset', type=str, default = 'tox21', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--input_model_file', type=str, default = '', help='filename to read the model (if there is any)')
    parser.add_argument('--filename', type=str, default = '', help='output filename')
    parser.add_argument('--seed', type=int, default=42, help = "Seed for splitting the dataset.")
    parser.add_argument('--runseed', type=int, default=0, help = "Seed for minibatch selection, random initialization.")
    parser.add_argument('--split', type = str, default="scaffold", help = "random or scaffold or random_scaffold")
    parser.add_argument('--eval_train', type=int, default = 0, help='evaluating training or not')
    parser.add_argument('--num_workers', type=int, default = 4, help='number of workers for dataset loading')

    # args for self-training
    parser.add_argument('--self_training', type=int, default = 0, help='use self-training or not')
    parser.add_argument('--use_unlabeled', type=int, default = 0, help='use unlabeled data or not')
    parser.add_argument('--teacher_epochs', type=int, default=100,
                        help='number of epochs to train teacher model(default: 100)')

    args = parser.parse_args()


    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)

    #Bunch of classification tasks
    if args.dataset == "tox21":
        num_tasks = 12
    elif args.dataset == "hiv":
        num_tasks = 1
    elif args.dataset == "pcba":
        num_tasks = 128
    elif args.dataset == "muv":
        num_tasks = 17
    elif args.dataset == "bace":
        num_tasks = 1
    elif args.dataset == "bbbp":
        num_tasks = 1
    elif args.dataset == "toxcast":
        num_tasks = 617
    elif args.dataset == "sider":
        num_tasks = 27
    elif args.dataset == "clintox":
        num_tasks = 2
    else:
        raise ValueError("Invalid dataset name.")

    #set up dataset
    dataset = MoleculeDataset("dataset/" + args.dataset, dataset=args.dataset)

    print(dataset)
    
    if args.split == "scaffold":
        smiles_list = pd.read_csv('dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1)
        print("scaffold")

    elif args.split == "random_scaffold":
        smiles_list = pd.read_csv('dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = random_scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = args.seed)
        print("random scaffold")
    elif args.split == "random":
        if args.self_training:
            train_dataset, valid_dataset, test_dataset, unlabeled_dataset, _ = random_split_for_self_training(
                dataset,
                null_value=0,
                frac_train=0.2,
                frac_valid=0.05,
                frac_test=0.05,
                frac_unlabeled=0.7,
                seed=args.seed)
        else:
            train_dataset, valid_dataset, test_dataset = random_split(dataset, null_value=0, frac_train=0.9,frac_valid=0.05, frac_test=0.05, seed = args.seed)
        print("random")
    else:
        raise ValueError("Invalid split option.")

    print(train_dataset[0])
    print('train_dataset', train_dataset)
    print('valid_dataset', valid_dataset)
    print('test_dataset', test_dataset)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

    if args.self_training:
        unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

    #set up model
    model = GNN_graphpred(args.num_layer, args.emb_dim, num_tasks, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling, gnn_type = args.gnn_type)
    if not args.input_model_file == "":
        model.from_pretrained(args.input_model_file)
    
    model.to(device)

    #set up optimizer
    #different learning rate for different part of GNN
    model_param_group = []
    model_param_group.append({"params": model.gnn.parameters()})
    if args.graph_pooling == "attention":
        model_param_group.append({"params": model.pool.parameters(), "lr":args.lr*args.lr_scale})
    model_param_group.append({"params": model.graph_pred_linear.parameters(), "lr":args.lr*args.lr_scale})
    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
    print(optimizer)

    train_acc_list = []
    val_acc_list = []
    test_acc_list = []

    time_start = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())

    best_val_score = -float('inf')
    test_score = -float('inf')
    best_epoch = 0

    n_iter=0
    if not args.filename == "":
        # fname = 'runs/finetune_cls_runseed' + str(args.runseed) + '/' + args.filename
        ### save path on pangu
        fname = '/mnt/ssd4/hm/saved_model/pre-gin/runs/finetune_cls_runseed' + str(args.runseed) + '/' + args.filename + '/' + time_start

        #delete the directory if there exists one
        if os.path.exists(fname):
            shutil.rmtree(fname)
            print("removed the existing file.")
        writer = SummaryWriter(fname)

    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch))
        
        # train(args, model, device, train_loader, optimizer)
        n_iter, train_loss_epoch = train(args, model, device, train_loader, optimizer, n_iter, writer)

        print("====Evaluation")
        if args.eval_train:
            train_acc = eval(args, model, device, train_loader)
        else:
            print("omit the training accuracy computation")
            train_acc = 0
        val_acc = eval(args, model, device, val_loader)
        test_acc = eval(args, model, device, test_loader)

        print("train loss: %f" %(train_loss_epoch))
        print("train: %f val: %f test: %f" %(train_acc, val_acc, test_acc))

        val_acc_list.append(val_acc)
        test_acc_list.append(test_acc)
        train_acc_list.append(train_acc)

        if val_acc >= best_val_score:
            best_epoch, best_val_score, test_score = epoch, val_acc, test_acc


        if not args.filename == "":
            writer.add_scalar('data/train auc', train_acc, epoch)
            writer.add_scalar('data/val auc', val_acc, epoch)
            writer.add_scalar('data/test auc', test_acc, epoch)
            writer.add_scalar('data/train loss', train_loss_epoch, epoch)

        print("")
    print('final test result is {}, seed: {} data: {} on epoch {}'.format(test_score, args.seed, args.dataset, best_epoch))
    time_end = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    print('start at: ', time_start)
    print('finish at: ', time_end)

    if not args.filename == "":
        writer.close()

if __name__ == "__main__":
    main()
