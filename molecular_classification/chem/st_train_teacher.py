import argparse
import time

from loader import MoleculeDataset
from torch_geometric.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, MultiStepLR


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from model import GNN, GNN_graphpred, GNN_graphpred_robust
from sklearn.metrics import roc_auc_score

from splitters import scaffold_split, random_split, random_split_for_self_training
import pandas as pd

import os
import shutil

from tensorboardX import SummaryWriter
 
import time
import copy
import csv
import pickle

# criterion = nn.BCEWithLogitsLoss(reduction = "none")
criterion = nn.CrossEntropyLoss()
criterion_eval = nn.Softmax(dim=1)


def train(args, model, device, loader, optimizer, writer):
    model.train()

    loss_sum = 0
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
    # for step, batch in enumerate(loader):
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        # y = batch.y.view(pred.shape).to(torch.float64)
        y = batch.y.view(batch.y.shape).to(torch.long)

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

        loss_sum += loss
    loss_avg = loss_sum/step

    return loss_avg

def eval(args, model, device, loader, time_start, status='pseudo'):
    model.eval()
    y_true = []
    y_scores = []
    smiles = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        batch_smiles = batch.smiles

        with torch.no_grad():
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            pred_probs = criterion_eval(pred)
        # print('batch.y', batch.y) # tensor([-1, -1, -1, -1, -1, -1, -1, -1], device='cuda:1')
        # print('pred', pred.shape) # torch.Size([8, 1])
        y_true_reshape = batch.y.reshape(batch.y.size(0),1)
        y_true.append(y_true_reshape)
        # y_true.append(batch.y.view(pred.shape))
        # print('y_true', y_true)
        # print('y_true', y_true.shape)
        # input()

        y_score = pred_probs[:,1]
        y_scores.append(y_score.reshape(y_score.size(0),1))
        # y_scores.append(pred)
        # smiles.append(batch.smiles)
        for i in batch_smiles:
            smiles.append(i)

    y_true = torch.cat(y_true, dim = 0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim = 0).cpu().numpy()
    # print('y_true', len(y_true))
    # print('y_scores', len(y_scores))
    # print('y_true.shape[1]', y_true.shape)
    # print('y_true.shape[1]', y_true.shape[1])
    # input()

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
    hard_preds = [1 if p > 0.5 else 0 for p in y_preds]
    smiles_preds = list(zip(smiles, hard_preds))
    if args.self_training:
        if 'pseudo' in status:
            current_iter = status.split('_')[1]
            if args.output_pseudo_dataset:
                output_dir_tmp = os.path.join(args.output_pseudo_dataset, args.dataset, time_start, 'iter{}'.format(current_iter), 'raw/')
                # print('output_dir_tmp', output_dir_tmp)
                if not os.path.exists(output_dir_tmp):
                    os.makedirs(output_dir_tmp)
            print('saving to pseudo dataset to...', os.path.join(output_dir_tmp + 'iter{}_preds.csv'.format(current_iter)))
            with open(os.path.join(output_dir_tmp + 'iter{}_preds.csv'.format(current_iter)), 'w') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(['smiles', 'HIV_active'])
                for i in smiles_preds:
                    csv_writer.writerow(i)
        elif status == 'eval':
            print('')

    return sum(roc_list)/len(roc_list) #y_true.shape[1]

def check_best_model(best_test, ite, tmp_dict):
    for key in tmp_dict:
        if key != ite:
            if best_test <= tmp_dict[key]['best_test']:
                return False
    return True


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
    # parser.add_argument('--teacher_epochs', type=int, default=100,
    #                     help='number of epochs to train teacher model(default: 100)')
    parser.add_argument('--st_iter', type=int, default=3, metavar='N',
                    help='self training iteration number')
    parser.add_argument('--output_pseudo_dataset', type=str, default = '', help='filename to save the generated files (if there is any)')
    parser.add_argument('--init_student', type=int, default = 0, help='evaluating training or not')
    parser.add_argument('--s_epochs', type=int, default=50,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr_decay', type=float, default=1,
                        help='weight decay (default: 0)')
    parser.add_argument('--load_teacher', type=int, default = 0, help='initial 1st student with teacher model or not')

    parser.add_argument('--load_s_best', type=int, default = 0, help='load best student from last iteration or not')

    args = parser.parse_args()
    print('args:', args)


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

    time_start = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())

    #set up dataset
    dataset = MoleculeDataset("dataset/" + args.dataset, dataset=args.dataset)

    print('dataset', dataset)
    
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
            # if args.use_unlabeled:
            #     train_dataset, valid_dataset, test_dataset, unlabeled_dataset = random_split_for_self_training(
            #         dataset,
            #         null_value=0,
            #         frac_train=0.9,
            #         frac_valid=0.05,
            #         frac_test=0.05,
            #         frac_unlabeled=0.0,
            #         seed=args.seed)
            # else:
            train_dataset, valid_dataset, test_dataset, unlabeled_dataset, train_idx = random_split_for_self_training(
                dataset,
                null_value=0,
                frac_train=0.3,
                frac_valid=0.1,
                frac_test=0.1,
                frac_unlabeled=0.5,
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

    # train_smiles = train_dataset.data.smiles[train_idx]
    train_smiles = [train_dataset.data.smiles[i] for i in train_idx]
    # train_smiles = [train_dataset.smiles_list[i] for i in train_idx]

    train_label_tmp = [train_dataset.data.y.tolist()[i] for i in train_idx]
    # train_label_tmp = train_dataset.data.y.tolist()
    train_label = [int((i+1)/2) for i in train_label_tmp]
    train_smiles_label = list(zip(train_smiles, train_label))

    # print('train_smiles', train_smiles)
    # print('train_label', train_label)
    # print('train_smiles', len(train_smiles))
    # print('train_label', len(train_label))
    # input()

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

    if args.self_training:
        unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

    #set up model
    model = GNN_graphpred_robust(args.num_layer, args.emb_dim, num_tasks, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling, gnn_type = args.gnn_type)
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
    print("teacher optimizer", optimizer)

    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = args.s_epochs, gamma=args.lr_decay)
    # m_list = []
    # for i in range(args.st_iter):
    #     m = i * args.s_epochs
    #     m_list.append(m)
    # # for i in range(args.st_iter):
    # #     if i == 0:
    # #         m_list.append(args.epochs)
    # #     else:
    # #         m = args.epochs + i * args.s_epochs
    # #         m_list.append(m)
    # print('m_list', m_list)
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = m_list, gamma=args.lr_decay)
    # print('lr_scheduler', lr_scheduler)
    # input()

    # print('current lr-------------------', lr_scheduler.get_last_lr())


    train_acc_list = []
    val_acc_list = []
    test_acc_list = []

    best_val_score = -float('inf')
    test_score = -float('inf')
    best_epoch = 0

    iter_dict = {}
    # n_iter=0
    # best_model_iter_dict = {}
    # res = {'student_iteration': 0, 'best_epoch': 0, 'best_val': 1e10, 'best_test': 1e10}
    # res[iter]['model']
    # res = {'s_iteration': []}

    if not args.filename == "":
        # fname = 'runs/finetune_cls_runseed' + str(args.runseed) + '/' + args.filename
        ### save path on pangu
        # fname = '/mnt/ssd4/hm/saved_model/pre-gin/runs/finetune_cls_runseed' + str(args.runseed) + '/' + args.filename + '/' + time_start
        fname = str(args.output_pseudo_dataset) + 'runs' + str(args.runseed) + '/' + args.filename + '/' + time_start

        #delete the directory if there exists one
        if os.path.exists(fname):
            shutil.rmtree(fname)
            print("removed the existing file.")
        writer = SummaryWriter(fname)

    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch))
        # train(args, model, device, train_loader, optimizer)
        # n_iter, train_loss_epoch = train(args, model, device, train_loader, optimizer, n_iter, writer)
        train_loss_epoch = train(args, model, device, train_loader, optimizer, writer)

        if epoch == 0:
            best_model = copy.deepcopy(model)

        print("====Evaluation")
        if args.eval_train:
            train_acc = eval(args, model, device, train_loader, time_start, status='eval')
        else:
            print("omit the training accuracy computation")
            train_acc = 0
        val_acc = eval(args, model, device, val_loader, time_start, status='eval')
        test_acc = eval(args, model, device, test_loader, time_start, status='eval')

        print("train loss: %f" %(train_loss_epoch))
        print("train: %f val: %f test: %f" %(train_acc, val_acc, test_acc))

        val_acc_list.append(val_acc)
        test_acc_list.append(test_acc)
        train_acc_list.append(train_acc)

        if val_acc >= best_val_score:
            best_epoch, best_val_score, test_score = epoch, val_acc, test_acc
            best_model = copy.deepcopy(model)
            best_opti_state = optimizer.state_dict()


        if not args.filename == "":
            writer.add_scalar('data/train auc', train_acc, epoch)
            writer.add_scalar('data/val auc', val_acc, epoch)
            writer.add_scalar('data/test auc', test_acc, epoch)
            writer.add_scalar('data/train loss', train_loss_epoch, epoch)

        print("")

    torch.save(best_model.state_dict(), os.path.join(fname, 'model.pt'))

    iter_dict[0] = {
        'student_iteration': 0,
        'best_epoch': best_epoch,
        'best_val': best_val_score,
        'best_test': test_score,
        'model': copy.deepcopy(best_model),
        'optimizer': best_opti_state
    }

    print('final test result is {}, seed: {} data: {} on epoch {}'.format(test_score, args.seed, args.dataset, best_epoch))
    # print('finish training teacher model at: ', time_end)

    total_score = {'train score': train_acc_list, 'val score': val_acc_list, 'test score': test_acc_list}
    # print('total_score', total_score)
    with open(os.path.join(fname, 'scores.txt'), 'wb') as f:
        pickle.dump(total_score, f)

    print('start at: ', time_start)
    time_end = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    print('finish at: ', time_end)

    if not args.filename == "":
        writer.close()

if __name__ == "__main__":
    main()
