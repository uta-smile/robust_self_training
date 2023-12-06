import argparse
import time

from torch_geometric.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, MultiStepLR
from loader import MoleculeDataset

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

# import losses
from losses import MeanAbsoluteError, GeneralizedCrossEntropy, JensenShannonDivergenceWeightedScaled

import time
import copy
import csv
import pickle

# criterion = nn.BCEWithLogitsLoss(reduction = "none")
# criterion = nn.CrossEntropyLoss(size_average=False)
criterion_teacher = nn.CrossEntropyLoss()
criterion_eval = nn.Softmax(dim=1)
# criterion = losses.get_criterion(2, args)

def get_criterion(num_classes, args):
    alpha = args.loss_alpha
    beta = args.loss_beta
    # print('alpha', alpha)
    # print('beta', beta)
    loss_options = {
        # 'SCE': SCELoss(alpha=alpha, beta=beta, num_classes=num_classes),
        'CE': torch.nn.CrossEntropyLoss(),
        # 'MAE': MeanAbsoluteError(scale=alpha, num_classes=num_classes),
        'GCE': GeneralizedCrossEntropy(num_classes=num_classes, q=args.q),
        # 'NCE+RCE': NCEandRCE(alpha=alpha, beta=beta, num_classes=num_classes),
        # 'JSDissect': JSDissect(num_classes, args.js_weights, args.dissect_js),
        # 'LS': LabelSmoothing(num_classes=num_classes, t=alpha),
        # 'JSWC': JensenShannonDivergenceWeightedCustom(num_classes=num_classes,weights=args.js_weights),
        'JSWCS': JensenShannonDivergenceWeightedScaled(num_classes=num_classes,weights=args.js_weights) # '[0.7 0.3]'
        # 'JSNoConsistency': JensenShannonNoConsistency(num_classes=num_classes,weights=args.js_weights),
        # 'bootstrap': Bootstrapping(num_classes=num_classes, t=alpha)
    }

    if args.loss in loss_options:
        print('args.loss', args.loss)
        criterion = loss_options[args.loss]
        return criterion

class JensenShannonDivergenceWeightedScaled(torch.nn.Module):
    def __init__(self, num_classes, weights):
        super(JensenShannonDivergenceWeightedScaled, self).__init__()
        self.num_classes = num_classes
        self.weights = [float(w) for w in weights.split(',')]
        
        self.scale = -1.0 / ((1.0-self.weights[0]) * np.log((1.0-self.weights[0])))
        assert abs(1.0 - sum(self.weights)) < 0.001
    
    def forward(self, pred, labels):
        # print('pred', pred.shape) # torch.Size([128, 10])
        # print('labels', labels.shape) # torch.Size([128])
        preds = list()
        if type(pred) == list:
            for i, p in enumerate(pred):
                preds.append(F.softmax(p, dim=1)) 
        else:
            preds.append(F.softmax(pred, dim=1))

        labels = F.one_hot(labels, self.num_classes).float()
        # print('------------------labels', labels.shape)
        distribs = [labels] + preds
        # print('------------------distribs', distribs.shape)
        assert len(self.weights) == len(distribs)

        mean_distrib = sum([w*d for w,d in zip(self.weights, distribs)])
        mean_distrib_log = mean_distrib.clamp(1e-7, 1.0).log()
        
        jsw = sum([w*custom_kl_div(mean_distrib_log, d) for w,d in zip(self.weights, distribs)])
        # print('mean_distrib_log', mean_distrib_log)
        # print('self.scale', self.scale)
        # print('jsw', jsw)

        # print(self.scale * jsw)
        # input('debug')
        return self.scale * jsw

class GeneralizedCrossEntropy(torch.nn.Module):
    def __init__(self, num_classes, q=0.7):
        super(GeneralizedCrossEntropy, self).__init__()
        self.num_classes = num_classes
        self.q = q
        # print("q:", q, ", type:", type(q))

    def forward(self, device, pred, labels):
        # print('pred', pred)
        # print('pred', pred.shape)
        # print('labels', labels)
        # print('labels', labels.shape)
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        # print('pred', pred)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(device)
        # print('label_one_hot', label_one_hot)
        # print('label_one_hot', label_one_hot.shape)

        gce = (1. - torch.pow(torch.sum(label_one_hot * pred, dim=1), self.q)) / self.q
        # print('gce', gce)
        # print('gce.mean()', gce.mean())
        # input()
        return gce.mean()

# class MeanAbsoluteError(torch.nn.Module):
#     def __init__(self, num_classes, scale=1.0):
#         super(MeanAbsoluteError, self).__init__()
#         self.device = device
#         self.num_classes = num_classes
#         self.scale = scale
#         return

#     def forward(self, pred, labels):
#         pred = F.softmax(pred, dim=1)
#         label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
#         mae = 1. - torch.sum(label_one_hot * pred, dim=1)
#         return self.scale * mae.mean()


def gjs_ce_loss(args, output, target, loss_criterion):
    # print('output', output)
    # print('output', output.shape)

    ce_loss = criterion_teacher(output, target)
    ce_loss = ce_loss / target.size(0)

    # criterion_gjs = losses.get_criterion(2, args)
    gjs_loss = loss_criterion(output, target)
    gjs_loss = gjs_loss / target.size(0)

    loss = ce_loss + args.alpha * gjs_loss

    return loss

def dmi_ce_loss(args, device, output, target):
    ce_loss = criterion_teacher(output, target)
    ce_loss = ce_loss / target.size(0)

    # print('000output', output)
    # output = F.sigmoid(output)
    # print('output', output)

    outputs = F.softmax(output, dim=1)
    # print('outputs', outputs)
    targets = target.reshape(target.size(0), 1).cpu()
    y_onehot = torch.FloatTensor(target.size(0), 2).zero_()

    y_onehot.scatter_(1, targets, 1)

    y_onehot = y_onehot.transpose(0, 1).to(torch.float64).to(device)
    mat = y_onehot @ outputs
    mat = mat / target.size(0)


    det = torch.det(mat.float())

    if det < 0:
        dmi_loss = torch.log(torch.abs(det) + 0.001)

    else:
        dmi_loss = -torch.log(torch.abs(det) + 0.001)
    # print('dmi_loss', dmi_loss)
    # input()

    loss = ce_loss + args.alpha * dmi_loss

    return loss

def train(args, model, device, loader, optimizer, writer, loss_criterion, status='teacher'):
    model.train()

    loss_sum = 0
    if status == 'teacher':
        for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        # for step, batch in enumerate(loader):
            batch = batch.to(device)
            # pred, pred_robust = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

            y = batch.y.view(batch.y.shape).to(torch.long)

            #Whether y is non-null or not.
            is_valid = y**2 > 0

            loss_mat = criterion_teacher(pred.double(), (y+1)/2)

            #loss matrix after removing null target
            loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))

            optimizer.zero_grad()
            loss = torch.sum(loss_mat)/torch.sum(is_valid)
            loss.backward()

            optimizer.step()

            loss_sum += loss
    elif status == 'student':
        # criterion = losses.get_criterion(2, args)

        for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        # for step, batch in enumerate(loader):
            batch = batch.to(device)
            # pred, pred_robust = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            # print('pred', pred)
            # print('pred', pred.shape)

            y = batch.y.view(batch.y.shape).to(torch.long)

            #Whether y is non-null or not.
            if args.gjs_CE_loss and args.loss == 'JSWCS':
                # print('---------------------using gjs_CE_loss')
                loss = gjs_ce_loss(args, pred.double(), (y+1)/2, loss_criterion)
            elif args.DMI_CE_loss:
                # print('---------------------using DMI_CE_loss')
                loss = dmi_ce_loss(args, device, pred.double(), (y+1)/2)
            elif args.loss == 'GCE':
                # print('---------------------using GCE')
                loss = loss_criterion(device, pred.double(), (y+1)/2)
            # elif args.loss == 'JSWCS':
            #     loss = criterion(pred.double(), (y+1)/2)
            else:
                # print('----------------run baseline without robust loss added')
                # print('input robust loss function, either --gjs_CE_loss 1, or --DMI_CE_loss 1, or --loss GCE, or --loss JSWCS')
                is_valid = y**2 > 0
                loss_mat = criterion_teacher(pred.double(), (y+1)/2)
                loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
                loss = torch.sum(loss_mat)/torch.sum(is_valid)

            optimizer.zero_grad()

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
    hard_preds = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        batch_smiles = batch.smiles

        with torch.no_grad():
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            pred_probs = criterion_eval(pred)

        a, hard_pred = pred_probs.max(1)
        hard_preds.append(hard_pred.reshape(hard_pred.size(0),1))

        y_score = pred_probs[:,1]

        y_true_reshape = batch.y.reshape(batch.y.size(0),1)

        y_true.append(y_true_reshape)

        y_scores.append(y_score.reshape(y_score.size(0),1))
        # smiles.append(batch.smiles)
        for i in batch_smiles:
            smiles.append(i)

    y_true = torch.cat(y_true, dim = 0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim = 0).cpu().numpy()

    hard_preds = torch.cat(hard_preds, dim = 0).cpu().numpy()

    roc_list = []
    for i in range(y_true.shape[1]):
        #AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == -1) > 0:
            is_valid = y_true[:,i]**2 > 0
            # print('(y_true[is_valid,i] + 1)/2', (y_true[is_valid,i] + 1)/2)
            # print('y_scores[is_valid,i]', y_scores[is_valid,i])
            # input()
            roc_list.append(roc_auc_score((y_true[is_valid,i] + 1)/2, y_scores[is_valid,i]))

    if len(roc_list) < y_true.shape[1]:
        print("Some target is missing!")
        print("Missing ratio: %f" %(1 - float(len(roc_list))/y_true.shape[1]))

    pred_value = [i for k in hard_preds for i in k]
    # print('pred_value', pred_value)
    # input()
    # hard_preds = [1 if p > 0.5 else 0 for p in y_preds]
    smiles_preds = list(zip(smiles, pred_value))
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
    parser.add_argument('--init_student', type=int, default = 0, help='initial each student train or not')
    parser.add_argument('--s_epochs', type=int, default=50,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr_decay', type=float, default=1,
                        help='weight decay (default: 0)')
    parser.add_argument('--load_teacher', type=int, default = 0, help='initial 1st student with teacher model or not')

    # gjs loss
    parser.add_argument('--loss', choices=['JSDissect', 'CE', 'NCE+RCE', 'SCE', 'LS', 'MAE', 'GCE', 'JSWC', 'JSWCS', 'JSNoConsistency', 'bootstrap'])
    parser.add_argument('--loss_alpha', default=1.0, type=float)
    parser.add_argument('--loss_beta', default=1.0, type=float)
    parser.add_argument('--q', default=0.2, type=float)
    parser.add_argument('--js_weights', help='First weight is for label, the next are in the order of "augs"', type=str, default='0.4,0.6')
    # JS specific
    parser.add_argument('--dissect_js', type=str, choices=['a','b','c','d','e','f', 'as','bs','cs','ds','es','fs', 'g','h', 'i']) # Defaults to None

    # Evaluation
    parser.add_argument('--eval_consistency', help='Check consistency of noisy vs noise free training examples.', action='store_true')
    parser.add_argument('--alpha', default=1.0, type=float)

    parser.add_argument('--load_s_best', type=int, default = 0, help='load best student from last iteration or not')
    parser.add_argument('--train_teacher', type=int, default = 0, help='train from the teacher model')
    parser.add_argument(
        '--teacher_model_file',
        type=str,
        default=
        '/mnt/ssd4/hm/saved_model/pre-gin/output/load_teacher/runs0/teacher_model_epo150_1/2022_01_02_14_55_22/model.pt',
        help='filename to load the teacher (if there is any)')
    parser.add_argument('--DMI_CE_loss', type=int, default = 0, help='add DMI_CE_loss or not')
    parser.add_argument('--gjs_CE_loss', type=int, default = 0, help='add gjs_CE_loss or not')

    args = parser.parse_args()
    print('args:', args)

    loss_criterion = get_criterion(2, args)

    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    print('--------running on device', device)
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
    if not args.train_teacher:
        if args.teacher_model_file == '':
            print('Please input teacher model path.')
        else:
            print('load teacher model from ...', args.teacher_model_file)
            model.load_state_dict(torch.load(args.teacher_model_file), strict=False)
            # model.to(device)

    model.to(device)

    #set up optimizer
    model_param_group = []
    model_param_group.append({"params": model.gnn.parameters()})
    if args.graph_pooling == "attention":
        model_param_group.append({"params": model.pool.parameters(), "lr":args.lr*args.lr_scale})
    model_param_group.append({"params": model.graph_pred_linear.parameters(), "lr":args.lr*args.lr_scale})
    # print('model_param_group', model_param_group[0][0])
    # input()
    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
    # print("teacher optimizer", optimizer)

    train_acc_list = []
    val_acc_list = []
    test_acc_list = []

    iter_dict = {}
    pseudo_dataset_dict = {}
    csv_pseudo_dataset_dict = {}

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

    if args.train_teacher:
        best_val_score = -float('inf')
        test_score = -float('inf')
        best_epoch = 0

        for epoch in range(1, args.epochs+1):
            print("====epoch " + str(epoch))

            train_loss_epoch = train(args, model, device, train_loader, optimizer, writer, loss_criterion, status='teacher')

            if epoch == 1:
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


            if not args.filename == "":
                writer.add_scalar('data/train auc', train_acc, epoch)
                writer.add_scalar('data/val auc', val_acc, epoch)
                writer.add_scalar('data/test auc', test_acc, epoch)
                writer.add_scalar('data/train loss', train_loss_epoch, epoch)

            print("")

        iter_dict[0] = {'student_iteration': 0, 'best_epoch': best_epoch, 'best_val': best_val_score, 'best_test': test_score, 'model': copy.deepcopy(best_model)}

        print('final test of teacher model is {}, seed: {} data: {} on epoch {}'.format(test_score, args.seed, args.dataset, best_epoch))
        best_model_tmp = copy.deepcopy(best_model)
    else:
        best_model_tmp = copy.deepcopy(model)

    if args.self_training:
        # train_acc = eval(args, model, device, unlabeled_loader, time_start, status='pseudo_0')
        train_acc = eval(args, best_model_tmp, device, unlabeled_loader, time_start, status='pseudo_0')

        pseudo_dataset_dir_tmp = os.path.join(args.output_pseudo_dataset, args.dataset, time_start, 'iter0')
        csv_pseudo_dataset_dict[0] = os.path.join(pseudo_dataset_dir_tmp, 'raw', 'iter0_preds.csv')
        # pseudo_dataset_0 = os.path.join("pseudo_dataset/", args.dataset, 'iter0', 'raw', 'iter0_preds.csv')
        print('pseudo_dataset_0', csv_pseudo_dataset_dict[0])
        # input()
        with open(csv_pseudo_dataset_dict[0], 'a+') as f:
            csv_writer = csv.writer(f)
            for i in train_smiles_label:
                csv_writer.writerow(i)
        pseudo_dataset_dict[0] = MoleculeDataset(pseudo_dataset_dir_tmp, dataset=args.dataset)
        # pseudo_dataset_0 = MoleculeDataset(pseudo_dataset_0, dataset=args.dataset)
        print('original teacher pseudo dataset', pseudo_dataset_dict[0])

        if not args.use_unlabeled:
            print('Not using unlabeled data.')
        else:
            for ite in range(1, args.st_iter+1):
                print('=========================running student iteration ', ite)
                pseudo_dataset = DataLoader(pseudo_dataset_dict[ite-1], batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
                # pseudo_dataset = pseudo_dataset_dict[ite-1]

                ite_best_val_score = -float('inf')
                ite_test_score = -float('inf')
                ite_best_epoch = 0

                if ite == 1:
                    if not args.load_teacher:
                        print('---------re-initialize student model.')
                        model = GNN_graphpred_robust(args.num_layer, args.emb_dim, num_tasks, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling, gnn_type = args.gnn_type)
                        model.to(device)
                    else:
                        model = copy.deepcopy(best_model_tmp)
                        # model.to(device)

                    model_param_group = []
                    model_param_group.append({"params": model.gnn.parameters()})
                    if args.graph_pooling == "attention":
                        model_param_group.append({"params": model.pool.parameters(), "lr":args.lr*args.lr_scale})
                    model_param_group.append({"params": model.graph_pred_linear.parameters(), "lr":args.lr*args.lr_scale})
                    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
                    print("initial student optimizer", optimizer)
                    m_list = []
                    for i in range(args.st_iter):
                        if i != 0:
                            m = i * args.s_epochs + 1
                            m_list.append(m)
                    print('m_list', m_list)
                    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = m_list, gamma=args.lr_decay)

                # else:
                #     model = copy.deepcopy(iter_dict[ite-1]['model'])
                else:
                    if args.load_s_best:
                        print('-----------------load best model from previous iteration')
                        model = copy.deepcopy(iter_dict[ite-1]['model'])
                        model_param_group = []
                        model_param_group.append({"params": model.gnn.parameters()})
                        if args.graph_pooling == "attention":
                            model_param_group.append({"params": model.pool.parameters(), "lr":args.lr*args.lr_scale})
                        model_param_group.append({"params": model.graph_pred_linear.parameters(), "lr":args.lr*args.lr_scale})
                        optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
                    else:
                        print('-----------------keep training from the last epoch of previous iteration')

                for epoch in range(1, args.s_epochs+1):
                    if args.train_teacher:
                        enumerate_epoch = epoch + args.epochs + (ite-1) * args.s_epochs
                    else:
                        enumerate_epoch = epoch + (ite-1) * args.s_epochs

                    # print("====epoch " + str(epoch + ite * args.epochs))
                    print('====epoch', enumerate_epoch)

                    train_loss_epoch = train(args, model, device, pseudo_dataset, optimizer, writer, loss_criterion, status='student')

                    # if epoch == args.epochs+1:
                    #     best_model = copy.deepcopy(best_model)

                    print("====Evaluation")
                    if args.eval_train:
                        train_acc = eval(args, model, device, pseudo_dataset, time_start, status='eval')
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

                    if val_acc >= ite_best_val_score:
                        ite_best_epoch, ite_best_val_score, ite_test_score = enumerate_epoch, val_acc, test_acc
                        best_model_tmp = copy.deepcopy(model)


                    if not args.filename == "":
                        writer.add_scalar('data/train auc', train_acc, enumerate_epoch)
                        writer.add_scalar('data/val auc', val_acc, enumerate_epoch)
                        writer.add_scalar('data/test auc', test_acc, enumerate_epoch)
                        writer.add_scalar('data/train loss', train_loss_epoch, enumerate_epoch)
                        writer.add_scalar('data/lr', lr_scheduler.get_last_lr()[0], enumerate_epoch)
                    lr_scheduler.step()
                    print("")

                iter_dict[ite] = {
                    'student_iteration': ite,
                    'best_epoch': ite_best_epoch,
                    'best_val': ite_best_val_score,
                    'best_test': ite_test_score,
                    'model': copy.deepcopy(best_model_tmp)
                }
                print('--------student iteration {}: best val {} at epoch {} with test {}'.format(ite, ite_best_val_score, ite_best_epoch, ite_test_score))

                # if args.init_student:
                #     if check_best_model(iter_dict[ite]['best_test'], ite, iter_dict):
                #         best_model = copy.deepcopy(best_model_tmp)
                # else:
                best_model = copy.deepcopy(best_model_tmp)

                train_acc = eval(args, best_model, device, unlabeled_loader, time_start, status='pseudo_{}'.format(ite))
                stu_pseudo_dataset_dir_tmp = os.path.join(args.output_pseudo_dataset, args.dataset, time_start, 'iter{}'.format(ite))

                csv_pseudo_dataset_dict[ite] = os.path.join(stu_pseudo_dataset_dir_tmp, 'raw',
                                                            'iter{}_preds.csv'.format(ite))

                print('csv_pseudo_dataset_{} save at'.format(ite), csv_pseudo_dataset_dict[ite])
                with open(csv_pseudo_dataset_dict[ite], 'a+') as f:
                    csv_writer = csv.writer(f)
                    for i in train_smiles_label:
                        csv_writer.writerow(i)

                pseudo_dataset_dict[ite] = MoleculeDataset(stu_pseudo_dataset_dir_tmp, dataset=args.dataset)
                print('student-teacher pseudo dataset generated from iteration {}'.format(ite), pseudo_dataset_dict[ite])

            # print('?? final test result is {}, seed: {} data: {} on epoch {}'.format(test_score, args.seed, args.dataset, best_epoch))
    print('iter_dict', iter_dict)

    tmp_all_test = []
    for key in iter_dict:
        tmp_all_test.append(iter_dict[key]['best_test'])
    for key in iter_dict:
        if iter_dict[key]['best_test'] == max(tmp_all_test):
            print('?? final test result is {}, seed: {} data: {} on epoch {}'.format(iter_dict[key]['best_test'], args.seed, args.dataset, iter_dict[key]['best_epoch']))

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
