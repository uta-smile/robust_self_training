from qm9 import dataset
from qm9.models import EGNN
import torch
from torch import nn, optim
import argparse
from qm9 import utils as qm9_utils
import utils
import json
import datetime
import copy

import time
import os
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='QM9 Example')
parser.add_argument('--exp_name', type=str, default='exp_1', metavar='N',
                    help='experiment_name')
parser.add_argument('--batch_size', type=int, default=96, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                    help='number of epochs to train (default: 10)')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=20, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--test_interval', type=int, default=1, metavar='N',
                    help='how many epochs to wait before logging test')
parser.add_argument('--outf', type=str, default='qm9/logs', metavar='N',
                    help='folder to output vae')
parser.add_argument('--lr', type=float, default=1e-3, metavar='N',
                    help='learning rate')
parser.add_argument('--nf', type=int, default=128, metavar='N',
                    help='learning rate')
parser.add_argument('--attention', type=int, default=1, metavar='N',
                    help='attention in the ae model')
parser.add_argument('--n_layers', type=int, default=7, metavar='N',
                    help='number of layers for the autoencoder')
parser.add_argument('--property', type=str, default='homo', metavar='N',
                    help='label to predict: alpha | gap | homo | lumo | mu | Cv | G | H | r2 | U | U0 | zpve')
parser.add_argument('--num_workers', type=int, default=0, metavar='N',
                    help='number of workers for the dataloader')
parser.add_argument('--charge_power', type=int, default=2, metavar='N',
                    help='maximum power to take into one-hot features')
parser.add_argument('--dataset_paper', type=str, default="cormorant", metavar='N',
                    help='cormorant, lie_conv')
parser.add_argument('--node_attr', type=int, default=0, metavar='N',
                    help='node_attr or not')
parser.add_argument('--weight_decay', type=float, default=1e-16, metavar='N',
                    help='weight decay')

parser.add_argument('--st_iter', type=int, default=3, metavar='N',
                    help='self training iteration number')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
dtype = torch.float32
print(args)

task_start_at = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())

utils.makedir(args.outf)
# utils.makedir(args.outf + "/" + args.exp_name)
utils.makedir(args.outf + "/" + args.exp_name)

# savedir for tensorboard
fname = str(args.outf) + "/" + "tensorboard" + "/" + str(args.exp_name) + '/' + task_start_at
print(fname)
os.makedirs(fname)

writer = SummaryWriter(fname)


dataloaders, charge_scale = dataset.retrieve_dataloaders(args.batch_size, args.num_workers)
# compute mean and mean absolute deviation
meann, mad = qm9_utils.compute_mean_mad(dataloaders, args.property) # todo
print('=====meann, mad=======')
print(meann, mad)

model = EGNN(in_node_nf=15, in_edge_nf=0, hidden_nf=args.nf, device=device,
                 n_layers=args.n_layers, coords_weight=1.0, attention=args.attention, node_attr=args.node_attr)

print(model)

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
loss_l1 = nn.L1Loss()


def train(epoch, loader, partition='train', best_model=None):

    # if partition == 'strain':
    #     model = best_model
    #     best_model = None

    lr_scheduler.step()
    current_lr = lr_scheduler.get_last_lr()
    # print('current_lr', current_lr)
    # input()
    res = {'loss': 0, 'counter': 0, 'loss_arr':[]}
    idx_pseudo_dict = {}
    tmp = []
    for i, data in enumerate(loader):
        # if partition == 'strain':
        #     print(i)
        if partition == 'train':
            model.train()
            optimizer.zero_grad()
        elif partition == 'strain':
            best_model.eval()
        else:
            model.eval()


        batch_size, n_nodes, _ = data['positions'].size()


        atom_positions = data['positions'].view(batch_size * n_nodes, -1).to(device, dtype)
        atom_mask = data['atom_mask'].view(batch_size * n_nodes, -1).to(device, dtype)
        edge_mask = data['edge_mask'].to(device, dtype)
        one_hot = data['one_hot'].to(device, dtype)
        charges = data['charges'].to(device, dtype)
        nodes = qm9_utils.preprocess_input(one_hot, charges, args.charge_power, charge_scale, device)

        nodes = nodes.view(batch_size * n_nodes, -1)
        # nodes = torch.cat([one_hot, charges], dim=1)
        edges = qm9_utils.get_adj_matrix(n_nodes, batch_size, device)
        label = data[args.property].to(device, dtype)

        if partition == 'strain':
            pred = best_model(h0=nodes, x=atom_positions, edges=edges, edge_attr=None, node_mask=atom_mask, edge_mask=edge_mask,
                     n_nodes=n_nodes)
        else:    
            pred = model(h0=nodes, x=atom_positions, edges=edges, edge_attr=None, node_mask=atom_mask, edge_mask=edge_mask,
                        n_nodes=n_nodes)


        if partition == 'train':
            loss = loss_l1(pred, (label - meann) / mad)
            loss.backward()
            optimizer.step()

        else:
            loss = loss_l1(mad * pred + meann, label)

        if partition == 'strain':
            pred = pred.cpu()
            pseudolabel = mad * pred + meann
            pseudolabel = pseudolabel.cpu()
 
            for idx, pseudo in zip(data['index'], pseudolabel):
                # print(pseudo.device)
                pseudo = pseudo.detach().numpy()
                idx_pseudo_dict[idx.numpy().tolist()] = pseudo


            # for idx, pseudo, pseudo_pred in zip(data['index'], pseudolabel, pred): # gai
            #     # print(pseudo.device)
            #     pseudo = pseudo.detach().numpy()
            #     pseudo_pred = pseudo_pred.detach().numpy()

            #     idx_pseudo_dict[idx.numpy().tolist()] = (pseudo, pseudo_pred) # gai

                


        res['loss'] += loss.item() * batch_size
        res['counter'] += batch_size
        res['loss_arr'].append(loss.item())

        prefix = ""
        if partition != 'train':
            prefix = ">> %s \t" % partition

        if i % args.log_interval == 0:
            print(prefix + "Epoch %d \t Iteration %d \t loss %.4f" % (epoch, i, sum(res['loss_arr'][-10:])/len(res['loss_arr'][-10:])))


        # if partition != 'strain': # gai
        #     break
        
    if partition == 'strain':
        return idx_pseudo_dict, _, _

    return res['loss'] / res['counter'], current_lr, model


if __name__ == "__main__":
    res = {'epochs': [], 'losess': [], 'best_val': 1e10, 'best_test': 1e10, 'best_epoch': 0}

    time_start_at = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    time_start = time.time()

    start_time = datetime.datetime.now()
    for epoch in range(0, args.epochs):
    # for epoch in range(0, 1): # gai
        # train(epoch, dataloaders['train'], partition='train')
        train_loss, train_lr, model = train(epoch, dataloaders['train'], partition='train')
        if epoch == 0:
            best_model = copy.deepcopy(model)
        # print('train loss', train_loss)
        # input()

        if epoch % args.test_interval == 0:
            val_loss, _, _ = train(epoch, dataloaders['valid'], partition='valid')
            test_loss, _, _ = train(epoch, dataloaders['test'], partition='test')
            res['epochs'].append(epoch)
            res['losess'].append(test_loss)

            if val_loss < res['best_val']:
                res['best_val'] = val_loss
                res['best_test'] = test_loss
                res['best_epoch'] = epoch
                best_model = copy.deepcopy(model)
            print("Val loss: %.4f \t test loss: %.4f \t epoch %d" % (val_loss, test_loss, epoch))
            print("Best: val loss: %.4f \t test loss: %.4f \t epoch %d" % (res['best_val'], res['best_test'], res['best_epoch']))
            cur_time = datetime.datetime.now()
            print(cur_time - start_time)
    
        writer.add_scalar('train loss', train_loss, epoch)
        writer.add_scalar('valid loss', val_loss, epoch)
        writer.add_scalar('test loss', test_loss, epoch)
        writer.add_scalar('training lr', train_lr, epoch)

        json_object = json.dumps(res, indent=4)
        with open(args.outf + "/" + args.exp_name + "/losess_{}.json".format(task_start_at), "w") as outfile:
            outfile.write(json_object)
    
    new_strain_dict, _, _ = train(epoch, dataloaders['strain'], partition='strain', best_model=best_model)
    print(len(new_strain_dict))

    news_dataloaders, charge_scale = dataset.retrieve_strain_dataloaders(args.batch_size, new_strain_dict, args.property, args.num_workers)
    

    for ite in range(1, args.st_iter+1):
        for epoch in range(0, args.epochs):
        # for epoch in range(0, 1): # gai
            # train(epoch, dataloaders['train'], partition='train')
            train_loss, train_lr, model = train(epoch, news_dataloaders['strainall'], partition='train')
            # print('train loss', train_loss)
            # input()

            if epoch % args.test_interval == 0:
                val_loss, _, _ = train(epoch + ite * args.epochs, dataloaders['valid'], partition='valid')
                test_loss, _, _ = train(epoch + ite * args.epochs, dataloaders['test'], partition='test')
                res['epochs'].append(epoch + ite * args.epochs)
                res['losess'].append(test_loss)

                if val_loss < res['best_val']:
                    res['best_val'] = val_loss
                    res['best_test'] = test_loss
                    res['best_epoch'] = epoch + ite * args.epochs
                    best_model = copy.deepcopy(model)
                print("Val loss: %.4f \t test loss: %.4f \t epoch %d" % (val_loss, test_loss, epoch + ite * args.epochs))
                print("Best: val loss: %.4f \t test loss: %.4f \t epoch %d" % (res['best_val'], res['best_test'], res['best_epoch']))
                cur_time = datetime.datetime.now()
                print(cur_time - start_time)
        
            writer.add_scalar('train loss', train_loss, epoch + ite * args.epochs)
            writer.add_scalar('valid loss', val_loss, epoch + ite * args.epochs)
            writer.add_scalar('test loss', test_loss, epoch + ite * args.epochs)
            writer.add_scalar('training lr', train_lr, epoch + ite * args.epochs)

            json_object = json.dumps(res, indent=4)
            with open(args.outf + "/" + args.exp_name + "/losess_{}.json".format(task_start_at), "w") as outfile:
                outfile.write(json_object)
        
        new_strain_dict, _, _ = train(epoch + ite * args.epochs, dataloaders['strain'], partition='strain', best_model=best_model)
        print(len(new_strain_dict))

        news_dataloaders, charge_scale = dataset.retrieve_strain_dataloaders(args.batch_size, new_strain_dict, args.property, args.num_workers)

    # input('debugggggggggggggjsasf')

        

    time_end_at = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    time_end = time.time()
    total = time.time() - time_start
    print('task start training at {} and end at {}'.format(time_start_at, time_end_at))
    print('finish in {}s'.format(total))
