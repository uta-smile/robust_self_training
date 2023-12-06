import sys
import os
import copy
import time

from arg import getArgparse
import torch
from torch import nn
from network import S4PRED
from get_dataset import loadfasta, new_dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import datetime
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

start = datetime.datetime.now()
args_dict = getArgparse()
device = torch.device(args_dict['device'])
learn_rate = args_dict['learn_rate']
epochs = args_dict['epochs']
batch_size = args_dict['batch_size']
save_path = args_dict['save_path']
teacher_fpath = args_dict['teacher_fpath']
options = args_dict['options']
num_workers = args_dict['num_workers']
dropout = args_dict['dropout']
iteration = args_dict['iteration']
data_path = args_dict['data_path']
loss_name = args_dict['loss_name']
loss_weights = args_dict['loss_weights']
is_init = args_dict['is_init']
load_student = args_dict['load_student']
load_model = args_dict['load_model']
args_q = args_dict['gce_q']
dmice_p = args_dict['dmice_p']

if not os.path.exists(save_path):
    os.mkdir(save_path)
if not os.path.exists(data_path):
    os.mkdir(data_path)
num_classes = 4
print(time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))


def main():
    criterion = get_criterion(num_classes, loss_name, loss_weights)
    test_loader = DataLoader(loadfasta("test"), batch_size=1, shuffle=False, num_workers=num_workers)
    valid_loader = DataLoader(loadfasta("valid"), batch_size=1, shuffle=False, num_workers=num_workers)
    self_train_loader = DataLoader(loadfasta("self_train"), batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = S4PRED(dropout=dropout).to(device)
    model = nn.DataParallel(model)
    pretrain_model = S4PRED(dropout=dropout).to(device)

    if load_model:
        checkpoint = torch.load(os.path.join(save_path, 'checkpoint.pth.tar'))
        i_num = checkpoint['iteration']
        ep_num = checkpoint['epoch']
        accuracy = checkpoint['accuracy']
        optimizer = checkpoint['optimizer']
        try:
            pretrain_model.load_state_dict(torch.load(os.path.join(save_path, str(i_num) + '_best.pkl')))
        except:
            pretrain_model = nn.DataParallel(pretrain_model)
            pretrain_model.load_state_dict(torch.load(os.path.join(save_path, str(i_num) + '_best.pkl')))
    else:
        i_num = 0
        ep_num = 0
        accuracy = [0.0]
        try:
            pretrain_model.load_state_dict(torch.load(teacher_fpath))
        except:
            pretrain_model = nn.DataParallel(pretrain_model)
            pretrain_model.load_state_dict(torch.load(teacher_fpath))
        optimizer = optim.Adam(model.parameters(), lr=learn_rate, betas=(0.9, 0.999))

    if not is_init:
        model.load_state_dict(torch.load(teacher_fpath))

    if ep_num == epochs - 1:
        i_num = i_num + 1
        ep_num = 0

    for i in range(i_num, iteration):
        best_model = copy.deepcopy(model)
        if load_student == 'best':
            self_train_model = copy.deepcopy(pretrain_model)
        else:
            self_train_model = copy.deepcopy(model)
        # optimizer = optim.Adam(model.parameters(), lr=learn_rate, betas=(0.9, 0.999))
        print(f'Iteration  {i} ')
        print('Get predicted data')
        get_predict(self_train_loader, i, pretrain_model)
        print('Loading new dataset')
        iteration_loader = DataLoader(new_dataset(os.path.join(data_path, str(i))), batch_size=batch_size, shuffle=True, num_workers=num_workers)
        for epoch in range(ep_num, epochs):
            print('## Epoch-%s' % epoch)
            model = train(self_train_model, iteration_loader, optimizer, criterion=criterion)
            best_model, last_model, accuracy = valid(model, best_model, valid_loader, epoch, i, accuracy,optimizer, criterion=criterion)
            test(last_model, test_loader)
        ep_num = 0
        accuracy = [0.0]
        print(f'------Starting test best-model of iteration {i} ------')
        test(best_model, test_loader)
        pretrain_model = copy.deepcopy(best_model)


def train(model, train_loader, optimizer, criterion):
    model.train()
    train_loss = 0
    epoch_acc = 0
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        sequence, label = data
        output = model(sequence.to(torch.long).to(device))  # batch_size * 4 * 700
        loss = criterion(output, label.to(torch.long).to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.25)
        optimizer.step()
        train_loss += loss.item()
        if len(output.shape) == 2:
            output = torch.unsqueeze(output, 0)
        _, predicted = torch.max(output.data, 1)  # batch_size * 700
        epoch_acc += getAccuracy(predicted.to('cpu'), label.to(torch.long).to('cpu'))
    print('Train Accuracy: ', epoch_acc / len(train_loader), ' Train Loss', train_loss / len(train_loader),
          datetime.datetime.now() - start)
    return model


def valid(model, best_model, valid_loader, epoch, iteration_num, accuracy,optimizer, criterion):
    model.eval()
    val_loss = 0
    epoch_acc = 0
    with torch.no_grad():
        for i, data in enumerate(valid_loader):
            sequence, label = data
            output = model(sequence.to(torch.long).to(device))  # batch_size * 4 * 700
            if len(output.shape) == 2:
                output = torch.unsqueeze(output, 0)
            loss = criterion(output, label.to(torch.long).to(device))
            val_loss += loss.item()
            _, predicted = torch.max(output.data, 1)  # batch_size * 700
            epoch_acc += getAccuracy(predicted.to('cpu'), label.to(torch.long).to(device))
    print('Val Accuracy: ', epoch_acc / len(valid_loader), ' Val Loss', val_loss / len(valid_loader))
    # save model
    torch.save(model.state_dict(), os.path.join(save_path, str(iteration_num + 1) + '_{}.pkl'.format(epoch)))
    # torch.save(model.state_dict(), os.path.join(save_path, str(iteration_num + 1) + '_last.pkl'))
    if epoch_acc / len(valid_loader) > max(accuracy):
        torch.save(model.state_dict(), os.path.join(save_path, str(iteration_num + 1) + '_best.pkl'))
        accuracy.append(epoch_acc / len(valid_loader))
        best_model = copy.deepcopy(model)
    last_model = save_model(iteration_num, epoch, model,optimizer, accuracy)
    return best_model, last_model, accuracy


def test(model, test_loader):
    model.eval()
    test_acc = 0
    f1 = 0
    precision = 0
    recall = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            sequence, label = data
            output = model(sequence.to(torch.long).to(device))  # batch_size * 4 * 700
            if len(output.shape) == 2:
                output = torch.unsqueeze(output, 0)
                # output = output[None]
            _, predicted = torch.max(output.data, 1)  # batch_size * 700
            test_acc += getAccuracy(predicted.to('cpu'), label.to(torch.long).to('cpu'))
            res = get_score(predicted.to('cpu'), label.to(torch.long).to('cpu'))
            f1 += res['f1']
            precision += res['precision']
            recall += res['recall']
        print('Test Accuracy: ', test_acc / len(test_loader), '\nF1 score', f1 / len(test_loader),
              '\nPrecision score', precision / len(test_loader),
              '\nRecall score', recall / len(test_loader), datetime.datetime.now() - start)


def get_predict(self_train_loader, model_name, pretrain_model):
    save_predicted_fpath = os.path.join(data_path, str(model_name))
    if os.path.exists(save_predicted_fpath):
        return
    pretrain_model.eval()

    f = open(os.path.join(data_path, str(model_name)), 'wb')
    with torch.no_grad():
        for i, data in enumerate(self_train_loader):
            sequence, label = data
            output = pretrain_model(sequence.to(torch.long).to(device))  # batch_size * 4 * 700
            if len(output.shape) == 2:
                output = torch.unsqueeze(output, 0)
            _, predicted = torch.max(output.data, 1)  # batch_size * 700
            np.savetxt(f, predicted.to('cpu').numpy(), fmt='%d')
    f.close()


def save_model(iteration, epoch, model, optimizer,accuracy):
    state = {
        'iteration': iteration,
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'accuracy': accuracy
    }
    torch.save(state, os.path.join(save_path, 'checkpoint.pth.tar'))
    return model


def getAccuracy(output, label):
    acc = 0
    for i in range(label.size(0)):
        total = 0
        count = 0
        for j in range(label.size(1)):
            if label[i][j].item() != 3:
                total += 1
                if label[i][j].item() == output[i][j].item():
                    count += 1
            else:
                break
        acc += count / total
    return acc / label.size(0)


def get_score(output, label):
    f1 = 0
    precision = 0
    recall = 0
    for i in range(label.size(0)):
        j = 0
        for j in range(len(label[i])):
            if label[i][j] == 3:
                break
        f1 += f1_score(label[i][:j + 1], output[i][:j + 1], average='weighted')
        precision += precision_score(label[i][:j + 1], output[i][:j + 1], average='weighted')
        recall += recall_score(label[i][:j + 1], output[i][:j + 1], average='weighted')
    return {'f1': f1 / label.size(0), 'precision': precision / label.size(0), 'recall': recall / label.size(0)}


def get_criterion(num_classes, options, loss_weights):
    loss_options = {
        'CE': torch.nn.CrossEntropyLoss(ignore_index=3),
        'JSWCS': JensenShannonDivergenceWeightedScaled(num_classes=num_classes, weights=loss_weights), # '[0.7 0.3]'
        'GCE': GeneralizedCrossEntropy(num_classes=num_classes, q=args_q),
        'DMI_CE': DMICE(num_classes=num_classes, dmice_p=dmice_p)
    }

    if options in loss_options:
        criterion = loss_options[options]
    else:
        raise("Unknown loss")
    return criterion


def custom_kl_div(prediction, target):
    output_pos = target * (target.clamp(min=1e-7).log() - prediction)
    zeros = torch.zeros_like(output_pos)
    output = torch.where(target > 0, output_pos, zeros)
    output = torch.sum(output, axis=1)
    return output.mean()


class JensenShannonDivergenceWeightedScaled(torch.nn.Module):
    def __init__(self, num_classes, weights):
        super(JensenShannonDivergenceWeightedScaled, self).__init__()
        self.num_classes = num_classes
        self.weights = [float(w) for w in weights.split(',')]

        self.scale = -1.0 / ((1.0 - self.weights[0]) * np.log((1.0 - self.weights[0])))
        assert abs(1.0 - sum(self.weights)) < 0.001

    def forward(self, pred, labels):
        pred = pred.permute(0, 2, 1)
        loss = 0
        j = 0
        for m in range(pred.shape[0]):
            for j in range(pred.shape[1]):
                if labels[m][j] == 3:
                    break
            new_pred = pred[m, :j, :3]
            new_labels = labels[m, :j]
            preds = list()
            if type(new_pred) == list:
                for i, p in enumerate(new_pred):
                    preds.append(F.softmax(p, dim=1))
            else:
                preds.append(F.softmax(new_pred, dim=1))
            new_labels = F.one_hot(new_labels.to(torch.long), self.num_classes - 1).float()
            distribs = [new_labels] + preds
            assert len(self.weights) == len(distribs)
            mean_distrib = sum([w * d for w, d in zip(self.weights, distribs)])
            mean_distrib_log = mean_distrib.clamp(1e-7, 1.0).log()
            jsw = sum([w * custom_kl_div(mean_distrib_log, d) for w, d in zip(self.weights, distribs)])
            loss += self.scale * jsw
        return loss / pred.shape[0]


class GeneralizedCrossEntropy(torch.nn.Module):
    def __init__(self, num_classes, q=0.7):
        super(GeneralizedCrossEntropy, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.q = q

    def forward(self, pred, labels):
        pred = pred.permute(0, 2, 1)
        loss = 0
        j = 0
        for m in range(pred.shape[0]):
            for j in range(pred.shape[1]):
                if labels[m][j] == 3:
                    break
            new_pred = pred[m, :j, :3]
            new_labels = labels[m, :j]

            new_pred = F.softmax(new_pred, dim=1)
            new_pred = torch.clamp(new_pred, min=1e-7, max=1.0)
            label_one_hot = torch.nn.functional.one_hot(new_labels, self.num_classes - 1).float().to(self.device)
            gce = (1. - torch.pow(torch.sum(label_one_hot * new_pred, dim=1), self.q)) / self.q
            loss += gce.mean()
        return loss / pred.shape[0]


class DMICE(torch.nn.Module):
    def __init__(self, num_classes, dmice_p=0.1):
        super(DMICE, self).__init__()
        self.dmice_p = dmice_p
        self.num_classes = num_classes

    def forward(self, pred, labels):
        ce_loss = torch.nn.CrossEntropyLoss(ignore_index=3)
        ce_loss = ce_loss(pred, labels)
        pred = pred.permute(0, 2, 1)
        loss = 0
        j = 0
        for i in range(pred.shape[0]):
            for j in range(pred.shape[1]):
                if labels[i][j] == 3:
                    break
            new_pred = pred[i, :j, :3]
            new_labels = labels[i, :j]
            outputs = F.softmax(new_pred, dim=1)
            targets = new_labels.reshape(new_labels.size(0), 1).cpu()
            y_onehot = torch.FloatTensor(new_labels.size(0), num_classes-1).zero_()
            y_onehot.scatter_(1, targets, 1)
            y_onehot = y_onehot.transpose(0, 1).to(device)
            mat = y_onehot @ outputs
            mat = mat / new_labels.size(0)
            det = torch.det(mat.float())
            if det < 0:
                dmi_loss = torch.log(torch.abs(det) + 0.001)
            else:
                dmi_loss = -torch.log(torch.abs(det) + 0.001)
            loss += dmi_loss
        return dmice_p * (loss/pred.shape[0]) + ce_loss


if __name__ == "__main__":
    main()
