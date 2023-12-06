import numpy as np
import torch
import torch.utils.data as data_utils
from torch.utils.data import Dataset

from arg import getArgparse

args_dict = getArgparse()
is_homologous = args_dict['is_homologous']


def loadfasta(type):
    if type == 'pseudo_label':
        seq = np.loadtxt('dataset/pseudo/pseudo_seq.txt')
        lab = np.loadtxt('dataset/pseudo/pseudo_lab.txt')
        dataset = data_utils.TensorDataset(torch.tensor(seq), torch.tensor(lab))
        return dataset
    if type == 'labelled':
        seq = np.loadtxt('dataset/train/train_seq.txt')
        lab = np.loadtxt('dataset/train/train_lab.txt')
        dataset = data_utils.TensorDataset(torch.tensor(seq), torch.tensor(lab))
        return dataset
    if type == 'test_labelled':
        seq = np.loadtxt('dataset/train/test_train_seq.txt')
        lab = np.loadtxt('dataset/train/test_train_lab.txt')
        dataset = data_utils.TensorDataset(torch.tensor(seq), torch.tensor(lab))
        return dataset
    if type == 'valid':
        seq = np.loadtxt('dataset/valid/valid_seq.txt')
        lab = np.loadtxt('dataset/valid/valid_lab.txt')
        dataset = data_utils.TensorDataset(torch.tensor(seq), torch.tensor(lab))
        return dataset
    if type == 'test':
        seq = np.loadtxt('dataset/test/cb513_seq.txt')
        lab = np.loadtxt('dataset/test/cb513_lab.txt')
        dataset = data_utils.TensorDataset(torch.tensor(seq), torch.tensor(lab))
        return dataset
    if type == 'self_train':
        if is_homologous:
            seq = np.loadtxt('dataset/pseudo/new_pseudo_seq.txt')
            lab = np.loadtxt('dataset/pseudo/new_pseudo_lab.txt')
        else:
            seq = np.loadtxt('dataset/pseudo/pseudo_seq.txt')
            lab = np.loadtxt('dataset/pseudo/pseudo_lab.txt')
        dataset = data_utils.TensorDataset(torch.tensor(seq), torch.tensor(lab))
        return dataset
    if type == 'test_self_train':
        seq = np.loadtxt('dataset/pseudo/test_pseudo_seq.txt')
        lab = np.loadtxt('dataset/pseudo/test_pseudo_lab.txt')
        dataset = data_utils.TensorDataset(torch.tensor(seq), torch.tensor(lab))
        return dataset


def new_dataset(path):
    labelled_seq = np.loadtxt('dataset/train/train_seq.txt')
    labelled_lab = np.loadtxt('dataset/train/train_lab.txt')
    if is_homologous:
        pseudo_seq = np.loadtxt('dataset/pseudo/new_pseudo_seq.txt')
    else:
        pseudo_seq = np.loadtxt('dataset/pseudo/pseudo_seq.txt')
    # test_pseudo_seq = np.loadtxt('dataset/pseudo/test_pseudo_seq.txt')
    predicted_lab = np.loadtxt(path)
    new_seq = np.vstack((labelled_seq, pseudo_seq))
    new_lab = np.vstack((labelled_lab, predicted_lab))
    dataset = data_utils.TensorDataset(torch.tensor(new_seq), torch.tensor(new_lab))
    return dataset


def loadfasta_onehot(type):
    if type == 'labelled':
        seq = np.load('dataset/oneh/train/train_seq.npy') # n*700*21
        lab = np.loadtxt('dataset/oneh/train/train_lab.txt')
        dataset = data_utils.TensorDataset(torch.tensor(seq), torch.tensor(lab))
        return dataset
    if type == 'valid':
        seq = np.load('dataset/oneh/valid/valid_seq.npy')
        lab = np.loadtxt('dataset/oneh/valid/valid_lab.txt')
        dataset = data_utils.TensorDataset(torch.tensor(seq), torch.tensor(lab))
        return dataset
    if type == 'test':
        seq = np.load('dataset/oneh/test/cb513_seq.npy')
        lab = np.loadtxt('dataset/oneh/test/cb513_lab.txt')
        dataset = data_utils.TensorDataset(torch.tensor(seq), torch.tensor(lab))
        return dataset
    if type == 'self_train':
        if is_homologous:
            seq = np.loadtxt('dataset/pseudo/new_pseudo_seq.txt')
            lab = np.loadtxt('dataset/pseudo/new_pseudo_lab.txt')
        else:
            seq = np.loadtxt('dataset/pseudo/pseudo_seq.txt')
            lab = np.loadtxt('dataset/pseudo/pseudo_lab.txt')
        dataset = OnehotDataset(torch.tensor(seq), torch.tensor(lab))
        return dataset
    if type == 'test_self_train':
        seq = np.loadtxt('dataset/pseudo/test_pseudo_seq.txt')
        lab = np.loadtxt('dataset/pseudo/test_pseudo_lab.txt')
        dataset = OnehotDataset(torch.tensor(seq), torch.tensor(lab))
        return dataset


def new_dataset_onehot(path):
    labelled_seq = np.loadtxt('dataset/train/train_seq.txt')
    labelled_lab = np.loadtxt('dataset/train/train_lab.txt')
    if is_homologous:
        pseudo_seq = np.loadtxt('dataset/pseudo/new_pseudo_seq.txt')
    else:
        pseudo_seq = np.loadtxt('dataset/pseudo/pseudo_seq.txt')
    # pseudo_seq = np.loadtxt('dataset/pseudo/test_pseudo_seq.txt')
    predicted_lab = np.loadtxt(path)
    new_seq = np.vstack((labelled_seq, pseudo_seq))
    new_lab = np.vstack((labelled_lab, predicted_lab))
    dataset = OnehotDataset(new_seq, new_lab)
    return dataset


class OnehotDataset(Dataset):
    def __init__(self, seq_list, lab_list, dict_size=21, padding_token=21):
        """Constructor function."""

        self.seqs = seq_list
        self.labs = lab_list
        self.dict_size = dict_size
        self.padding_token = padding_token
    def __len__(self):
        """Get the total number of elements."""
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        lab = self.labs[idx]
        onehot_seq = np.zeros((seq.shape[0], self.dict_size))
        for i in range(seq.shape[0]):
            if int(seq[i]) == self.padding_token:
                break
            onehot_seq[i][int(seq[i])] = 1
        return torch.tensor(onehot_seq), torch.tensor(lab)

