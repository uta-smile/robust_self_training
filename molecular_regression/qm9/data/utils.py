import torch
import numpy as np

import logging
import os

from torch.utils.data import DataLoader
from qm9.data.dataset import ProcessedDataset
from qm9.data.prepare import prepare_dataset


def initialize_datasets(args, datadir, dataset, subset=None, splits=None,
                        force_download=False, subtract_thermo=False):
    """
    Initialize datasets.

    Parameters
    ----------
    args : dict
        Dictionary of input arguments detailing the cormorant calculation.
    datadir : str
        Path to the directory where the data and calculations and is, or will be, stored.
    dataset : str
        String specification of the dataset.  If it is not already downloaded, must currently by "qm9" or "md17".
    subset : str, optional
        Which subset of a dataset to use.  Action is dependent on the dataset given.
        Must be specified if the dataset has subsets (i.e. MD17).  Otherwise ignored (i.e. GDB9).
    splits : str, optional
        TODO: DELETE THIS ENTRY
    force_download : bool, optional
        If true, forces a fresh download of the dataset.
    subtract_thermo : bool, optional
        If True, subtracts the thermochemical energy of the atoms from each molecule in GDB9.
        Does nothing for other datasets.

    Returns
    -------
    args : dict
        Dictionary of input arguments detailing the cormorant calculation.
    datasets : dict
        Dictionary of processed dataset objects (see ????? for more information).
        Valid keys are "train", "test", and "valid"[ate].  Each associated value
    num_species : int
        Number of unique atomic species in the dataset.
    max_charge : pytorch.Tensor
        Largest atomic number for the dataset.

    Notes
    -----
    TODO: Delete the splits argument.
    """


    # Set the number of points based upon the arguments
    num_pts = {'train': args.num_train,
               'test': args.num_test, 'valid': args.num_valid,
               'strain': args.num_strain}

    # Download and process dataset. Returns datafiles.
    datafiles = prepare_dataset(
        datadir, dataset, subset, splits, force_download=force_download)

    # Load downloaded/processed datasets
    datasets = {}
    for split, datafile in datafiles.items():
        with np.load(datafile) as f:
            datasets[split] = {key: torch.from_numpy(
                val) for key, val in f.items()}

    # Basic error checking: Check the training/test/validation splits have the same set of keys.
    keys = [list(data.keys()) for data in datasets.values()]
    assert all([key == keys[0] for key in keys]
               ), 'Datasets must have same set of keys!'

    # Get a list of all species across the entire dataset
    all_species = _get_species(datasets, ignore_check=False)

    # Now initialize MolecularDataset based upon loaded data

    datasets = {split: ProcessedDataset(data, num_pts=num_pts.get(
        split, -1), included_species=all_species, subtract_thermo=subtract_thermo) for split, data in datasets.items()}
    # print(datasets['train'])
    # input('debug')

    # Now initialize MolecularDataset based upon loaded data

    # Check that all datasets have the same included species:
    assert(len(set(tuple(data.included_species.tolist()) for data in datasets.values())) ==
           1), 'All datasets must have same included_species! {}'.format({key: data.included_species for key, data in datasets.items()})

    # These parameters are necessary to initialize the network
    num_species = datasets['train'].num_species
    max_charge = datasets['train'].max_charge

    # Now, update the number of training/test/validation sets in args
    args.num_train = datasets['train'].num_pts
    args.num_valid = datasets['valid'].num_pts
    args.num_test = datasets['test'].num_pts
    args.num_strain = datasets['strain'].num_pts

    return args, datasets, num_species, max_charge


def initialize_strain_datasets(args, datadir, dataset, new_strain_dict, prop, subset=None, splits=None,
                        force_download=False, subtract_thermo=False):
    """
    Initialize datasets.

    Parameters
    ----------
    args : dict
        Dictionary of input arguments detailing the cormorant calculation.
    datadir : str
        Path to the directory where the data and calculations and is, or will be, stored.
    dataset : str
        String specification of the dataset.  If it is not already downloaded, must currently by "qm9" or "md17".
    subset : str, optional
        Which subset of a dataset to use.  Action is dependent on the dataset given.
        Must be specified if the dataset has subsets (i.e. MD17).  Otherwise ignored (i.e. GDB9).
    splits : str, optional
        TODO: DELETE THIS ENTRY
    force_download : bool, optional
        If true, forces a fresh download of the dataset.
    subtract_thermo : bool, optional
        If True, subtracts the thermochemical energy of the atoms from each molecule in GDB9.
        Does nothing for other datasets.

    Returns
    -------
    args : dict
        Dictionary of input arguments detailing the cormorant calculation.
    datasets : dict
        Dictionary of processed dataset objects (see ????? for more information).
        Valid keys are "train", "test", and "valid"[ate].  Each associated value
    num_species : int
        Number of unique atomic species in the dataset.
    max_charge : pytorch.Tensor
        Largest atomic number for the dataset.

    Notes
    -----
    TODO: Delete the splits argument.
    """


    # Set the number of points based upon the arguments
    num_pts = {'train': args.num_train,
               'test': args.num_test, 'valid': args.num_valid,
               'strain': args.num_strain}

    # Download and process dataset. Returns datafiles.
    datafiles = prepare_dataset(
        datadir, dataset, subset, splits, force_download=force_download)

    # Load downloaded/processed datasets
    datasets = {}

    strain_datafile = datafiles['strain']
    train_datafile = datafiles['train']

    valid_datafile = datafiles['valid']
    test_datafile = datafiles['test']


    with np.load(train_datafile) as f:
        train_dataset = {key: torch.from_numpy(
            val) for key, val in f.items()}

    with np.load(strain_datafile) as f:
        strain_dataset = {key: torch.from_numpy(
            val) for key, val in f.items()}

    with np.load(valid_datafile) as f:
        valid_dataset = {key: torch.from_numpy(
            val) for key, val in f.items()}
    with np.load(test_datafile) as f:
        test_dataset = {key: torch.from_numpy(
            val) for key, val in f.items()}

    pseudo_label = []
    pseudo_label_thermo = []
    pseudo_pred = [] # gai
    for idx in strain_dataset['index']:
        pseudo_label.append(new_strain_dict[idx.numpy().tolist()])
        pseudo_label_thermo.append(0)
        # pseudo_label.append(new_strain_dict[idx.numpy().tolist()][0]) # gai
        # pseudo_pred.append(new_strain_dict[idx.numpy().tolist()][1]) # gai
    pseudo_label = np.array(pseudo_label)
    pseudo_label_thermo = np.array(pseudo_label_thermo)
    # pseudo_pred = np.array(pseudo_pred) # gai


    # old_label = strain_dataset[prop] - strain_dataset['{}_thermo'.format(prop)]
    
    strain_dataset[prop] = torch.from_numpy(pseudo_label)
    if '{}_thermo'.format(prop) in strain_dataset:
        strain_dataset['{}_thermo'.format(prop)] = torch.from_numpy(pseudo_label_thermo)

    # debug
    # with open('/smile/nfs/yuzhi/egnn-main/debug/duibi.csv', 'w') as w:
    #     for old, pseudo, pseudo_pre in zip(old_label, strain_dataset[prop], pseudo_pred):
    #         w.write('{}, {}, {}\n'.format(old, pseudo, pseudo_pre))
    # input('debug')
    

    # print(strain_dataset)

    strainall_dataset = {}
    for key in strain_dataset:
        cat_value = torch.cat((train_dataset[key], strain_dataset[key]), 0)
        strainall_dataset[key] = cat_value
    # 'index'
    # print(strainall_dataset)
    

    datasets['strainall'] = strainall_dataset
    datasets['valid'] = valid_dataset
    datasets['test'] = test_dataset

    # for split, datafile in datafiles.items():
    #     with np.load(datafile) as f:
    #         datasets[split] = {key: torch.from_numpy(
    #             val) for key, val in f.items()}

    # Basic error checking: Check the training/test/validation splits have the same set of keys.
    keys = [list(data.keys()) for data in datasets.values()] 
    assert all([key == keys[0] for key in keys]
               ), 'Datasets must have same set of keys!'

    # Get a list of all species across the entire dataset
    all_species = _get_species(datasets, ignore_check=False)

    # Now initialize MolecularDataset based upon loaded data
    datasets = {split: ProcessedDataset(data, num_pts=num_pts.get(
        split, -1), included_species=all_species, subtract_thermo=subtract_thermo) for split, data in datasets.items()}

    # Now initialize MolecularDataset based upon loaded data

    # Check that all datasets have the same included species:
    assert(len(set(tuple(data.included_species.tolist()) for data in datasets.values())) ==
           1), 'All datasets must have same included_species! {}'.format({key: data.included_species for key, data in datasets.items()})

    # These parameters are necessary to initialize the network
    num_species = datasets['strainall'].num_species
    max_charge = datasets['strainall'].max_charge

    # Now, update the number of training/test/validation sets in args
    args.num_train = datasets['strainall'].num_pts
    args.num_valid = datasets['valid'].num_pts
    args.num_test = datasets['test'].num_pts
    args.num_strain = datasets['strainall'].num_pts

    return args, datasets, num_species, max_charge

def _get_species(datasets, ignore_check=False):
    """
    Generate a list of all species.

    Includes a check that each split contains examples of every species in the
    entire dataset.

    Parameters
    ----------
    datasets : dict
        Dictionary of datasets.  Each dataset is a dict of arrays containing molecular properties.
    ignore_check : bool
        Ignores/overrides checks to make sure every split includes every species included in the entire dataset

    Returns
    -------
    all_species : Pytorch tensor
        List of all species present in the data.  Species labels should be integers.

    """
    # Get a list of all species in the dataset across all splits
    all_species = torch.cat([dataset['charges'].unique()
                             for dataset in datasets.values()]).unique(sorted=True)

    # Find the unique list of species in each dataset.
    split_species = {split: species['charges'].unique(
        sorted=True) for split, species in datasets.items()}

    # If zero charges (padded, non-existent atoms) are included, remove them
    if all_species[0] == 0:
        all_species = all_species[1:]

    # Remove zeros if zero-padded charges exst for each split
    split_species = {split: species[1:] if species[0] ==
                     0 else species for split, species in split_species.items()}

    # Now check that each split has at least one example of every atomic spcies from the entire dataset.
    if not all([split.tolist() == all_species.tolist() for split in split_species.values()]):
        # Allows one to override this check if they really want to. Not recommended as the answers become non-sensical.
        if ignore_check:
            logging.error(
                'The number of species is not the same in all datasets!')
        else:
            raise ValueError(
                'Not all datasets have the same number of species!')

    # Finally, return a list of all species
    return all_species
