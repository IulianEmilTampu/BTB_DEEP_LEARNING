from __future__ import print_function

import argparse
import pdb
import os
import sys
import math

# internal imports
from utils.file_utils import save_pkl, load_pkl
from utils.utils import *
from utils.core_utils import train
from datasets.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset

# pytorch imports
import torch
from torch.utils.data import DataLoader, sampler
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np
import hydra
from omegaconf import DictConfig, open_dict
import pathlib

# %% UTILITIES

def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def check_splits(cfg:DictConfig):
    '''
    Utility that checks:
    1. that the given split directory exists
    2. that the flagged folds for training are available
    3. that the train, val and test slide IDs have available features.
    '''

    # check that the split folder exists
    if not os.path.isdir(cfg.task.split_dir):
        raise ValueError(f'The given split directory is not valid. Check task configuration.\nGiven: {cfg.task.split_dir}')
    
    # check that the folds specified exist
    folds_check = [os.path.isfile(os.path.join(cfg.task.split_dir, f'split_{f}.csv')) for f in range(cfg.k)]
    if all(folds_check):
        print(f'The split files for all the folds are available.')
    else:
        raise ValueError(f'Some of the split files were not found: {[i for i, s in enumerate(folds_check) if not s]}')
    
    # check that the slide ids are available in the feature folder
    for f in range(cfg.k):
        # load the .csv split file 
        split_file = pd.read_csv(os.path.join(cfg.task.split_dir, f'split_{f}.csv'))
        # check that the training, val and test columns are available, and check if the features are available
        for c in ['train', 'val', 'test']:
            if c not in split_file.columns:
                raise ValueError(f'The {c} column is not available as a column in the split .csv file for split nbr {f}.')
            else:
                # check features
                slide_ids = split_file[c].dropna().reset_index(drop=True).tolist()
                feature_check = [os.path.isfile(os.path.join(cfg.task.data_root_dir, 'pt_files', f'{sid}.pt')) for sid in slide_ids]
                if not all(feature_check):
                    raise ValueError(f'Missing feature files for {feature_check.count(False)} slide_ids for set {c} and split {f}.')
    
    # if survives until here, the check is passed
    print('Check of split files passed!')

# %% MAIN

@hydra.main(
    version_base="1.2.0", config_path=os.path.join('config', 'classification'), config_name='default'
)
def main(cfg:DictConfig):
    # run set up
    # # get device
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # # seed everything
    seed_torch(cfg.seed)

    # parse training and model settings
    encoding_size = cfg.encoding_size # TODO Make this to be infered when loading the vectors.

    settings = {'num_splits': cfg.k, 
                'k_start': cfg.k_start,
                'k_end': cfg.k_end,
                'task': cfg.task,
                'max_epochs': cfg.max_epochs, 
                'results_dir': cfg.results_dir, 
                'lr': cfg.lr,
                'experiment': cfg.exp_code,
                'reg': cfg.reg,
                'label_frac': cfg.label_frac,
                'bag_loss': cfg.bag_loss,
                'seed': cfg.seed,
                'model_type': cfg.model_type,
                'model_size': cfg.model_size,
                "use_drop_out": cfg.drop_out,
                'weighted_sample': cfg.weighted_sample,
                'opt': cfg.opt}

    if cfg.model_type in ['clam_sb', 'clam_mb']:
        settings.update({'bag_weight': cfg.bag_weight,
                            'inst_loss': cfg.inst_loss,
                            'B': cfg.B})
    
    # add task arguments to the level 0 of cfg
    with open_dict(cfg):
        cfg.n_classes = cfg.task.n_classes
        cfg.subtyping = cfg.task.subtyping

    # parse task and dataset information
    print('\nLoad Dataset...')
    if cfg.task.description:
        print(f'Task description: {cfg.task.description}')

    dataset = Generic_MIL_Dataset(csv_path = cfg.task.csv_path,
                            data_dir= cfg.task.data_root_dir,
                            shuffle = cfg.shuffle, 
                            seed = cfg.seed, 
                            print_info = cfg.print_info,
                            label_dict = cfg.task.label_dict,
                            patient_strat= cfg.patient_strat,
                            ignore=cfg.task.ignore)

    if all([cfg.task.n_classes > 2, cfg.model_type in ['clam_sb', 'clam_mb']]):
            assert cfg.task.subtyping 
    
    # make folder to where the model training outputs are saved
    cfg.results_dir = os.path.join(cfg.results_dir, str(cfg.exp_code) + '_s{}'.format(cfg.seed))
    if not os.path.isdir(cfg.results_dir):
        pathlib.Path(cfg.results_dir).mkdir(parents=True, exist_ok=False)

    # check if the given splits
    check_splits(cfg)
    settings.update({'split_dir': cfg.task.split_dir})

    with open(cfg.results_dir + '/experiment_{}.txt'.format(cfg.exp_code), 'w') as f:
        print(settings, file=f)
    f.close()

    print("################# Settings ###################")
    for key, val in settings.items():
        print("{}:  {}".format(key, val))        

    if cfg.k_start == -1:
        start = 0
    else:
        start = cfg.k_start
    if cfg.k_end == -1:
        end = cfg.k
    else:
        end = cfg.k_end

    all_test_auc = []
    all_val_auc = []
    all_test_acc = []
    all_val_acc = []
    folds = np.arange(start, end)

    for i in folds:
        train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False, 
                csv_path=os.path.join(cfg.task.split_dir, f'split_{i}.csv'))
        
        datasets = (train_dataset, val_dataset, test_dataset)
        results, test_auc, val_auc, test_acc, val_acc  = train(datasets, i, cfg)
        all_test_auc.append(test_auc)
        all_val_auc.append(val_auc)
        all_test_acc.append(test_acc)
        all_val_acc.append(val_acc)
        # write results to pkl
        filename = os.path.join(cfg.results_dir, 'split_{}_results.pkl'.format(i))
        save_pkl(filename, results)

    final_df = pd.DataFrame({'folds': folds, 'test_auc': all_test_auc, 
        'val_auc': all_val_auc, 'test_acc': all_test_acc, 'val_acc' : all_val_acc})

    if len(folds) != cfg.k:
        save_name = 'summary_partial_{}_{}.csv'.format(start, end)
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(cfg.results_dir, save_name))


if __name__ == "__main__":
    main()
    print("finished!")
    print("end script")


