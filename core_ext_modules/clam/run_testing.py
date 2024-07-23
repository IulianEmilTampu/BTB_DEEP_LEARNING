
# %% 
'''
Main script for running testing of a model on a given task. It runs the evaluation
on all the models given in cfg.trained_model path and for all the splits in the task. 
'''
from __future__ import print_function

import argparse
import pdb
import os
import sys
import math
from datetime import datetime
from tqdm import tqdm

# internal imports
from utils.file_utils import save_pkl, load_pkl
from utils.utils import get_simple_loader
from utils.core_utils import summary
from utils.eval_utils import initiate_model
from datasets.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset

# pytorch imports
import torch
from torch.utils.data import DataLoader, sampler
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np
import hydra
import omegaconf
from omegaconf import DictConfig, open_dict, OmegaConf
import pathlib

# %% MAIN

@hydra.main(
    version_base="1.2.0", config_path=os.path.join('config', 'classification'), config_name='default_testing'
)

def main(cfg:DictConfig):
    # run evaluation for all the models in the list of trained_models
    with tqdm(total=len(cfg.trained_models), unit='model') as model_tqdm:
        for m in cfg.trained_models:
            with open_dict(cfg):
                cfg.trained_model = m
                single_model_evaluation(cfg)
            model_tqdm.update()

def single_model_evaluation(cfg):
    # %%  LOAD MODEL TRAINING CONFIGURATION
    # # training configuration file for automatic parsing of the model configuration (needed for model initialization and data generator).
    # check the given trained model path
    if not os.path.isdir(cfg.trained_model):
        raise ValueError(f'The path to the trained model folder does not exist. Give {cfg.trained_model}.')
    else:
        # check that the hydra configuration file is also available (needed to parse information about the model and the task that was trained for)
        if not os.path.isfile(pathlib.Path(cfg.trained_model, f'hydra_config.yaml')):
            raise ValueError(f'The hydra configuration file for the trained model could not be found. In future implementations, the model information can be given or parsed from the trained model name.')
        else:
            # load the training configuration file
            training_cfg = omegaconf.OmegaConf.load(pathlib.Path(cfg.trained_model, f'hydra_config.yaml'))

    # %%  load the task information and initialize the data generator on which to run the evaluation
    # add task arguments to the level 0 of cfg
    with open_dict(cfg):
        cfg.n_classes = cfg.task.n_classes
        cfg.subtyping = cfg.task.subtyping
        cfg.feature_level = training_cfg.feature_level
        cfg.magnification = training_cfg.magnification
        cfg.patch_size = training_cfg.patch_size
        cfg.shuffle = False
        cfg.seed = 20091229
        cfg.feature_extractor = training_cfg.feature_extractor
        cfg.patient_strat = training_cfg.patient_strat
        cfg.print_info = False
    
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
    
    # %% FOR ALL THE FOLDS, RUN EVALUATION 
    # # build model initialization arguments using the trained model configuration
    model_args = omegaconf.DictConfig({
        'model_type' : training_cfg.model_type,
        'model_size' : training_cfg.model_size,
        'drop_out' : training_cfg.drop_out,
        'n_classes' : training_cfg.task.n_classes,
        'feature_encoding_size' : training_cfg.encoding_size,
    })

    # # define results dir and save cfg
    with open_dict(cfg):
        cfg.evaluation_results_dir = os.path.join(cfg.trained_model, f'evaluation_task_{cfg.task.version}')
    pathlib.Path(cfg.evaluation_results_dir).mkdir(parents=True, exist_ok=True)

    # # for all the folds
    with tqdm(total=training_cfg.k, unit='fold') as fold_tqdm:
        for ckpt_idx in range(training_cfg.k):
            # check that the .pt file of the trained model is available for the specified fold.
            if not os.path.isfile(pathlib.Path(cfg.trained_model, f's_{ckpt_idx}_checkpoint.pt')):
                raise ValueError(f'The .pt file of the model checkpoint for the specified fold could not be found.')
            else:
                model_checkpoint_path = pathlib.Path(cfg.trained_model, f's_{ckpt_idx}_checkpoint.pt')
            
            # # # load model
            model = initiate_model(model_args, model_checkpoint_path)

            # # # build dataset (TODO here it is assumed that the task to evaluate has only one split. Add a second for loop o
            # iterate over the different splits). 
            datasets = dataset.return_splits(from_id=False, 
                    csv_path=os.path.join(cfg.task.split_dir, f'split_{0}.csv'))
            datasets_idx = {'train':0, 'validation':1, 'test':2}
            split_dataset = datasets[datasets_idx[cfg.set_to_evaluate]]
            loader = get_simple_loader(split_dataset)

            # # # get summary performance on the dataset
            evaluation_results, error, auc, _ = summary(model, loader, cfg.n_classes)
            filename = os.path.join(cfg.evaluation_results_dir, f'split_{ckpt_idx}_results.pkl')
            save_pkl(filename, evaluation_results)

            fold_tqdm.update()


if __name__ == "__main__":

    main()