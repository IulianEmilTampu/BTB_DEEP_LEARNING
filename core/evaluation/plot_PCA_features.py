# %% 
'''
Utility that plots the first three PCA components of the feature available in the .pt files.
'''

import os
import numpy as np
from sklearn.decomposition import PCA
import seaborn as sns
import pandas as pd 
import glob 
import torch
import matplotlib.pyplot as plt
import random

# %% PATHS

FEATURE_DIR = '/flush/iulta54/Research/Data/BTB/EXTRACTED_FEATURES/clam_resnet50/BTB_CLAM_fp_feature_extraction/2024-04-03/pt_files'
DATASET_DESCRIPTION_CSV = '/flush/iulta54/Research/P11-BTB_DEEP_LEARNING/dataset_csv_file/BTB_csv_for_training/dataset_summary/dataset_description_tumor_category_rep_0_folds_5.csv'

# load dataset_description
dataset_description = pd.read_csv(DATASET_DESCRIPTION_CSV)

# nbr classes
nbr_classes = len(pd.unique(dataset_description.label))
unique_classes = sorted(list(pd.unique(dataset_description.label)))


# label dict
if 'label_integer' in dataset_description.columns:
    label_dict = dict([(pd.unique(dataset_description.loc[dataset_description.label_integer==i].label)[0], i) for i in range(nbr_classes)])
else:
    label_dict = dict([(unique_classes[i], i) for i in range(len(unique_classes))])


# %% LOAD FEATURES

training_ft, test_ft = [], []
fold_nbr = 0
max_nbr_files = 100

for set, set_name in zip((training_ft, test_ft), ('train', 'test')):
    # get slide_ids for the set
    slide_ids = list(dataset_description.loc[dataset_description[f'fold_{fold_nbr+1}']==set_name].slide_id)
    # list of set files
    feature_files = [f for f in glob.glob(os.path.join(FEATURE_DIR, '*.pt')) if os.path.basename(f).split('.')[0] in slide_ids]
    random.shuffle(feature_files)
    # load file and save
    for idf, f in enumerate(feature_files):
        print(f'Loading file {idf+1:4d}/{len(feature_files)} for the {set_name} set.        \r', end='')
        feature = torch.load(f)
        slide_id = os.path.basename(f).split('.')[0]
        set.append((slide_id, label_dict[dataset_description.loc[dataset_description.slide_id==slide_id].label.values[0]], feature.numpy()))

        if all([max_nbr_files!=-1, idf >= max_nbr_files]):
            break

# %% FIT PCA
tl=PCA(n_components=3)
train_embedding=tl.fit_transform(np.vstack([f[2] for f in training_ft]))

# setting
hue_labels = []
[hue_labels.extend([v[1]]*v[2].shape[0]) for v in training_ft]

# %%
# populate axis
fig , axis = plt.subplots(nrows=2, ncols=2, figsize = (10, 10))
for idx, (ax, dim_indexes, view_names) in enumerate(zip(
    fig.axes, ([0,1], [0,2], [1,2]), ('dim_1 - vs - dim_2', 'dim_1 - vs - dim_3', 'dim_2 - vs - dim_3')
)):
    sns.scatterplot(x= train_embedding[:,dim_indexes[0]], y = train_embedding[:,dim_indexes[1]], hue=hue_labels, style=None, legend=False if idx !=2 else True, ax=ax, alpha=0.2)
    # set title
    ax.set_title(f'{"PCA".upper()} ({view_names})')

    # remove legend for all apart from last plot
    if idx == 2:
        ax.legend(loc='center left',ncol=3, bbox_to_anchor=(1.1, 0.5))
        plt.setp(ax.get_legend().get_texts(), fontsize='6')

# hide last axis
axis[1,1].axis('off')

# %%