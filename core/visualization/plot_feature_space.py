# %% 
'''
Script that plots the feature space using PCA or other dimensionality reduction methods for features extracted using different pre-trained models. 
'''

import os 
import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt
import pathlib
from datetime import datetime

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns

import tqdm 
import torch

# %% UTILITIES

def plot_PCA_embeddings(
    embedding,
    hue_labels,
    style_labels=None,
    draw: bool = True,
    save_figure: str = None,
    save_path: str = None,
    prefix: str = "Embeddings_cluster",
    nbr_legend_columns: int = 3,
    value_ranges=None,
    marker_size:int=5,
):
    # define hue order
    hue_order = list(dict.fromkeys(hue_labels))
    hue_order.sort()
    # create axis
    fig, axis = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

    # populate axis
    for idx, (ax, dim_indexes, view_names) in enumerate(
        zip(
            fig.axes,
            ([0, 1], [0, 2], [1, 2]),
            ("dim_1 - vs - dim_2", "dim_1 - vs - dim_3", "dim_2 - vs - dim_3"),
        )
    ):
        sns.scatterplot(
            x=embedding[:, dim_indexes[0]],
            y=embedding[:, dim_indexes[1]],
            hue=hue_labels,
            hue_order=hue_order,
            style=style_labels,
            legend=False if idx != 2 else True,
            ax=ax,
            s=marker_size,
        )
        # set axis limits
        if value_ranges:
            ax.set_xlim(
                (value_ranges[dim_indexes[0]][0], value_ranges[dim_indexes[0]][1])
            )
            ax.set_ylim(
                (value_ranges[dim_indexes[1]][0], value_ranges[dim_indexes[1]][1])
            )

        # set title
        ax.set_title(f"PCA ({view_names})")

        # remove legend for all apart from last plot
        if idx == 2:
            # ax.legend(
            #     loc="center left", ncol=nbr_legend_columns, bbox_to_anchor=(1.1, 0.5)
            # )
            # plt.setp(ax.get_legend().get_texts(), fontsize="5")

            # remove legend for all apart from last plot
            lgnd = ax.legend(
                loc="center left", ncol=3, bbox_to_anchor=(1.1, 0.5), fontsize=5
            )
            # plt.setp(ax.get_legend().get_texts(), fontsize="5")
            for markers in lgnd.legendHandles:
                markers._sizes = [10]

    # hide last axis
    axis[1, 1].axis("off")

    if save_figure:
        fig.savefig(
            os.path.join(save_path, f"{prefix}.pdf"),
            dpi=100,
            bbox_inches="tight",
        )
        fig.savefig(
            os.path.join(save_path, f"{prefix}.png"),
            dpi=100,
            bbox_inches="tight",
        )
    if draw:
        plt.show()
    else:
        plt.close(fig)

# %% PATHS and DEFINITION OF WHAT TO PLOT

PATH_TO_MAIN_FEATURES_DIRECTORY = '/local/data2/iulta54/Data/BTB/histology_features' # this is the folder containing the subject or patient level features, from all the feature extractor available. It is used to build the feature path given the setting of what to plot.
PATH_TO_DATASET_DESCRIPTION = '/local/data1/iulta54/Code/BTB_DEEP_LEARNING/dataset_csv_file/BTB_AGGREGATED_CLINICAL_AND_WSI_INFORMATION_KS_LK_GOT_UM_LUND_UPP_ANONYM_20240704.csv' # path to the .csv file describing the mapping between classes and feature IDs

# what to plot
feature_level = 'patient'
feature_extractor = 'vit_conch'
classification_level = 'WHO_TUMOR_CATEGORY'
magnification = 20
patch_size = 224

# plot settings
nbr_random_features_to_plot_per_sample = -1 # specify the number of features per sample to take from the bag of features. This can be set to -1 to load all the features (can be computationally expensive). 

# build path to the feature folder using the given settings 
PATH_TO_FEATURE_DIRECTORY = os.path.join(PATH_TO_MAIN_FEATURES_DIRECTORY, f'{feature_level}_level_features', f'clam_features_mag_x{magnification}_size_{patch_size}',feature_extractor)

# load dataset csv file
dataset_description = pd.read_csv(PATH_TO_DATASET_DESCRIPTION)
# add patient level ID from the WSI id if feature_level == patient
if feature_level == 'patient':
    dataset_description['ANONYMIZED_CODE'] = dataset_description.apply(lambda x : '_'.join(x.ANONYMIZED_CODE.split('_')[0:3]), axis=1)

# make save path for the plotting
SAVE_PATH = '/local/data1/iulta54/Code/BTB_DEEP_LEARNING/outputs'
pathlib.Path(os.path.join(SAVE_PATH, f'feature_clustering_plot_{datetime.now().strftime("t%H%M%S")}'))
# %% PRINT THE NUMBER OF SAMPLES FOUND IN THE FEATURE FOLDER

list_of_feature_files = os.listdir(os.path.join(PATH_TO_FEATURE_DIRECTORY, 'pt_files'))
list_of_feature_files = [os.path.join(PATH_TO_FEATURE_DIRECTORY, 'pt_files',f) for f in list_of_feature_files]

print(f'Found {len(list_of_feature_files)} features to plot.')

# %% LOAD THE FEATURES and GATHER FOR DIMENSIONALITY REDUCTION  

list_features_class = []

for feature_file in tqdm.tqdm(list_of_feature_files):
    # load .pt file
    feature_tensor = torch.load(feature_file)
    # select nbr_random_features_to_plot_per_sample
    if nbr_random_features_to_plot_per_sample != -1:
        instance_feature_index = np.random.randint(low=0, high=feature_tensor.shape[0], size=nbr_random_features_to_plot_per_sample)
    else:
        instance_feature_index = list(range(feature_tensor.shape[0]))
    
    # get the instance level features using the index
    instance_feature_array = feature_tensor[instance_feature_index].numpy()
    instance_feature_array = np.mean(instance_feature_array,axis=0)

    # get the class for this feature file using the dataset_description and the classification_level
    feature_id = os.path.basename(feature_file).split('.')[0]
    class_of_feature = dataset_description.loc[dataset_description.ANONYMIZED_CODE == feature_id][classification_level].values[0]
    
    # save
    # list_features_class.extend([[f, class_of_feature] for f in instance_feature_array])
    list_features_class.extend([[instance_feature_array, class_of_feature]])

# %% GET TRANSFORM FOR DIMENSIONALITY REDUCTION - using PCA, but tSNE or others can be applied as well

# get PCA transform
pca = PCA(n_components=3)
pca_training = pca.fit_transform(np.vstack([f[0] for f in list_features_class]))

# %% APPLY TRANSFORMATION
# apply transform
projected_features = pca.transform(np.vstack([f[0] for f in list_features_class]))

# %% PLOT 

plot_PCA_embeddings(
                projected_features,
                hue_labels=[str(f[1]) for f in list_features_class],
                style_labels=None,
                draw=True,
                save_figure=False,
                save_path=None,
                prefix=None,
                nbr_legend_columns=1,
                value_ranges=None,
                marker_size=15,
            )

# %% tSNE

projected_features = TSNE(n_components=3, learning_rate='auto',
                  init='random', perplexity=3).fit_transform(np.vstack([f[0] for f in list_features_class]))

# %% 

plot_PCA_embeddings(
                projected_features,
                hue_labels=[str(f[1]) for f in list_features_class],
                style_labels=None,
                draw=True,
                save_figure=False,
                save_path=None,
                prefix=None,
                nbr_legend_columns=1,
                value_ranges=None,
            )