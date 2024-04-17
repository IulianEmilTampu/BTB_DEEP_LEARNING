# %%
import pandas as pd
import math
import pickle
import sklearn.metrics as metrics
import os
import glob
import numpy as np
import itertools
from matplotlib import pyplot as plt
import matplotlib.colors as colors
from pathlib import Path
import json

# local imports 
from utils import plotConfusionMatrix, get_performance_metrics, plotROC

# %% PATHS

DATASET_DESCRIPTION_CSV = '/local/data1/iulta54/Code/BTB_DEEP_LEARNING/outputs/BTB_csv_for_training/dataset_summary/dataset_description_tumor_category_rep_0_folds_5.csv'
OUTPUT_TRAINING_DIR = '/local/data1/iulta54/Code/BTB_DEEP_LEARNING/outputs/BTB_tumor_category_clam_sb_vit_uni_mag_x20_size_224_s29122009'
SAVE_PATH = Path(OUTPUT_TRAINING_DIR, 'summary_evaluation')
SAVE_PATH.mkdir(parents=True, exist_ok=True)

# load dataset_description
dataset_description = pd.read_csv(DATASET_DESCRIPTION_CSV)

# %% LOAD EXPERIMENT SETTING FILE (.txt or .yaml)

experiment_settings_file = glob.glob(os.path.join(OUTPUT_TRAINING_DIR, 'experiment_*'))[0]

if experiment_settings_file.endswith('.txt'):
    import ast

    with open(experiment_settings_file) as f:
        for line in f:
            experiment_settings = ast.literal_eval(line)
elif experiment_settings_file.endswith('.yaml'):
    import yaml
    experiment_settings = yaml.safe_load(Path(experiment_settings_file).read_text())
else:
    raise ValueError('Training settings file could not be loaded.')

# %% INFERE NBR FOLDS, CLASSES and LABEL_DICT

# nbr folds
nbr_folds = len(list(glob.glob(os.path.join(OUTPUT_TRAINING_DIR, '*.pkl'))))
# assert nbr_folds == experiment_settings['num_splits']

# nbr classes
nbr_classes = len(pd.unique(dataset_description.label))
unique_classes = sorted(list(pd.unique(dataset_description.label)))
class_fractions = [sum(dataset_description.label == l)/len(dataset_description) for l in unique_classes]
# assert nbr_classes == len(experiment_settings['task']['label_dict'])

# label dict
# if 'label_integer' in dataset_description.columns:
#     label_dict = dict([(pd.unique(dataset_description.loc[dataset_description.label_integer==i].label)[0], i) for i in range(nbr_classes)])
# else:
#     label_dict = dict([(unique_classes[i], i) for i in range(len(unique_classes))])
label_dict = experiment_settings['task']['label_dict']

# %% EVALUATE EACH SPLIT and SAVE PERFORMANCES  

# aggregated metrics
prec_lists = []
rec_lists = []
f1_lists = []
mac_prec = []
mic_prec = []
mac_rec = []
mic_rec = []
mac_f1 = []
mic_f1 = []
AUC_list = []
acc_list = []
mcc_list = []
class_acc_list = []
balanced_acc = []

# figure settings
roc_figure_setting = {
        'average_roc_line_width':4,
        'average_roc_line_style':':',
        'per_class_roc_line_width': 2,
        'per_class_roc_line_style' : '-',
        'xlabel_font_size':20,
        'ylabel_font_size':20,
        'title_font_size':20,
        'legend_font_size' : 8,
    }
cm_figure_settings = {
        'pred_count_font_size': 10,
        'xlabel_font_size':10,
        'xticks_font_size':10,
        'xticks_rotation':45, 
        'xticks_horizontal_alignment':'right',
        'ylabel_font_size':10,
        'yticks_font_size':10,
        'title_font_size':12,
        'legend_font_size' : 12,
    }

for i in range(nbr_folds):
    print(f'Working on fold {i+1:2d}/{nbr_folds}')
    # open summary file
    evaluation_file = os.path.join(OUTPUT_TRAINING_DIR, f'split_{i}_results.pkl')
    with open(evaluation_file, 'rb') as f:
        results = pickle.load(f)
        res_df = pd.DataFrame.from_dict(results, orient='index')

    # get logits, prediction and labels
    probs = np.vstack([res_df.iloc[i].prob[0] for i in range(len(res_df))])
    preds = np.argmax(probs, axis=-1)
    labels = np.stack([res_df.iloc[i].label for i in range(len(res_df))])
    
    # # make one hot label encoding
    one_hot_labels = np.zeros_like(probs)
    one_hot_labels[np.arange(one_hot_labels.shape[0]),labels] = 1
    
    # get performance metrics
    GT = one_hot_labels
    PRED = probs
    metric_dict = get_performance_metrics(GT, PRED, average="macro")

    # save for aggregation
    prec_lists.append(metric_dict['precision'])
    rec_lists.append(metric_dict['recall'])
    f1_lists.append(metric_dict['f1-score'])
    mac_prec.append(metric_dict['overall_precision'])
    mic_prec.append(metric_dict['micro_avg_precision'])
    mac_rec.append(metric_dict['overall_recall'])
    mic_rec.append(metric_dict['micro_avg_recall'])
    mac_f1.append(metric_dict['overall_f1-score'])
    mic_f1.append(metric_dict['micro_avg_f1-score'])
    class_acc_list.append(metric_dict['accuracy'])
    acc_list.append(np.mean(metric_dict['overall_accuracy']))
    mcc_list.append(metric_dict['matthews_correlation_coefficient'])
    AUC_list.append(metric_dict['overall_auc'])
    balanced_acc.append(metric_dict['balanced_accuracy'])

    # plots
    plotConfusionMatrix(GT, PRED, classes=unique_classes,
                        figure_setting=cm_figure_settings,
                        compute_random_accuracy=True,
                        savePath=SAVE_PATH,
                        saveName='CF_k'+str(i),
                        draw=False)
    plotROC(GT, PRED, classes = unique_classes, figure_setting=roc_figure_setting,
             savePath=SAVE_PATH,
             saveName='ROC_k'+str(i))

# %% PRINT AVERAGE OVER THE FOLDS
pm_symbol = f" \u00B1 "    

# Class wise
# # k-fold mean
prec_avg = np.mean(np.array(prec_lists), axis=0)
rec_avg = np.mean(np.array(rec_lists), axis=0)
f1_avg = np.mean(np.array(f1_lists), axis=0)

# # k-fold sd
prec_sd = np.std(np.array(prec_lists), axis=0)
rec_sd = np.std(np.array(rec_lists), axis=0)
f1_sd = np.std(np.array(f1_lists), axis=0)


# print
print(f'Summary performance over {nbr_folds}.')
for values, what in zip((acc_list, mcc_list, AUC_list, balanced_acc), ('Accuracy', 'MCC', 'AUC', 'Balanced accuracy')):
    median_fold_index = np.argsort(values)[len(values)//2]
    print(f'{what:17s}: {np.mean(values):0.4f}{pm_symbol}{np.std(values):0.4f} (range [{np.min(values):0.4f}, {np.max(values):0.4f}], median at fold: {median_fold_index})')

# per class f1 scores
print('Per-class F1 scores')
for c, v in label_dict.items():
    values = np.array(f1_lists)[:,v]
    print(f'    {c:45s}: {np.mean(values):0.4f}{pm_symbol}{np.std(values):0.4f} (range [{np.min(values):0.4f}, {np.max(values):0.4f}])')

# %% SAVE SUMMARY IN A .csv FILE THAT CAN BE AGGREGATED WITH ALL THE OTHER RUNS

summary_df = []

# gather information
for f in range(nbr_folds):
    # get feature extractor from the name of where the features were saved
    feature_dir = experiment_settings['task']['data_root_dir']
    if '_resnet50' in feature_dir:
        features = 'resnet50'
    elif '_vit_hipt' in feature_dir:
        features = 'vit_hipt'
    elif '_vit_uni' in feature_dir:
        features = 'vit_uni'
    elif '_vit_conch' in feature_dir:
        features = 'vit_conch'
    else:
        features = None
    #

    temp_dict = {
    'model': 'clam' if 'model_type' in experiment_settings.keys() else 'hipt',
    'aggregation': experiment_settings['model_type'] if 'model_type' in experiment_settings.keys() else 'abmil',
    'features': features,
    'classification_level': experiment_settings['task']['description'],
    'classes': list(experiment_settings['task']['label_dict'].keys()),
    'nbr_classes': len(experiment_settings['task']['label_dict'].keys()),
    'class_fractions': class_fractions,
    'repetition': os.path.basename(experiment_settings['task']['split_dir']).split('_')[-1],
    'fold':f,
    'set': 'test',
    'mcc':mcc_list[f],
    'balanced_accuracy': balanced_acc[f],
    'accuracy': acc_list[f],
    'auc': AUC_list[f],
    }
    
    # add the per class f1, precision and recall in separate columns
    for c in range(nbr_classes):
        for values, what in zip((f1_lists, prec_lists, rec_lists), ('f1', 'precision', 'recall')):
            temp_dict[(f'class_{c}_{what}')] = values[f][c]

    summary_df.append(temp_dict)

summary_df = pd.DataFrame(summary_df)

# save 
summary_df.to_csv(os.path.join(SAVE_PATH, 'summary_evaluation.csv'))
# %% 