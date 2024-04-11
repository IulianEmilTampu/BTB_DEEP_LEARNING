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

# local imports 
from utils import plotConfusionMatrix, get_performance_metrics, plotROC

# %% PATHS

DATASET_DESCRIPTION_CSV = '/flush/iulta54/Research/P11-BTB_DEEP_LEARNING/dataset_csv_file/BTB_csv_for_training/dataset_summary/dataset_description_tumor_family_rep_0_folds_10.csv'
OUTPUT_TRAINING_DIR = '/flush/iulta54/Research/P11-BTB_DEEP_LEARNING/outputs/classification/BTB_tumor_family_clam_vit_hipt_s29122009'
SAVE_PATH = Path(OUTPUT_TRAINING_DIR, 'summary_evaluation')
SAVE_PATH.mkdir(parents=True, exist_ok=True)

# load dataset_description
dataset_description = pd.read_csv(DATASET_DESCRIPTION_CSV)

# %% INFERE NBR FOLDS, CLASSES and LABEL_DICT

# nbr folds
nbr_folds = len(list(glob.glob(os.path.join(OUTPUT_TRAINING_DIR, '*.pkl'))))

# nbr classes
nbr_classes = len(pd.unique(dataset_description.label))
unique_classes = sorted(list(pd.unique(dataset_description.label)))


# label dict
if 'label_integer' in dataset_description.columns:
    label_dict = dict([(pd.unique(dataset_description.loc[dataset_description.label_integer==i].label)[0], i) for i in range(nbr_classes)])
else:
    label_dict = dict([(unique_classes[i], i) for i in range(len(unique_classes))])

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

    # # print metrics
    # print('\n\nThe metrics for fold ' + str(i) + ' are:\n')
    # print('Precision: ' + str(metric_dict['precision']))
    # print('Recall: ' + str(metric_dict['recall']))
    # print('Accuracy: ' + str(metric_dict['accuracy']))
    # print('F1-score: ' + str(metric_dict['f1-score']))
    # print('Overall accuracy: ' + str(metric_dict['overall_accuracy']))
    # print('Macro-average precision: ' + str(metric_dict['overall_precision']))
    # print('Micro-average precision: ' + str(metric_dict['micro_avg_precision']))
    # print('Macro-average recall: ' + str(metric_dict['overall_recall']))
    # print('Micro-average recall: ' + str(metric_dict['micro_avg_recall']))
    # print('Macro-average F1-score: ' + str(metric_dict['overall_f1-score']))
    # print('Micro-average F1-score: ' + str(metric_dict['micro_avg_f1-score']))
    # print('MCC: ' + str(metric_dict['matthews_correlation_coefficient']))
    
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

# %% COMPUTE AVERAGE OVER THE FOLDS

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
for values, what in zip((acc_list, mcc_list, AUC_list), ('Accuracy', 'MCC', 'AUC')):
    median_fold_index = np.argsort(values)[len(values)//2]
    print(f'{what:10s}: {np.mean(values):0.4f}{pm_symbol}{np.std(values):0.4f} (range [{np.min(values):0.4f}, {np.max(values):0.4f}], median at fold: {median_fold_index})')

# per class f1 scores
print('Per-class F1 scores')
for c, v in label_dict.items():
    values = np.array(f1_lists)[:,v]
    print(f'    {c:45s}: {np.mean(values):0.4f}{pm_symbol}{np.std(values):0.4f} (range [{np.min(values):0.4f}, {np.max(values):0.4f}])')

# %% 