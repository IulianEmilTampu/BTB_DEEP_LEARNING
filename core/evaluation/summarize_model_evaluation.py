# %%
import pandas as pd
from tqdm import tqdm
import math
import pickle
import sklearn.metrics as metrics
import os
import glob
import numpy as np
import itertools
import pathlib
from matplotlib import pyplot as plt
import matplotlib.colors as colors
from pathlib import Path
import json
import hydra
from datetime import datetime
import omegaconf
from omegaconf import DictConfig, open_dict, OmegaConf

# local imports 
from utils import plotConfusionMatrix, get_performance_metrics, plotROC

# %% UTILITIES

def check_model_integrity(trained_model_dir:str):
    ''' 
    Utility that checks that all the folds have been trained and that the split results are available.
    '''
    if not os.path.isdir(trained_model_dir):
        print(f'Not a model directory. Given {trained_model_dir}')
        return None
    
    # open the hydra_config file and get the expected numbers of folds
    training_cfg = pathlib.Path(trained_model_dir, f'hydra_config.yaml')
    if os.path.isfile(training_cfg):
        training_cfg = omegaconf.OmegaConf.load(pathlib.Path(trained_model_dir, f'hydra_config.yaml'))
        expected_nbr_folds = training_cfg.k

        # get the number of result.pkl files
        pkl_files_check = [os.path.isfile(pathlib.Path(trained_model_dir, f'split_{i}_results.pkl')) for i in range(expected_nbr_folds)]

        if all(pkl_files_check):
            return trained_model_dir
        else:
            raise ValueError(f'Not all expected folds ({expected_nbr_folds}) have available results.pkl files. Missing {expected_nbr_folds - sum(pkl_files_check)}.\nModel: {trained_model_dir}')
    else:
        print(f'Given model folder missing hydra_config.yaml file. Given {trained_model_dir}')
        return None
# %% MAIN
@hydra.main(
    version_base="1.2.0", config_path=os.path.join('../../configs', 'evaluation'), config_name='default'
)
def main(cfg:DictConfig):
    # get list of models to evaluate, by checking if the given trained_model_dir contains one model or several models
    if not os.path.isdir(cfg.trained_model_dir):
        raise ValueError(f'The given trained_model_dir does not exist.')
    else:
        process_stack = []
        # check if the directory contains .pt files
        if any(['.pt' in f for f in os.listdir(cfg.trained_model_dir)]):
            print(f'Evaluating a single model.')
            # check that there are all the fold models are is_available
            check_model_integrity(cfg.trained_model_dir)
            # if passes the check, the model path is added to the process stack
            process_stack.append(check_model_integrity(cfg.trained_model_dir))
        else:
            # check all the models in the folder
            for d in os.listdir(cfg.trained_model_dir):
                process_stack.append(check_model_integrity(pathlib.Path(cfg.trained_model_dir,d)))
            # remove Nones
            process_stack = [x for x in process_stack if x is not None]
            
    print(f'Found {len(process_stack)} models to evaluate.')

    # % define plotting configuration

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

    # % if aggregate_evaluation_csv_files is True and there is more than one model, aggregate csv files

    if cfg.aggregate_evaluation_csv_files and len(process_stack) > 1:
        for_aggregation = []

    # % loop through the process stack and evaluate
    with tqdm(total=len(process_stack), unit='model') as model_tqdm:
        for trained_model_dir in process_stack:
            # create save path 
            SAVE_PATH = Path(trained_model_dir, 'summary_evaluation')
            SAVE_PATH.mkdir(parents=True, exist_ok=True)

            # load training configuration
            training_cfg = omegaconf.OmegaConf.load(pathlib.Path(trained_model_dir, f'hydra_config.yaml'))
            nbr_folds = training_cfg.k

            # get classes and label_dict
            nbr_classes = training_cfg.task.n_classes
            label_dict = training_cfg.task.label_dict
            unique_classes = list(label_dict.keys())

            # load the dataset_descriptor .csv file (using the task information. TODO make it not dependent on the local machine by loading the local paths)
            dataset_description = pd.read_csv(training_cfg.task.csv_path)
            class_fractions = [sum(dataset_description.label == l)/len(dataset_description) for l in unique_classes]

            # % EVALUATE EACH SPLIT and SAVE PERFORMANCES  
            # aggregated metrics
            prec_lists = []
            rec_lists = []
            f1_lists = []
            averaged_prec = []
            averaged_rec = []
            averaged_f1 = []
            averaged_AUC_list = []
            acc_list = []
            mcc_list = []
            class_acc_list = []
            balanced_acc = []

            with tqdm(total=nbr_folds, unit='folds', leave=False) as fold_tqdm:
                for i in range(nbr_folds):
                    # open summary file
                    evaluation_file = pathlib.Path(trained_model_dir, f'split_{i}_results.pkl')
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
                    averaged_prec.append(metric_dict['overall_precision'])
                    averaged_rec.append(metric_dict['overall_recall'])
                    averaged_f1.append(metric_dict['overall_f1-score'])
                    class_acc_list.append(metric_dict['accuracy'])
                    acc_list.append(np.mean(metric_dict['overall_accuracy']))
                    mcc_list.append(metric_dict['matthews_correlation_coefficient'])
                    averaged_AUC_list.append(metric_dict['overall_auc'])
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
                    
                    fold_tqdm.update()
            
            # % SAVE SUMMARY IN A .csv FILE THAT CAN BE AGGREGATED WITH ALL THE OTHER RUNS

            summary_df = []

            # gather information
            for f in range(nbr_folds):

                temp_dict = {
                'model': 'clam' if 'model_type' in training_cfg.keys() else 'hipt',
                'aggregation': training_cfg.model_type if 'model_type' in training_cfg.keys() else 'abmil',
                'features': training_cfg.feature_extractor,
                'classification_level': training_cfg.task.description,
                'classes': list(label_dict),
                'nbr_classes': len(label_dict),
                'class_fractions': class_fractions,
                'repetition': os.path.basename(training_cfg.task.split_dir).split('_')[-1],
                'fold':f,
                'set': 'test',
                'mcc':mcc_list[f],
                'balanced_accuracy': balanced_acc[f],
                'accuracy': acc_list[f],
                'auc': averaged_AUC_list[f],
                'f1-score': averaged_f1[f],
                }
                
                # add the per class f1, precision and recall in separate columns
                for c in range(nbr_classes):
                    for values, what in zip((f1_lists, prec_lists, rec_lists), ('f1', 'precision', 'recall')):
                        temp_dict[(f'class_{c}_{what}')] = values[f][c]

                summary_df.append(temp_dict)

            summary_df = pd.DataFrame(summary_df)

            # save 
            summary_df.to_csv(os.path.join(SAVE_PATH, 'summary_evaluation.csv'))
            
            # save for aggregation
            if cfg.aggregate_evaluation_csv_files and len(process_stack) > 1:
                for_aggregation.append(summary_df)
            
            if cfg.print_single_model_summary:
                # % PRINT AVERAGE OVER THE FOLDS
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
                for values, what in zip((acc_list, mcc_list, averaged_AUC_list, balanced_acc), ('Accuracy', 'MCC', 'AUC', 'Balanced accuracy')):
                    median_fold_index = np.argsort(values)[len(values)//2]
                    print(f'{what:17s}: {np.mean(values):0.4f}{pm_symbol}{np.std(values):0.4f} (range [{np.min(values):0.4f}, {np.max(values):0.4f}], median at fold: {median_fold_index})')

                # per class f1 scores
                print('Per-class F1 scores')
                for c, v in label_dict.items():
                    values = np.array(f1_lists)[:,v]
                    print(f'    {c:45s}: {np.mean(values):0.4f}{pm_symbol}{np.std(values):0.4f} (range [{np.min(values):0.4f}, {np.max(values):0.4f}])')

            model_tqdm.update()

    # % save aggregation 
    if cfg.aggregate_evaluation_csv_files and len(process_stack) > 1:
        summary_evaluation_df = pd.concat(for_aggregation, axis=0, ignore_index=True)

    aggregated_file_path = os.path.join(cfg.aggregation_save_dir, f'aggregated_evaluation_{datetime.now().strftime("%Y%m%d")}.csv')
    summary_evaluation_df.to_csv(aggregated_file_path)
    print(f'Aggregated file save as: {aggregated_file_path}')

# %% 

if __name__ == "__main__":
    main()
    print("finished!")
    print("end script")