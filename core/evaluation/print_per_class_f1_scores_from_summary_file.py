# %% IMPORTS
''' 
Utility that given the .csv summary file, prints a table with the per-class metrics (f1-score, precision, recall). 
'''

import os 
import pandas as pd 
import pathlib
import numpy as np 
import scipy.stats as st

# local imports
from utils import aggregate_evaluation_for_metric


# %% PATHS

AGGREGATED_CSV_FILE = '/local/data1/iulta54/Code/BTB_DEEP_LEARNING/outputs/2024_07_07/aggregated_evaluation_20240716.csv'
TIME_STAMP = pathlib.Path(AGGREGATED_CSV_FILE).parts[-1].split('.')[0].split('_')[-1]
SAVE_PATH = pathlib.Path(os.path.join(os.path.dirname(AGGREGATED_CSV_FILE), f'per_class_metrics_abmil_clam_{TIME_STAMP}'))
SAVE_PATH.mkdir(parents=True, exist_ok=True)

# load aggregated file
summary_evaluation_df = pd.read_csv(AGGREGATED_CSV_FILE)

# %% FOR EACH OF THE METRICS and FOR EACH OF THE CLASSIFICATION TASKS, FEATURE EXTRACTOR AND AGGREGATION
# groupby the dataset and apply average over the different columns

aggregated_information = summary_evaluation_df.drop(['classes','class_fractions', 'model', 'set'], axis=1).groupby(['classification_level','features', 'aggregation']).agg(lambda x : aggregate_evaluation_for_metric(x))

# remove columns that are not of interest
aggregated_information = aggregated_information.drop(['nbr_classes','repetition', 'fold', 'mcc', 'balanced_accuracy', 'accuracy', 'auc', 'f1-score'], axis=1)

print(aggregated_information.to_markdown(tablefmt="pipe", stralign='center'))

# %% SAVE TO FILE 
with open(os.path.join(SAVE_PATH, 'table_summary_per_class_evaluation.md'), 'w') as f:
    print(aggregated_information.to_markdown(tablefmt="pipe", stralign='center'), file=f)


# %%
