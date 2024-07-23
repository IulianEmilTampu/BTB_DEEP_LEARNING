# %% 
import os 
import glob 
import pandas as pd 
from datetime import datetime

# %% PATHS
print('Aggregation of summary_evaluation.csv files.')
OUTPUT_TRAINING_DIRS = '/local/data1/iulta54/Code/BTB_DEEP_LEARNING/outputs/2024_07_07'
SAVE_PATH = OUTPUT_TRAINING_DIRS

# %% SCOUT THE OUTPUT_TRAINING_DIRS for summary_evaluation.csv files

summary_evaluation_dfs = []
for path, subdirs, files in os.walk(OUTPUT_TRAINING_DIRS):
    for name in files:
        if 'summary_evaluation.csv' in os.path.basename(name):
            summary_evaluation_dfs.append(pd.read_csv(os.path.join(path, name)))

print(f'Found {len(summary_evaluation_dfs)} evaluation files.')

# make dataframe
summary_evaluation_df = pd.concat(summary_evaluation_dfs, axis=0, ignore_index=True)

# %% REFINE 

'''
Classification level flag
    Category classification	-> category
    Family classification -> family
    Type classification -> type
 '''

def refine_classification_level(x):
    if any([x.classification_level == 'Category classification', x.classification_level == 'tumor_category']):
        return 'tumor_category'
    elif any([x.classification_level == 'Family classification',x.classification_level == 'tumor_family']):
        return 'tumor_family'
    elif any([x.classification_level == 'Type classification', x.classification_level == 'tumor_type']):
        return 'tumor_type'
    else:
        return None

summary_evaluation_df['classification_level'] = summary_evaluation_df.apply(lambda x: refine_classification_level(x), axis=1)

# %% SAVE

aggregated_file_path = os.path.join(SAVE_PATH, f'aggregated_evaluation_{datetime.now().strftime("%Y%m%d")}.csv')
summary_evaluation_df.to_csv(aggregated_file_path)
print(f'Aggregated file save as: {aggregated_file_path}')
# %%