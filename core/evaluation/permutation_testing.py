# %% IMPORTS
import pathlib
import itertools
import numpy as np
import pandas as pd
from scipy.stats import permutation_test

# %% UTILITIES

def statistic(x, y, axis):
    return np.mean(x, axis=axis) - np.mean(y, axis=axis)

# %% PATHS

AGGREGATED_CSV_FILE = '/local/data1/iulta54/Code/BTB_DEEP_LEARNING/outputs/2024_05_01_npb_trainings/aggregated_evaluation_20240507.csv'
TIME_STAMP = pathlib.Path(AGGREGATED_CSV_FILE).parts[-1].split('.')[0].split('_')[-1]
SAVE_PATH = pathlib.Path(os.path.join(os.path.dirname(AGGREGATED_CSV_FILE), f'permutation_testing_{TIME_STAMP}'))
SAVE_PATH.mkdir(parents=True, exist_ok=True)

# load aggregated file
summary_evaluation_df = pd.read_csv(AGGREGATED_CSV_FILE)

# %% RUN PERMUTATION TESTING TO COMPARE THE PERFORMANCE BETWEEN MODELS (MODEL ONE - VS - MODEL TWO).

# specify the settings of the permutation test
permutation_type = 'samples'
alternative = 'two-sided'
nbr_permutations = 10000

# specify at which classification level to perform the comparison
classification_level_for_comparison = ["tumor_category", "tumor_family", "tumor_type"]

# specify which models to compare
models_to_compare = ['resnet50', 'vit_uni']
list_of_comparisons = list(itertools.combinations(models_to_compare, 2))

# # select version of the model to use (select between the models trained using abmil, clam_sb, clam_mb)
model_version_strategy = 'max'

# specify which metrics to compare
metric_and_text = {
    "mcc": {
        "metric_name": "Matthews correlation coefficient [-1,1]",
        "metric_range": [0, 1],
    },
    "auc": {"metric_name": "AUC (One-vs-Rest) [0,1]", "metric_range": [0, 1]},
    "balanced_accuracy": {"metric_name": "Balanced accuracy [0,1]", "metric_range": [0, 1]},
    "f1-score": {"metric_name": "F1 score (weighted) [0,1]", "metric_range": [0, 1]},
}

# some text formatting settings
pm_symbol = f" \u00B1 "
float_format='0.2'
new_line_symbol='<br>'

# define a pandas dataframe where the test results will be saved. 
# Using a dataframe to be able to save the results into a markdown table.
test_results_df = []
temp_df_row = dict.fromkeys(['classification_level' ,'compared_models', 'metric', 'population_1_stats', 'population_2_stats', 'test_result','nbr_samples_in_population','nbr_permutations'])

# loop through the different metrics, select the model version for the two models to compare and run permutation test.
for cl in classification_level_for_comparison:
    for metric in metric_and_text.keys():
        for comparison in list_of_comparisons:
            populations = []
            model_version_list = []
            for model in comparison:
                # get the model version based on the specification
                aus_model_df = summary_evaluation_df.loc[(summary_evaluation_df.features == model) & (summary_evaluation_df.classification_level == cl)]
                model_version_list.append(aus_model_df.groupby(['aggregation']).agg({metric: 'mean'}).idxmax()[0])
                # get the population
                populations.append(np.array(aus_model_df.loc[aus_model_df.aggregation == model_version_list[-1]][metric]))
            
            # perform comparison between the two populations
            population_1, population_2 = populations[0], populations[1]
            res = permutation_test((population_1, population_2), statistic, vectorized=True, permutation_type=permutation_type,
                       n_resamples=nbr_permutations, alternative=alternative, axis=0)
            
            # save information in the summary df
            test_results_df.append(
                {
                    'classification_level':cl,
                    'metric': metric,
                    'compared_models': f'{comparison[0]}_{model_version_list[0]}{new_line_symbol}vs{comparison[1]}_{model_version_list[1]}',
                    f'population_1_stats{new_line_symbol}mean{pm_symbol}std{new_line_symbol}[0.05, 0.95] q.': f'{comparison[0]}_{model_version_list[0]}{new_line_symbol}{populations[0].mean():{float_format}f}{pm_symbol}{populations[0].std():{float_format}f}{new_line_symbol}[{np.quantile(populations[0], 0.05):{float_format}f}, {np.quantile(populations[0], 0.95):{float_format}f}]',
                    f'population_2_stats{new_line_symbol}mean{pm_symbol}std{new_line_symbol}[0.05, 0.95] q.': f'{comparison[1]}_{model_version_list[1]}{new_line_symbol}{populations[1].mean():{float_format}f}{pm_symbol}{populations[1].std():{float_format}f}{new_line_symbol}[{np.quantile(populations[1], 0.05):{float_format}f}, {np.quantile(populations[1], 0.95):{float_format}f}]',
                    'statistic': f'{res.statistic:0.5f}',
                    'p-value': f'{res.pvalue:0.6f}',
                    'nbr_samples_in_population': len(populations[0]),
                    'nbr_permutations': nbr_permutations,
                    'permutation_type' : permutation_type,
                    'alternative' :alternative,
                }
            ) 

# %% save permutation test results to markdown table

test_results_df = pd.DataFrame(test_results_df)
print(test_results_df.to_markdown(tablefmt="pipe", stralign='center'))
with open(os.path.join(SAVE_PATH, 'table_summary_permutation_test.md'), 'w') as f:
    print(test_results_df.to_markdown(tablefmt="pipe", stralign='center'), file=f)

# %%
