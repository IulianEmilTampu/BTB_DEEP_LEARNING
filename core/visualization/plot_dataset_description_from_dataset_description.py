# %% 
'''
Plots the class distribution given the dataset_description.csv file
'''

import os 
import pathlib
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 

# %% PATHS

WHAT = 'tumor_category'
DATASET_DESCRIPTION_CSV = f'/local/data2/iulta54/Data/BTB/experiments_csv_files/patient_level_features/nonparametric_bootstraping/{WHAT}/all/dataset_descriptor.csv'
DATASET_DESCRIPTION_CSV = '/local/data2/iulta54/Data/BTB/experiments_csv_files/patient_level_features/test_20240723_counts/splits_tumor_category_npb_wsi_features/dataset_descriptor.csv'
SAVE_PATH = '/local/data1/iulta54/Code/BTB_DEEP_LEARNING/outputs'
SAVE_PATH = pathlib.Path(SAVE_PATH, 'dataset_summary_plots_tumor_category_20240723')
SAVE_PATH.mkdir(parents=True, exist_ok=True)

# load csv
dataset_description = pd.read_csv(DATASET_DESCRIPTION_CSV)
dataset_description = dataset_description.rename(columns={'case_id':'cases', 'slide_id':'WSIs'})

# %% PLOT CLASS DISTRIBUTION BASED ON CASE_ID (SUBJECT)

PLOT_NAME = f'{WHAT}_subject_and_WSIs_count' 
SAVE_PLOT = True

# re-arrange dataset_summary to contain total counts
dataset_summary_for_plot = dataset_description.groupby('label').agg({'cases': lambda x: len(pd.unique(x)), 'WSIs': 'count'}).reset_index().melt(['label'], var_name='COUNT_OVER', value_name='COUNTS')

# define orders
hue_var = 'COUNT_OVER'
hue_order = ['cases', 'WSIs']
col_order = list(pd.unique(dataset_summary_for_plot.label))
y_var = 'COUNTS'
col_var = 'label'

# build figure
plt.figure(figsize=(80, 10))
g = sns.catplot(
    dataset_summary_for_plot, kind="bar",
    x="COUNT_OVER", y="COUNTS", col="label",
    height=4, aspect=.5,
    hue_order=hue_order,
    col_order=col_order,
)
# fix axis limits
g.set(ylim=(0, dataset_summary_for_plot.COUNTS.max()+50), ylabel='Count', xlabel='Count over')

# add counts and fraction over the entire dataset
# with g.axes[0, n] we loop though the different SITES
width_bar_plot = 0.8
text_versical_offset_count = 8 * len(col_order)
text_vetical_offset_fraction = 4

for col_indx, col in enumerate(col_order):
    ax = g.axes[0, col_indx] # this looks at the bar plots of a class
    # add values for the different hue values (case_id and slide_id)
    for hue_indx, hue in enumerate(hue_order):
        # get value for this bar
        count = dataset_summary_for_plot.loc[(dataset_summary_for_plot[col_var]==col) & (dataset_summary_for_plot[hue_var]==hue)].COUNTS.values[0]
        fraction_over_all_sites = count / dataset_summary_for_plot.loc[dataset_summary_for_plot[hue_var]==hue].COUNTS.sum()*100
        # the x position in the axix is just the index of the hue
        ax.text(hue_indx, count+text_versical_offset_count, f"{count}", ha="center", weight="bold")
        ax.text(hue_indx, count+text_vetical_offset_fraction, f"({fraction_over_all_sites:0.1f}%)", ha="center")
        # 
        ax.set_title(col,fontsize=5)
# add legend with the information about the total number of subjects and 
ax = g.axes[0, 0]
textstr = "\n".join([f"Counts over all labels",     
            f"{'Nbr cases*':12s}: {dataset_summary_for_plot.loc[dataset_summary_for_plot.COUNT_OVER=='cases'].COUNTS.sum()}",
            f"{'Nbr WSIs':12s}: {dataset_summary_for_plot.loc[dataset_summary_for_plot.COUNT_OVER=='WSIs'].COUNTS.sum()}",
            f"* cases == unique pairs subject-diagnosis",
            ])

text_box_props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
x_pos = 0
y_pos = dataset_summary_for_plot.loc[(dataset_summary_for_plot[col_var]==col_order[-1]) & (dataset_summary_for_plot[hue_var]==hue_order[-1])].COUNTS.values[0] - 150
ax.text(x_pos, y_pos, textstr, fontsize=15, bbox=text_box_props, verticalalignment='top')

# save
if SAVE_PLOT:
    plt.savefig(
        fname=os.path.join(SAVE_PATH, f"{PLOT_NAME}.pdf"),
        dpi=200,
        format="pdf",
        bbox_inches='tight',
    )

    plt.savefig(
        fname=os.path.join(SAVE_PATH, f"{PLOT_NAME}.png"),
        dpi=200,
        format="png",
        bbox_inches='tight'
    )

# %% PLOT MALE, FEMALE, NA 

FULL_DATASET_DESCRIPTION_PATH = '/local/data1/iulta54/Code/BTB_DEEP_LEARNING/dataset_csv_file/BTB_AGGREGATED_CLINICAL_AND_WSI_INFORMATION_KS_LK_GOT_UM_LUND_UPP_ANONYM_20240704.csv'
full_dataset_summary = pd.read_csv(FULL_DATASET_DESCRIPTION_PATH, encoding="ISO-8859-1")

# add to the training dataset descriptor the gender information
dataset_description['GENDER'] = dataset_description.apply(lambda x : full_dataset_summary.loc[full_dataset_summary.ANONYMIZED_CODE == x.WSIs].GENDER.values[0], axis=1)
# and age
dataset_description['AGE_YEARS'] = dataset_description.apply(lambda x : full_dataset_summary.loc[full_dataset_summary.ANONYMIZED_CODE == x.WSIs].AGE_YEARS.values[0], axis=1)
dataset_description['AGE_MONTHS'] = dataset_description.apply(lambda x : full_dataset_summary.loc[full_dataset_summary.ANONYMIZED_CODE == x.WSIs].AGE_MONTHS.values[0], axis=1)

# re-arrange dataset_summary to contain total counts
dataset_summary_for_plot = dataset_description.groupby('label').agg({'cases': lambda x: len(pd.unique(x)), 'WSIs': 'count'}).reset_index().melt(['label'], var_name='COUNT_OVER', value_name='COUNTS')
