# %%

import os
import glob
import time
import csv
import json
import pathlib
import platform
import numpy as np
import pandas as pd
import seaborn as sns
from copy import deepcopy
from itertools import cycle
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
import matplotlib.patheffects as path_effects
from matplotlib.patches import Rectangle
import matplotlib.patheffects as pe
from scipy import stats
import itertools

print('Plotting summary models performance from aggregated .csv file.')

# %% UTILITIES
def make_summary_plot(df, plot_settings, df_ensemble=None):
    # check if the data is available for all the box plots
    expected_plots = [
        (m, p) for m in plot_settings["x_order"] for p in plot_settings["hue_order"]
    ]

    # check if we have the data for these combinations
    expected_plots_flags = []
    for c in expected_plots:
        # filter data based on the classification level and model. If it exists, flag as 1, else as 0
        if (
            len(
                df.loc[
                    (df[plot_settings["x_variable"]] == c[0])
                    & (df[plot_settings["hue_variable"]] == c[1])
                ]
            )
            == 0
        ):
            expected_plots_flags.append(0)
        else:
            expected_plots_flags.append(1)

    box_plot = sns.boxplot(
        x=plot_settings["x_variable"],
        y=plot_settings["metric_to_plot"],
        hue=plot_settings["hue_variable"],
        data=df,
        palette="Set3",
        hue_order=plot_settings["hue_order"],
        order=plot_settings["x_order"],
        ax=plot_settings["axis"] if plot_settings["axis"] else None,
    )
    # ### refine plot

    # ## title
    box_plot.set_title(
        plot_settings["figure_title"],
        fontsize=plot_settings["title_font_size"],
    )

    # ## fix axis
    # format y axis
    box_plot.set_ylim([plot_settings["ymin"], plot_settings["ymax"]])

    ylabels = [f"{x:,.2f}" for x in box_plot.get_yticks()]
    box_plot.set_yticklabels(ylabels, fontsize=plot_settings["y_axis_tick_font_size"])
    box_plot.set_ylabel(
        plot_settings["metric_to_plot_name"],
        fontsize=plot_settings["y_axis_label_font_size"],
    )

    # format x axis
    box_plot.set_xlabel("")
    box_plot.set_xticklabels(
        box_plot.get_xticklabels(),
        rotation=plot_settings["x_axis_tick_rotation"],
        fontsize=plot_settings["x_axis_tick_font_size"],
    )

    # ## add pattern to boxplots and legend
    hatches = []
    hatches.extend(
        plot_settings["available_hatches"][0 : len(plot_settings["hue_order"])]
        * len(plot_settings["x_order"])
    )
    legend_colors, legend_hatch = [], []

    # fix the boxplot patches to match the expected boxplots (and take care of those that are not present)
    list_of_boxplot_patches = [
        patch for patch in box_plot.patches if type(patch) == mpl.patches.PathPatch
    ][::-1]
    list_of_boxplot_patches_corrected = []
    for b in expected_plots_flags:
        if b == 1:
            list_of_boxplot_patches_corrected.append(list_of_boxplot_patches.pop())
        else:
            list_of_boxplot_patches_corrected.append(0)

    for i, patch in enumerate(list_of_boxplot_patches_corrected):
        if patch != 0:
            # Boxes from left to right
            hatch = hatches[i]
            patch.set_hatch(hatch)
            patch.set_edgecolor("k")
            patch.set_linewidth(1)
            # save color and hatch for the legend
            legend_colors.append(patch.get_facecolor())
            legend_hatch.append(hatch)


    # ## fix legend
    if plot_settings["plot_legend"]:
        legend_labels = [
            f"{plot_settings['hue_variable'].title()} = {l.get_text()}"
            for idx, l in enumerate(box_plot.legend_.texts)
        ]
        # make patches (based on hue)
        legend_patch_list = [
            mpl.patches.Patch(fill=True, label=l, hatch=h, facecolor=c, edgecolor="k")
            for l, c, h in zip(legend_labels, legend_colors, legend_hatch)
        ]
        # if the Ensemble value is showed, add this to the legend
        if plot_settings["show_ensemble"]:
            legend_patch_list.append(
                mlines.Line2D(
                    [],
                    [],
                    color="k",
                    marker="X",
                    linestyle="None",
                    markersize=20,
                    label="Ensemble",
                )
            )
        # remake legend
        box_plot.legend(
            handles=legend_patch_list,
            loc="best",
            handleheight=4,
            handlelength=6,
            labelspacing=1,
            fontsize=plot_settings["legend_font_size"],
            # bbox_to_anchor=(1.01, 1),
        )
    else:
        box_plot.get_legend().remove()

def make_summary_string(x):
    # get all the information needed
    cl = x.classification_level
    nc = x.nbr_classes
    model = x.model

    return f"{cl}\n({nc} classes)\n{model}"

# %% PATHS

AGGREGATED_CSV_FILE = '/flush/iulta54/Research/P11-BTB_DEEP_LEARNING/outputs/classification/aggregated_evaluation_20240412.csv'
TIME_STAMP = pathlib.Path(AGGREGATED_CSV_FILE).parts[-1].split('.')[0].split('_')[-1]
SAVE_PATH = pathlib.Path(os.path.join(os.path.dirname(AGGREGATED_CSV_FILE), f'plots_aggregated_evaluation_{TIME_STAMP}'))
SAVE_PATH.mkdir(parents=True, exist_ok=True)


# load aggregated file
summary_evaluation_df = pd.read_csv(AGGREGATED_CSV_FILE)

# %% PLOT
'''
Here for each of the classification_levels (category, family and type), plot the models performance as box-plots
'''
SAVE_FIGURE = True

# define the order in the plot
classification_level_order = ["category", "family", "type"]
nbr_classes = [pd.unique(summary_evaluation_df.loc[summary_evaluation_df.classification_level==c].nbr_classes)[0] for c in classification_level_order]
model_order = ["clam"]
hue_order = ['resnet50', 'vit_hipt', 'vit_uni']

# the x order to plot
x_order = [
    f"{cl}\n({nc} classes)\n{model}"
    for cl, nc in zip(classification_level_order, nbr_classes)
    for model in model_order
]


# add summary string to dataframe
summary_evaluation_df["summary_string"] = summary_evaluation_df.apply(
    lambda x: make_summary_string(x), axis=1
)


# plot and save for all the metrics
metric_and_text = {
    "mcc": {
        "metric_name": "Matthews correlation coefficient [-1,1]",
        "metric_range": [0, 1],
    },
    "auc": {"metric_name": "AUC [0,1]", "metric_range": [0, 1]},
    "balanced_accuracy": {"metric_name": "Balanced accuracy [0,1]", "metric_range": [0, 1]},
}

#  DEBUG
# metric_and_text = {
#     "mcc": {
#         "metric_name": "Matthews correlation coefficient [-1,1]",
#         "metric_range": [0, 1],
#     }
# }

for metric, metric_specs in metric_and_text.items():
    # refine plot settings
    plot_settings = {
        "metric_to_plot": metric,
        "metric_to_plot_name": metric_specs["metric_name"],
        "x_variable": "summary_string",
        "x_order": x_order,
        "hue_variable": "features",
        "hue_order": hue_order,
        "figure_title": "Summary performance",
        "show_ensemble": False,
        "show_ensemble_value": False,
        "tick_font_size": 12,
        "title_font_size": 20,
        "y_axis_label_font_size": 20,
        "y_axis_tick_font_size": 20,
        "x_axis_tick_font_size": 20,
        "x_axis_tick_rotation": 0,
        "legend_font_size": 12,
        "available_hatches": [
            "\\",
            "xx",
            "/",
            "\\",
            "|",
            "-",
            ".",
        ],
        "ymin": metric_specs["metric_range"][0] - 0.1,
        "ymax": metric_specs["metric_range"][1],
        "plot_legend": True,
        "axis": None,
        "numeric_value_settings": {"fontsize": 12, "rotation": -45},
        "ensemble_numeric_value_space": 0.09,
    }

    # plot
    fig = plt.figure(figsize=(22, 10))
    make_summary_plot(
        summary_evaluation_df,
        plot_settings,
        df_ensemble=None,
    )

    if SAVE_FIGURE:
        file_name = f"Summary_classification_performance_{metric.title()}_{TIME_STAMP}"
        fig.savefig(
            os.path.join(SAVE_PATH, file_name + ".pdf"), bbox_inches="tight", dpi=100
        )
        fig.savefig(
            os.path.join(SAVE_PATH, file_name + ".png"), bbox_inches="tight", dpi=100
        )
        plt.close(fig)
    else:
        plt.show()
# %% MAKE SUMMARY TEXT 

# define the order in the text
classification_level_order = ["category", "family", "type"]
nbr_classes = [pd.unique(summary_evaluation_df.loc[summary_evaluation_df.classification_level==c].nbr_classes)[0] for c in classification_level_order]
model_order = ["clam"]
feature_extractor_order = ['resnet50', 'vit_hipt', 'vit_uni']
metrics = ['mcc', 'balanced_accuracy', 'auc']

def aggregate_evaluation_for_mectric(x, pm_symbol = f" \u00B1 " , format='0.2'):
    mean = np.mean(x)
    std = np.std(x)
    min = np.min(x)
    max = np.max(x)
    min_05_q = np.quantile(x, 0.05)
    max_95_q = np.quantile(x, 0.95)

    # string for printing
    return f"{mean:{format}f}{pm_symbol}{std:{format}f}\nrange [{min:{format}f}, {max:{format}f}]\nquantile [{min_05_q:{format}f}, {max_95_q:{format}f}]"

# define aggregation dictionary to pass to the .agg groupby
aggregation_dict = dict.fromkeys(metrics)
for m in aggregation_dict.keys():
    aggregation_dict[m] = lambda x : aggregate_evaluation_for_mectric(x)


# compress summary_evaluation_df to be able to plot it using to_markdown
compressed_summary = summary_evaluation_df.groupby(['classification_level', 'model', 'features']).agg(aggregation_dict)
print(compressed_summary.to_markdown(tablefmt="pipe", stralign='center'))




