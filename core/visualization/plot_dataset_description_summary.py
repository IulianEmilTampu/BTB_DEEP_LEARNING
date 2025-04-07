# %%
"""
Plots the class distribution given the dataset_description.csv file
"""

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# %% PATHS

DATASET_DESCRIPTION_CSV = "/flush/iulta54/Research/P11-BTB_DEEP_LEARNING/outputs/splits_tumor_type_cv/dataset_descriptor.csv"
SAVE_PATH = os.path.dirname(DATASET_DESCRIPTION_CSV)

# load csv
dataset_description = pd.read_csv(DATASET_DESCRIPTION_CSV)

# %% PLOT CLASS DISTRIBUTION BASED ON CASE_ID (SUBJECT)

PLOT_NAME = "Tumor_type_subject_and_WSIs_count"
SAVE_PLOT = False

# re-arrange dataset_summary to contain total counts
dataset_summary_for_plot = (
    dataset_description.groupby("label")
    .agg({"case_id": lambda x: len(pd.unique(x)), "slide_id": "count"})
    .reset_index()
    .melt(["label"], var_name="COUNT_OVER", value_name="COUNTS")
)


# define orders
hue_var = "COUNT_OVER"
hue_order = ["case_id", "slide_id"]
col_order = list(pd.unique(dataset_summary_for_plot.label))
y_var = "COUNTS"
col_var = "label"

# build figure
plt.figure(figsize=(80, 10))
g = sns.catplot(
    dataset_summary_for_plot,
    kind="bar",
    x="COUNT_OVER",
    y="COUNTS",
    col="label",
    height=4,
    aspect=0.5,
    hue_order=hue_order,
    col_order=col_order,
)
# fix axis limits
g.set(
    ylim=(0, dataset_summary_for_plot.COUNTS.max() + 50),
    ylabel="Count",
    xlabel="Count over",
)

# add counts and fraction over the entire dataset
# with g.axes[0, n] we loop though the different SITES
width_bar_plot = 0.8
text_versical_offset_count = 4 * len(col_order)
text_vetical_offset_fraction = 2

for col_indx, col in enumerate(col_order):
    ax = g.axes[0, col_indx]  # this looks at the bar plots of a site
    # add values for the different hue values (case_id and slide_id)
    for hue_indx, hue in enumerate(hue_order):
        # get value for this bar
        count = dataset_summary_for_plot.loc[
            (dataset_summary_for_plot[col_var] == col)
            & (dataset_summary_for_plot[hue_var] == hue)
        ].COUNTS.values[0]
        fraction_over_all_sites = (
            count
            / dataset_summary_for_plot.loc[
                dataset_summary_for_plot[hue_var] == hue
            ].COUNTS.sum()
            * 100
        )
        # the x position in the axix is just the index of the hue
        ax.text(
            hue_indx,
            count + text_versical_offset_count,
            f"{count}",
            ha="center",
            weight="bold",
        )
        ax.text(
            hue_indx,
            count + text_vetical_offset_fraction,
            f"({fraction_over_all_sites:0.1f}%)",
            ha="center",
        )
        #
        ax.set_title(col, fontsize=5)
# add legend with the information about the total number of subjects and
ax = g.axes[0, -1]
textstr = "\n".join(
    [
        f"Counts over all labels",
        f"{'Nbr cases*':12s}: {dataset_summary_for_plot.loc[dataset_summary_for_plot.COUNT_OVER=='case_id'].COUNTS.sum()}",
        f"{'Nbr WSIs':12s}: {dataset_summary_for_plot.loc[dataset_summary_for_plot.COUNT_OVER=='slide_id'].COUNTS.sum()}",
        f"* cases are counted as unique\nsubject-diagnosis pairs.",
    ]
)

text_box_props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
x_pos = 2
y_pos = (
    dataset_summary_for_plot.loc[
        (dataset_summary_for_plot[col_var] == col_order[-1])
        & (dataset_summary_for_plot[hue_var] == hue_order[-1])
    ].COUNTS.values[0]
    + 100
)
ax.text(
    x_pos, y_pos, textstr, fontsize=15, bbox=text_box_props, verticalalignment="top"
)

# save

if SAVE_PLOT:
    plt.savefig(
        fname=os.path.join(SAVE_PATH, f"{PLOT_NAME}.pdf"),
        dpi=200,
        format="pdf",
        bbox_inches="tight",
    )

    plt.savefig(
        fname=os.path.join(SAVE_PATH, f"{PLOT_NAME}.png"),
        dpi=200,
        format="png",
        bbox_inches="tight",
    )

# %%
