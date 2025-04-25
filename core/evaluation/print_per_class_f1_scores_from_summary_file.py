# %% IMPORTS
"""
Utility that given the .csv summary file, prints a table with the per-class metrics (f1-score, precision, recall).
"""

import os
import pandas as pd
import pathlib
import numpy as np
import scipy.stats as st

# local imports
from utils import aggregate_evaluation_for_metric


# %% PATHS

AGGREGATED_CSV_FILE = "/local/d2/iulta54/Research/P7_BTB_DEEP_LEARNING/outputs/classification_results_post_review_balanced_classes/aggregated_evaluation_20250409.csv"
TIME_STAMP = pathlib.Path(AGGREGATED_CSV_FILE).parts[-1].split(".")[0].split("_")[-1]
SAVE_PATH = pathlib.Path(
    os.path.join(
        os.path.dirname(AGGREGATED_CSV_FILE), f"per_class_metrics_{TIME_STAMP}"
    )
)
SAVE_PATH.mkdir(parents=True, exist_ok=True)

# load aggregated file
summary_evaluation_df = pd.read_csv(AGGREGATED_CSV_FILE)

# check if there is any column with the name "Unnamed" and drop it
if "Unnamed: 0" in summary_evaluation_df.columns:
    summary_evaluation_df = summary_evaluation_df.drop(
        summary_evaluation_df.columns[
            summary_evaluation_df.columns.str.contains("Unnamed: 0")
        ],
        axis=1,
    )

# %% FOR EACH OF THE METRICS and FOR EACH OF THE CLASSIFICATION TASKS, FEATURE EXTRACTOR AND AGGREGATION
# groupby the dataset and apply average over the different columns

aggregated_information = (
    summary_evaluation_df.drop(["classes", "class_fractions", "model", "set"], axis=1)
    .groupby(["classification_level", "features", "aggregation"])
    .agg(lambda x: aggregate_evaluation_for_metric(x))
)

# remove columns that are not of interest
aggregated_information = aggregated_information.drop(
    [
        "nbr_classes",
        "repetition",
        "fold",
        "mcc",
        "balanced_accuracy",
        "accuracy",
        "auc",
        "f1-score",
    ],
    axis=1,
)

print(aggregated_information.to_markdown(tablefmt="pipe", stralign="center"))

# %% SAVE TO FILE, ONE FOR EACH CLASSIFICATION TASK


def convert_string_to_list(input_string):
    # Remove the leading and trailing square brackets and split the string by "', '"
    list_of_strings = input_string.strip("[]").split("', '")

    # Remove any leading or trailing single quotes from each element
    list_of_strings = [s.strip("'") for s in list_of_strings]

    return list_of_strings


# Example input
input_string = "['Choroid plexus tumors', 'Embryonal tumors', 'Germ cell tumors', 'Gliomas, glioneuronal tumors, and neuronal tumors', 'Meningiomas', 'Mesenchymal, non-meningothelial tumors', 'Tumors of the sellar region']"

# get unique classification levels
classification_levels = summary_evaluation_df["classification_level"].unique()
# get a list af strings for the classes in each level
classification_levels_strings = [
    convert_string_to_list(
        summary_evaluation_df[
            summary_evaluation_df["classification_level"] == classification_level
        ]["classes"].unique()[0]
    )
    for classification_level in classification_levels
]

# create a dictionary with the classification levels and the classes
classification_levels_dict = {
    classification_level: classification_levels_strings[i]
    for i, classification_level in enumerate(classification_levels)
}

# loop through the classification levels and save the aggregated information for each level changing the column names using the classes
for classification_level, classes in classification_levels_dict.items():
    # get the aggregated information for the classification level
    temp = aggregated_information.loc[
        classification_level,
        aggregated_information.columns.str.contains("f1|precision|recall"),
    ]

    # rename the columns using the classes
    temp.columns = [
        f"{metric}_{class_name}"
        for metric in ["f1", "precision", "recall"]
        for class_name in classes
    ]
    # save to file
    # with open(
    #     os.path.join(SAVE_PATH, f"table_summary_per_class_evaluation_{classification_level}.md"),
    #     "w",
    # ) as f:
    print(temp.to_markdown(tablefmt="pipe", stralign="center"), file=f)

# %%
with open(os.path.join(SAVE_PATH, "table_summary_per_class_evaluation.md"), "w") as f:
    print(
        aggregated_information.to_markdown(tablefmt="pipe", stralign="center"), file=f
    )


# %% SAVE PER METRIC

metrics = ["f1", "precision", "recall"]

for metric in metrics:
    # drop all the columns that do not have the metric in the text
    temp = aggregated_information.loc[
        :, aggregated_information.columns.str.contains(metric)
    ]
    # save
    with open(
        os.path.join(SAVE_PATH, f"table_summary_per_class_evaluation_{metric}.md"), "w"
    ) as f:
        print(temp.to_markdown(tablefmt="pipe", stralign="center"), file=f)
