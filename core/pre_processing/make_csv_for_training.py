# %%
"""
Utility that given the summary file (overall_summary_data_with_wsi_paths_date.csv),
creates the .csv files needed to run HIPT or CLAM classification training.
"""

import os
import sys
import warnings
import hydra
import datetime
import pandas as pd
from tqdm import tqdm

from pathlib import Path
import yaml
import omegaconf
from omegaconf import DictConfig, OmegaConf

from sklearn.model_selection import (
    KFold,
    train_test_split,
    StratifiedKFold,
    StratifiedGroupKFold,
    GroupShuffleSplit,
    StratifiedShuffleSplit,
)

# %% UTILITIES

def case_id_from_anonymized_code(x: str):
    """
    The BTB anonymized codes are in the following format.
    BTB2024_site_case_diagnosis_pad_glass-id
    """
    # return x.split("_")[2]
    return "_".join(x.split("_")[2:4])


def site_id_from_anonymized_code(x: str):
    """
    The BTB anonymized codes are in the following format.
    BTB2024_site_case_pad_glass-id
    """
    try:
        return x.split("_")[1]
    except:
        print(x.split("_"))


def get_repetition_split(
    cfg: DictConfig, df, random_seed: int = 29122009, print_summary: bool = False
):
    """
    Utility that splits the slide_ids in the df using a per case_id split (subject wise-splitting).
    It applies label and/or site stratification is requested.

    INPUT
        cfg : DictConfig
            Configuration dictionary
        df : pandas Dataframe.
            Dataframe with the case_id, slide_id, label and site (if requested) information.
        random_seed : int
            Seeds the random split

    OUTPUT
        df : pandas Dataframe
            Dataframe with each of the slide_id as training, val or test for each of the specified folds.
    """

    if print_summary:
        # print summary before start splitting
        print_df_summary(df)

    # ################## work on splitting
    if cfg.class_stratification:
        # ################## TEST SET
        # perform stratified split
        sgkf = StratifiedGroupKFold(
            n_splits=int(1 / cfg.test_fraction),
            shuffle=True,
            random_state=random_seed,
        )

        train_val_ix, test_ix = next(sgkf.split(df, y=df.label, groups=df.case_id))

        # get testing set
        df_test_split = df.loc[test_ix].reset_index()
        if print_summary:
            print(
                f'{"Test set":9s}: {len(test_ix):5d} {"test":10} files ({len(pd.unique(df_test_split.case_id)):4d} unique subjects ({pd.unique(df_test_split.label)} {[len(pd.unique(df_test_split.loc[df_test_split.label == c].case_id)) for c in list(pd.unique(df_test_split.label))]}))'
            )

        # get train_val set
        df_train_val_split = df.loc[train_val_ix].reset_index()
        # make a copy of the df_train_val_split to use as back bone for the dataframe to be returned (add the test at the end)
        dataset_split_df = df_train_val_split.copy()

        # ################# TRAINING and VALIDATION SETs
        # build nbr_splits. This is needed in case the cfg.number_of_folds and cfg.validation_fraction is provided
        if cfg.number_of_folds == 1:
            if cfg.validation_fraction is not None:
                n_splits = int(1 / cfg.validation_fraction)
            else:
                n_splits = 2
        else:
            n_splits = cfg.number_of_folds

        sgkf = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_seed,
        )

        # if only one internal fold is requested, do as in the testing. Else,
        # get all the folds (just use next as many times as the one requested by nbr of folds)
        for cv_f, (train_ix, val_ix) in enumerate(
            sgkf.split(
                df_train_val_split,
                groups=df_train_val_split.case_id,
                y=df_train_val_split.label,
            )
        ):
            # add a column in the dataset_split_df and flag all the files based on the split
            dataset_split_df[f"fold_{cv_f+1}"] = "train"
            # flag the training files
            dataset_split_df.loc[val_ix, f"fold_{cv_f+1}"] = "validation"

            # add to the df_test_split the flag for this fold
            df_test_split[f"fold_{cv_f+1}"] = "test"

            # check that there are elements in the training, val and test for each of the classes
            for s_ix, s in zip((train_ix, val_ix), ("train", "validation")):
                aus_df = df_train_val_split.loc[s_ix]
                # get nbr. unique subjects per class
                classes = list(pd.unique(df.label))
                classes.sort()
                per_class_nbr_subjs = [
                    len(pd.unique(aus_df.loc[aus_df.label == c].case_id))
                    for c in list(pd.unique(df.label))
                ]
                # if any of the classes has nbr_subjs == 0, raise warning
                if any([i == 0 for i in per_class_nbr_subjs]):
                    warnings.warn(
                        f"Some of the classes in {s} set have nbr_subjs == 0 (fold=={cv_f}).\n   Unique classes: {list(pd.unique(df.label))}\n   Unique subjects: {per_class_nbr_subjs}"
                    )

            # print summary
            if print_summary:
                aus_df = df_train_val_split.loc[train_ix]
                print(
                    f'Fold {cv_f+1:4d}: {len(train_ix):5d} {"training":10} files ({len(pd.unique(df_train_val_split.loc[train_ix].case_id)):4d} unique subjects ({list(pd.unique(aus_df.label))} {[len(pd.unique(aus_df.loc[aus_df.label == c].case_id)) for c in list(pd.unique(aus_df.label))]}))'
                )
                aus_df = df_train_val_split.loc[val_ix]
                print(
                    f'Fold {cv_f+1:4d}: {len(val_ix):5d} {"validation":10} files ({len(pd.unique(df_train_val_split.loc[val_ix].case_id)):4d} unique subjects ({list(pd.unique(aus_df.label))} {[len(pd.unique(aus_df.loc[aus_df.label == c].case_id)) for c in list(pd.unique(aus_df.label))]}))'
                )

            if cv_f + 1 == cfg.number_of_folds:
                break

    else:
        # create split without stratification
        # ################## TEST SET
        gs = GroupShuffleSplit(
            n_splits=1,
            train_size=(1 - cfg.test_fraction),
            random_state=random_seed,
        )
        train_val_ix, test_ix = next(gs.split(df, groups=df.case_id))

        df_test_split = df.loc[test_ix].reset_index()

        if print_summary:
            print(
                f'{"Test set":9s}: {len(test_ix):5d} {"test":10} files ({len(pd.unique(df_test_split.case_id)):4d} unique subjects ({pd.unique(df_test_split.label)} {[len(pd.unique(df_test_split.loc[df_test_split.label == c].case_id)) for c in list(pd.unique(df_test_split.label))]}))'
            )

        # ################## TRAINING and VALIDATION SETs
        df_train_val_split = df.loc[train_val_ix].reset_index()
        # make a copy of the df_train_val_split to use as back bone for the dataframe to be returned (add the test at the end)
        dataset_split_df = df_train_val_split.copy()

        gs = GroupShuffleSplit(
            n_splits=cfg.number_of_folds,
            train_size=(1 - cfg.validation_fraction)
            if cfg.number_of_folds == 1
            else (1 - 1 / cfg.number_of_folds),
            random_state=random_seed,
        )

        for cv_f, (train_ix, val_ix) in enumerate(
            gs.split(df_train_val_split, groups=df_train_val_split.case_id)
        ):
            # add a column in the dataset_split_df and flag all the files based on the split
            dataset_split_df[f"fold_{cv_f+1}"] = "validation"
            # flag the training files
            dataset_split_df.loc[train_ix, f"fold_{cv_f+1}"] = "train"

            # add to the df_test_split the flag for this fold
            df_test_split[f"fold_{cv_f+1}"] = "test"

            # print summary
            if print_summary:
                aus_df = df_train_val_split.loc[train_ix]
                print(
                    f'Fold {cv_f+1:4d}: {len(train_ix):5d} {"training":10} files ({len(pd.unique(df_train_val_split.loc[train_ix].case_id)):4d} unique subjects ({list(pd.unique(aus_df.label))} {[len(pd.unique(aus_df.loc[aus_df.label == c].case_id)) for c in list(pd.unique(aus_df.label))]}))'
                )
                aus_df = df_train_val_split.loc[val_ix]
                print(
                    f'Fold {cv_f+1:4d}: {len(val_ix):5d} {"validation":10} files ({len(pd.unique(df_train_val_split.loc[val_ix].case_id)):4d} unique subjects ({list(pd.unique(aus_df.label))} {[len(pd.unique(aus_df.loc[aus_df.label == c].case_id)) for c in list(pd.unique(aus_df.label))]}))'
                )

    # finish up the dataset_split_df by merging the df_test_split
    dataset_split_df = pd.concat(
        [dataset_split_df, df_test_split], ignore_index=True
    ).reset_index(drop=True)

    # remove level_0
    dataset_split_df = dataset_split_df.drop(columns=["level_0", "index"])

    return dataset_split_df


def get_repetition_split_v2(
    cfg: DictConfig, df, random_seed: int = 29122009, print_summary: bool = False
):
    """
    Utility that splits the slide_ids in the df using a per case_id split (subject wise-splitting).
    It applies label stratification is requested. TODO site stratification

    INPUT
        cfg : DictConfig
            Configuration dictionary
        df : pandas Dataframe.
            Dataframe with the case_id, slide_id, label and site (if requested) information.
        random_seed : int
            Seeds the random split

    OUTPUT
        df : pandas Dataframe
            Dataframe with each of the slide_id as training, val or test for each of the specified folds.
    """

    if print_summary:
        # print summary before start splitting
        print_df_summary(df)

    # get indexes in the dataset for each of the subjects
    unique_case_ids = list(pd.unique(df.case_id))
    case_id_to_index_map = {}  # maps where each slide for each case id are.
    case_id_to_label = {}
    for c_id in unique_case_ids:
        case_id_to_index_map[c_id] = df.index[df.case_id == c_id].tolist()
        case_id_to_label[c_id] = pd.unique(df.loc[df.case_id == c_id].label).tolist()[0]

    # get a df which has two columns: case_id and label
    df_for_split = pd.DataFrame(case_id_to_label.items(), columns=["case_id", "label"])

    # ################## work on splitting
    if cfg.class_stratification:
        # get test set case_id indexes and then the train and validation case_id indexes.
        # Using these, create a column for each fold and flag the slide_id as test, train aor validation.

        split_indexes = []  # saves the split indexes for each of the folds.

        # ################## TEST SET
        # The number of splits for the first split (test and train_val) is computed based on the fraction
        # of the test set
        if cfg.test_fraction != 0:
            n_splits = int(1 / cfg.test_fraction)
            skf = StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=random_seed,
            )

            train_val_ix, test_ix = next(
                skf.split(X=df_for_split.case_id, y=df_for_split.label)
            )
        else:
            # test fraction set to 0. All the samples used for training and validation
            test_ix = []
            train_val_ix = list(range(len(df_for_split)))

        # ################## TRAIN and VALIDATION
        df_train_val_for_split = df_for_split.loc[train_val_ix].reset_index()

        # Build nbr_splits considering that the fraction cfg.validation_fraction is wrt to the entire dataset.
        if cfg.number_of_folds == 1:
            if cfg.validation_fraction is not None:
                n_splits = int(1 / (cfg.validation_fraction / (1 - cfg.test_fraction)))
            else:
                n_splits = 2
        else:
            n_splits = cfg.number_of_folds

        skf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_seed,
        )

        # get splits for all the folds
        for cv_f, (train_ix, val_ix) in enumerate(
            skf.split(X=df_train_val_for_split.case_id, y=df_train_val_for_split.label)
        ):
            # save indexes
            split_indexes.append(
                {
                    "test": list(df_for_split.loc[test_ix, "case_id"]),
                    "train": list(df_train_val_for_split.loc[train_ix, "case_id"]),
                    "validation": list(df_train_val_for_split.loc[val_ix, "case_id"]),
                }
            )

            if cv_f == cfg.number_of_folds - 1:
                break
    else:
        raise ValueError("Not label-stratified split is not implemented.")

    # add folds columns to the dataframe and return
    splitted_df = df.copy()
    for cv_f, s in enumerate(split_indexes):
        # create column for this fold
        splitted_df[f"fold_{cv_f+1}"] = "NA"
        for split_name, case_ids in s.items():
            # check that there are elements for each of the classes
            classes = list(pd.unique(splitted_df.label))
            classes.sort()
            per_class_nbr_subjs = df_for_split.loc[df_for_split.case_id.isin(case_ids)]
            per_class_nbr_subjs = [
                len(
                    pd.unique(
                        per_class_nbr_subjs.loc[per_class_nbr_subjs.label == c].case_id
                    )
                )
                for c in classes
            ]

            # if any of the classes has nbr_subjs == 0, raise warning
            if any([i == 0 for i in per_class_nbr_subjs]):
                warnings.warn(
                    f"Some of the classes in {split_name} set have nbr_subjs == 0 (fold=={cv_f})."
                )
                print(
                    f"{[print(f'{classes[i]:42s}: {per_class_nbr_subjs[i]}') for i in range(len(classes))]}"
                )

            # get the indexes of the slide ids for these case_id
            slide_indexes = []
            [
                slide_indexes.extend(case_id_to_index_map[case_id])
                for case_id in case_ids
            ]

            # set flag the slide ids
            splitted_df.loc[slide_indexes, f"fold_{cv_f+1}"] = split_name

        #     print(f'{cv_f}: {split_name} -> {len(case_ids) / len(df_for_split) * 100:0.2f}% (subj: {len(case_ids)}, slides: {len(slide_indexes)})')
        # print('\n')
    return splitted_df


def print_df_summary(df):
    # print totals first
    print(f"Number of slides: {len(df)}")
    print(f"Number of unique case_ids (subjects): {len(pd.unique(df.case_id))}")
    if "site_id" in df.columns:
        print(f"Number of sites: {len(pd.unique(df.site_id))}")
    print(f"Number of unique classes/labels: {len(pd.unique(df.label))}")

    # break down on a class level
    if "site_id" in df.columns:
        aus = df.groupby(["label"]).agg(
            {
                "case_id": lambda x: len(pd.unique(x)),
                "slide_id": lambda x: len(x),
                "site_id": lambda x: len(pd.unique(x)),
            }
        )
    else:
        aus = df.groupby(["label"]).agg(
            {"case_id": lambda x: len(pd.unique(x)), "slide_id": lambda x: len(x)}
        )
    print(aus)


def save_for_hipt(cfg, df, repetition_number):
    """
    See README.md file for a detailed description of how the HIPT framework needs the .csv files saved for training and evaluation (tune).
    """

    for fold in range(cfg.number_of_folds):
        for split, file_name in zip(
            ["train", "validation", "test"], ["train.csv", "tune.csv", "test.csv"]
        ):
            # filter dataframe
            df_for_save = df.loc[df[f"fold_{fold+1}"] == split]
            df_for_save = df_for_save[["slide_id", "label_integer"]]
            df_for_save = df_for_save.rename(columns={"label_integer": "label"})

            # save
            save_path = Path(
                cfg.output_dir,
                cfg.experiment_name,
                "hipt",
                f"{cfg.classification_level}_reps_{cfg.number_of_repetitions}_folds_{cfg.number_of_folds}",
                f"repetition_{repetition_number}",
                f"fold_{fold+1}",
            )
            save_path.mkdir(parents=True, exist_ok=True)
            df_for_save.to_csv(
                Path(save_path, file_name), index_label=False, index=False
            )

    # save cfg to file so that it can be reproduced
    OmegaConf.save(
        cfg,
        Path(
            cfg.output_dir,
            cfg.experiment_name,
            "hipt",
            f"{cfg.classification_level}_reps_{cfg.number_of_repetitions}_folds_{cfg.number_of_folds}",
            "config.yaml",
        ),
    )


def save_for_clam(cfg, df, repetition_number):
    """
    See README.md file for a detailed description of how the CLAM framework needs the .csv files saved for training and evaluation (tune).
    """

    # make save_path
    if cfg.split_strategy == "cv":
        save_path = Path(
            cfg.output_dir,
            cfg.experiment_name,
            "clam",
            f"{cfg.classification_level}_reps_{cfg.number_of_repetitions}_folds_{cfg.number_of_folds}",
            f"repetition_{repetition_number}",
        )
    elif cfg.split_strategy == "npb":
        save_path = Path(
            cfg.output_dir,
            cfg.experiment_name,
            "clam",
            f"{cfg.classification_level}_reps_{cfg.npb_replicates}",
            f"repetition_{0}",
        )
    else:
        raise NotImplemented

    save_path.mkdir(parents=True, exist_ok=True)

    for fold in range(cfg.number_of_folds):
        # # make and save split_nbr.csv file
        df_for_save = df[[f"fold_{fold+1}", "slide_id"]]
        df_for_save = df_for_save.rename(columns={f"fold_{fold+1}": "set"})

        # combine and take out each set
        gb = df_for_save.groupby(["set"])
        train = (
            gb.get_group("train")
            .drop(columns=["set"])
            .rename(columns={"slide_id": "train"})
            .reset_index()
        )
        validation = (
            gb.get_group("validation")
            .drop(columns=["set"])
            .rename(columns={"slide_id": "val"})
            .reset_index()
        )
        if cfg.test_fraction != 0:
            test = (
                gb.get_group("test")
                .drop(columns=["set"])
                .rename(columns={"slide_id": "test"})
                .reset_index()
            )
        else:
            test = pd.DataFrame(columns=['test'])

        # concatenate and save
        split_to_save = pd.concat([train, validation, test], axis=1).drop(
            columns=["index"]
        )
        if cfg.split_strategy == "npb":
            split_to_save.to_csv(
                Path(save_path, f"split_{fold+repetition_number}.csv"),
                index_label=False,
                index=False,
            )
        else:
            split_to_save.to_csv(
                Path(save_path, f"split_{fold}.csv"), index_label=False, index=False
            )

        # # make and save split_nbr_bool.csv file
        df_for_save["train"] = df_for_save.apply(lambda x: x.set == "train", axis=1)
        df_for_save["val"] = df_for_save.apply(lambda x: x.set == "validation", axis=1)
        df_for_save["test"] = df_for_save.apply(lambda x: x.set == "test", axis=1)

        # refine and save
        split_bool_to_save = df_for_save.drop(columns=["set"])
        if cfg.split_strategy == "npb":
            split_bool_to_save.to_csv(
                Path(save_path, f"split_{fold+repetition_number}_bool.csv"),
                index_label=False,
                index=False,
            )
        else:
            split_bool_to_save.to_csv(
                Path(save_path, f"split_{fold}_bool.csv"),
                index_label=False,
                index=False,
            )

        # # make and save split_nbr_descriptor.csv file
        df_for_save = df[[f"fold_{fold+1}", "slide_id", "label"]]
        df_for_save = df_for_save.rename(columns={f"fold_{fold+1}": "set"})

        # create dummy columns needed for later
        df_for_save["train"] = df_for_save.apply(lambda x: x.set == "train", axis=1)
        df_for_save["val"] = df_for_save.apply(lambda x: x.set == "validation", axis=1)
        df_for_save["test"] = df_for_save.apply(lambda x: x.set == "test", axis=1)

        gb = df_for_save.groupby(["label"]).agg(
            {"train": "sum", "val": "sum", "test": "sum"}
        )
        if cfg.split_strategy == "npb":
            gb.to_csv(
                Path(save_path, f"split_{fold+repetition_number}_descriptor.csv"),
                index_label="class",
                index=True,
            )
        else:
            gb.to_csv(
                Path(save_path, f"split_{fold}_descriptor.csv"),
                index_label="class",
                index=True,
            )

    # save cfg to file so that it can be reproduced
    if cfg.split_strategy == "npb" and repetition_number == 0:
        OmegaConf.save(
            cfg,
            Path(
                cfg.output_dir,
                cfg.experiment_name,
                "clam",
                f"{cfg.classification_level}_reps_{cfg.npb_replicates}",
                "config.yaml",
            ),
        )
    elif cfg.split_strategy == "cv":
        OmegaConf.save(
            cfg,
            Path(
                cfg.output_dir,
                cfg.experiment_name,
                "clam",
                f"{cfg.classification_level}_reps_{cfg.number_of_repetitions}_folds_{cfg.number_of_folds}",
                "config.yaml",
            ),
        )


def get_label_to_integer_map(unique_labels: list):
    """
    Returns a dictionary with keys the string for a label and as value an integer.
    Here using alphabetical order and integers form 0 to len(unique_labels)
    """
    unique_labels.sort()
    label_to_integer_map = dict.fromkeys(unique_labels)
    for k_index, k in enumerate(label_to_integer_map.keys()):
        label_to_integer_map[k] = k_index

    return label_to_integer_map


# %% MAIN


@hydra.main(
    version_base="1.2.0",
    config_path="../../configs/pre_processing",
    config_name="make_csv_for_training",
)
def main(cfg: DictConfig):
    # load BT_csv file
    btb_csv = pd.read_csv(cfg.btb_csv_path, encoding="ISO-8859-1")
    # make sure we have bools in USE_DURING_ANALYSIS and ACCEPTABLE_IMAGE_QUALITY columns
    d = {"True": True, "False": False, "UNMATCHED_WSI": "UNMATCHED_WSI",'TRUE':True, 'FALSE': False}
    btb_csv["USE_DURING_ANALYSIS"] = btb_csv["USE_DURING_ANALYSIS"].map(d)
    d = {
        "TRUE": True,
        "FALSE": False,
        "UNMATCHED_WSI": "UNMATCHED_WSI",
        "UNMATCHED": "UNMATCHED",
    }
    btb_csv["ACCEPTABLE_IMAGE_QUALITY"] = btb_csv["ACCEPTABLE_IMAGE_QUALITY"].map(d)

    # include only those that are acceptable for the analysis (USE_DURING_ANALYSIS==True & ACCEPTABLE_IMAGE_QUALITY==True)
    btb_csv = btb_csv.loc[
        (btb_csv.USE_DURING_ANALYSIS == True)
        & (btb_csv.ACCEPTABLE_IMAGE_QUALITY == True)
    ]

    # remove/include class_labels or sites from the dataset if requested
    if cfg.classes_to_include:
        btb_csv = btb_csv.loc[
            btb_csv[cfg.class_column_name].isin(cfg.classes_to_include)
        ]

    elif cfg.classes_to_exclude:
        btb_csv = btb_csv.loc[
            ~btb_csv[cfg.class_column_name].isin(cfg.classes_to_exclude)
        ]

    if any([cfg.site_to_exclude, cfg.site_to_include]):
        # check if the columns refering to the site is available. If not, print warning and infere.
        if not cfg.site_column_name in btb_csv.columns:
            warnings.warn(f'The given site name does not exist. Attempting to infere site using a site to anonym key mapping.')
            # define site to anonymized code mapping
            code_to_site = {
            '5e4761c2' : 'LUND', 
            'fc173989' : 'KS',
            '103f236b' : 'GOT',
            '6c730372' : 'LK',
            '9a2a64c4' : 'UMEA',
            '9fb809d6' : 'UPPSALA',
            }
            btb_csv["SITE"] = btb_csv.apply(
            lambda x: code_to_site[site_id_from_anonymized_code(x.ANONYMIZED_CODE)], axis=1
            )

            # re-initialize the site column name 
            cfg.site_column_name = 'SITE'

        # remove/include specified sites
        if cfg.site_to_include:
            print(f'Including slides belonging to sites: {cfg.site_to_include}')
            btb_csv = btb_csv.loc[btb_csv[cfg.site_column_name].isin(cfg.site_to_include)]
        if cfg.site_to_exclude:
            print(f'Removing slides belonging to sites: {cfg.site_to_exclude}')
            btb_csv = btb_csv.loc[~btb_csv[cfg.site_column_name].isin(cfg.site_to_exclude)]

    # create a new Dataframe with only the ANONYMIZED_CODE, CLASS_LABEL and SITE (if needed).
    # Use the class_column to get the class_label at the right classification level.
    df_for_split = btb_csv[["ANONYMIZED_CODE", cfg.class_column_name]]
    df_for_split = df_for_split.rename(
        columns={"ANONYMIZED_CODE": "slide_id", cfg.class_column_name: "label"}
    )
    df_for_split = df_for_split.dropna(subset=["label"])

    # get case_id (subject id) from the anonymized codes (slide_id). This is needed to perform a per-case/subject split
    df_for_split["case_id"] = df_for_split.apply(
        lambda x: case_id_from_anonymized_code(x.slide_id), axis=1
    )
    # get site_id if site stratification
    if cfg.site_stratification:
        df_for_split["site_id"] = df_for_split.apply(
            lambda x: site_id_from_anonymized_code(x.slide_id), axis=1
        )

    # # if creating splits for patient level features, compress the dataframe (one row per case_id with slide id == case id. The Anonymized code is also trimmed to only have the site and subject codes)
    if cfg.feature_level == "patient":
        print('Performing split on patient level features.')
        # compact information
        df_for_split["slide_id"] = df_for_split.apply(
            lambda x: "_".join(x.slide_id.split("_")[0:3]), axis=1
        )

        if cfg.site_stratification:
            df_for_split = (
                df_for_split.groupby(["case_id"])
                .agg(
                    {
                        "slide_id": lambda x: pd.unique(x)[0],
                        "label": lambda x: pd.unique(x)[0],
                        "site_id": lambda x: pd.unique(x)[0],
                    }
                )
                .reset_index()
            )
        else:
            df_for_split = (
                df_for_split.groupby(["case_id"])
                .agg(
                    {
                        "slide_id": lambda x: pd.unique(x)[0],
                        "label": lambda x: pd.unique(x)[0],
                    }
                )
                .reset_index()
            )
    print("\n\n")
    print_df_summary(df_for_split)
    print(df_for_split)

    # check if the slide_ids are available as extracted features
    if cfg.check_available_features:
        slide_ids = list(df_for_split.slide_id.values)
        feature_check = [
            os.path.isfile(os.path.join(cfg.feature_dir, sid + ".pt"))
            for sid in slide_ids
        ]
        if not all(feature_check):
            print(
                f"Not all slide_ids have available feature files.\nFound {feature_check.count(True)}.\nMissing {feature_check.count(False)} out of {len(slide_ids)} ({feature_check.count(False) / len(slide_ids) * 100:0.2f}%)."
            )

        if cfg.skip_missing_features:
            print(
                f"ATTENTION! Removing the missing slide_ids (skip_missing_features == {cfg.skip_missing_features})"
            )
            slide_id_with_features = [
                slide_ids[i] for i, c in enumerate(feature_check) if c
            ]
            df_for_split = df_for_split.loc[
                df_for_split.slide_id.isin(slide_id_with_features)
            ]

    # remove (if requested) labels with too few subjects
    if cfg.min_nbr_subjects_per_class != -1:
        min_nbr_subjects_per_label = int(cfg.min_nbr_subjects_per_class)
        labels_to_keep = [
            l
            for l in list(pd.unique(df_for_split.label))
            if len(pd.unique(df_for_split.loc[df_for_split.label == l].case_id))
            >= min_nbr_subjects_per_label
        ]
        labels_to_remove = [
            l for l in list(pd.unique(df_for_split.label)) if l not in labels_to_keep
        ]
        percentage_slides_to_remove = (
            len(df_for_split.loc[df_for_split.label.isin(labels_to_remove)])
            / len(df_for_split)
        ) * 100
        print(
            f"Removed {len(labels_to_remove)} labels based on the min nbr. of subject filter ( >= {min_nbr_subjects_per_label}) ({percentage_slides_to_remove:0.2f}% of the slides)."
        )
        print(f"Using {len(labels_to_keep)} labels.")
        df_for_split = df_for_split.loc[df_for_split.label.isin(labels_to_keep)]

    # map the class string to an integer (starts from 0)
    label_to_integer_map = get_label_to_integer_map(list(pd.unique(df_for_split.label)))
    df_for_split["label_integer"] = df_for_split.apply(
        lambda x: label_to_integer_map[x.label], axis=1
    )
    print(label_to_integer_map)

    # reset index prior splitting
    df_for_split = df_for_split.reset_index()

    if cfg.split_strategy == "cv":
        # start splitting, one split per repetition
        with tqdm(total=cfg.number_of_repetitions, unit="rep") as rep_tqdm:
            for r in range(cfg.number_of_repetitions):
                # create split for this repetition
                df_split = get_repetition_split_v2(
                    cfg,
                    df_for_split,
                    random_seed=cfg.random_seed + r,
                    print_summary=False,
                )

                # save in the format of the specified classification framework
                for f in cfg.framework:
                    if f.lower() == "hipt":
                        # save using hipt format
                        save_for_hipt(cfg, df_split, repetition_number=r)
                    elif f.lower() == "clam":
                        # save using clam format
                        save_for_clam(cfg, df_split, repetition_number=r)
                    else:
                        raise ValueError(
                            f"The given framework is not supported. Given {f}. If need support for this framework, see the definition of save_for_hipt of save_for_clam."
                        )

                rep_tqdm.update()

            # save the raw dataframe for this repetition. This can be used as dataset_description.csv in CLAM
            save_path = Path(cfg.output_dir, cfg.experiment_name)
            save_path.mkdir(parents=True, exist_ok=True)
            # re-order columns before saving
            col_order = (
                ["case_id", "slide_id", "label", "label_integer", "site_id"]
                if cfg.site_stratification
                else ["case_id", "slide_id", "label", "label_integer"]
            )
            [col_order.append(f"fold_{f+1}") for f in range(cfg.number_of_folds)]
            df_split = df_split[col_order]
            # df_split.to_csv(Path(save_path, f'dataset_description_{cfg.classification_level}_rep_{r}_folds_{cfg.number_of_folds}.csv'), index_label=False, index=False)
            df_split.to_csv(
                Path(save_path, f"dataset_descriptor.csv"),
                index_label=False,
                index=False,
            )

    elif cfg.split_strategy == "npb":
        with tqdm(total=cfg.npb_replicates, unit="replicas") as rep_tqdm:
            temp_split_dfs = []
            for r in range(cfg.npb_replicates):
                # update cfg to have number_of_folds=1
                cfg.number_of_folds = 1
                # create split for this replicate
                df_split = get_repetition_split_v2(
                    cfg,
                    df_for_split,
                    random_seed=cfg.random_seed + r,
                    print_summary=True if r == 0 else False,
                )

                # save in the format of the specified classification framework
                for f in cfg.framework:
                    if f.lower() == "hipt":
                        # save using hipt format
                        save_for_hipt(cfg, df_split, repetition_number=r)
                    elif f.lower() == "clam":
                        # save using clam format
                        save_for_clam(cfg, df_split, repetition_number=r)
                    else:
                        raise ValueError(
                            f"The given framework is not supported. Given {f}. If need support for this framework, see the definition of save_for_hipt of save_for_clam."
                        )

                    # add this 'fold' to the df_for_split dataframe
                    df_split = df_split.rename(columns={"fold_1": f"fold_{r+1}"})
                    temp_split_dfs.append(df_split[f"fold_{r+1}"])

                rep_tqdm.update()

            # add all the folds to the dataframe
            temp_split_dfs = pd.concat(temp_split_dfs, axis=1)
            df_for_split = pd.concat([df_for_split, temp_split_dfs], axis=1)

            # save the raw dataframe for this repetition. This can be used as dataset_description.csv in CLAM
            save_path = Path(cfg.output_dir, cfg.experiment_name)
            save_path.mkdir(parents=True, exist_ok=True)
            # re-order columns before saving
            col_order = (
                ["case_id", "slide_id", "label", "label_integer", "site_id"]
                if cfg.site_stratification
                else ["case_id", "slide_id", "label", "label_integer"]
            )
            [col_order.append(f"fold_{r+1}") for r in range(cfg.npb_replicates)]
            df_for_split = df_for_split[col_order]
            # df_split.to_csv(Path(save_path, f'dataset_description_{cfg.classification_level}_rep_{r}_folds_{cfg.number_of_folds}.csv'), index_label=False, index=False)
            df_for_split.to_csv(
                Path(save_path, f"dataset_descriptor.csv"),
                index_label=False,
                index=False,
            )
    else:
        raise ValueError(
            f"The given split strategy is not implemented. Given {cfg.split_strategy}. Implemented [cv, npb]"
        )

    # save a task .yaml template file (if requested)
    if cfg.save_classification_task_yaml_template:
        # build dictionary
        classification_task_template = {
            "description": cfg.classification_level,
            "data_root_dir": "",
            "csv_path": "",
            "split_dir": "",
            "n_classes": len(label_to_integer_map),
            "subtyping": True if len(label_to_integer_map) > 2 else False,
            "label_dict": label_to_integer_map,
            "ignore": [],
        }
        # save
        save_path = Path(
            cfg.output_dir, cfg.experiment_name, "classification_task_yaml_templates"
        )
        save_path.mkdir(parents=True, exist_ok=True)

        with open(
            os.path.join(save_path, f"{cfg.classification_level}.yaml"), "w"
        ) as outfile:
            yaml.dump(classification_task_template, outfile, default_flow_style=False)


if __name__ == "__main__":
    main()
# %%
