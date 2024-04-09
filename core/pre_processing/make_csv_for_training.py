# %% 
'''
Utility that given the summary file (overall_summary_data_with_wsi_paths_date.csv),
creates the .csv files needed to run HIPT or CLAM classification training.
'''

import os
import hydra
import datetime
import pandas as pd

from pathlib import Path
import omegaconf
from omegaconf import DictConfig

from sklearn.model_selection import (
    KFold,
    train_test_split,
    StratifiedKFold,
    StratifiedGroupKFold,
    GroupShuffleSplit,
    StratifiedShuffleSplit,
)

# %% UTILITIES

def case_id_from_anonymized_code(x:str):
    '''
    The BTB anonymized codes are in the following format.
    BTB2024_site_case_pad_glass-id 
    '''
    return x.split('_')[2]

def site_id_from_anonymized_code(x:str):
    '''
    The BTB anonymized codes are in the following format.
    BTB2024_site_case_pad_glass-id 
    '''
    try:
        return x.split('_')[1]
    except:
        print(x.split('_'))

def get_repetition_split(cfg:DictConfig, df, random_seed:int=29122009, print_summary:bool=False):
    '''
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
    '''

    if print_summary:
        # print summary before start splitting
        print_df_summary(df)

    
    # ################## work on splitting
    if cfg.class_stratification:
        print("Performing stratified data split (on a per case_id/subject bases).")

        # ################## TEST SET
        # perform stratified split
        sgkf = StratifiedGroupKFold(
            n_splits=int(1 / cfg.test_fraction),
            shuffle=True,
            random_state=random_seed,
        )

        train_val_ix, test_ix = next(
            sgkf.split(df, y=df.label, groups=df.case_id)
        )
    
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
        sgkf = StratifiedGroupKFold(
            n_splits=cfg.number_of_folds
            if cfg.number_of_folds != 1
            else 2,
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
        train_val_ix, test_ix = next(
            gs.split(df, groups=df.case_id)
        )

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

def print_df_summary(df):
    # print totals first 
    print(f'Number of slides: {len(df)}')
    print(f'Number of unique case_ids (subjects): {len(pd.unique(df.case_id))}')
    if 'site_id' in df.columns:
        print(f'Number of sites: {len(pd.unique(df.site_id))}')
    print(f'Number of unique classes/labels: {len(pd.unique(df.label))}')

    # break down on a class level
    if 'site_id' in df.columns:
        aus = df.groupby(['label']).agg({'case_id': lambda x : len(pd.unique(x)), 'slide_id': lambda x : len(x), 'site_id': lambda x : len(pd.unique(x))})
    else:
        aus = df.groupby(['label']).agg({'case_id': lambda x : len(pd.unique(x)), 'slide_id': lambda x : len(x)})
    print(aus)

def save_for_hipt(cfg, df, repetition_number):
    '''
    See README.md file for a detailed description of how the HIPT framework needs the .csv files saved for training and evaluation (tune).
    '''

    # make save_path 
    save_path = Path(cfg.output_dir, cfg.experiment_name, 'hipt', f'repetition_{repetition_number}', f'fold_{fold+1}')
    save_path.mkdir(parents=True, exist_ok=True)

    for fold in range(cfg.number_of_folds):
        for split, file_name in zip(['train', 'validation', 'test'], ['train.csv', 'tune.csv', 'test.csv']):
            # filter dataframe 
            df_for_save = df.loc[df[f'fold_{fold+1}']==split]
            df_for_save = df_for_save[['slide_id', 'label_integer']]
            df_for_save = df_for_save.rename(columns={'label_integer':'label'})

            # save 
            save_path = Path(cfg.output_dir, cfg.experiment_name, 'hipt', f'repetition_{repetition_number}', f'fold_{fold+1}')
            save_path.mkdir(parents=True, exist_ok=True)
            df_for_save.to_csv(Path(save_path, file_name), index_label=False, index=False)


def save_for_clam(cfg, df, repetition_number):
    '''
    See README.md file for a detailed description of how the CLAM framework needs the .csv files saved for training and evaluation (tune).
    '''

    # make save_path
    save_path = Path(cfg.output_dir, cfg.experiment_name, 'clam', f'repetition_{repetition_number}')
    save_path.mkdir(parents=True, exist_ok=True)

    for fold in range(cfg.number_of_folds):
        # # make and save split_nbr.csv file
        df_for_save = df[[f'fold_{fold+1}', 'slide_id']]
        df_for_save = df_for_save.rename(columns={f'fold_{fold+1}':'set'})
        
        # combine and take out each set
        gb = df_for_save.groupby(['set'])
        train = gb.get_group(('train',)).drop(columns=['set']).rename(columns={'slide_id':'train'}).reset_index()
        validation = gb.get_group(('validation',)).drop(columns=['set']).rename(columns={'slide_id':'val'}).reset_index()
        test = gb.get_group(('test',)).drop(columns=['set']).rename(columns={'slide_id':'test'}).reset_index()

        # concatenate and save
        split_to_save = pd.concat([train, validation, test], axis=1).drop(columns=['index'])
        split_to_save.to_csv(Path(save_path, f'split_{fold}.csv'), index_label=False, index=False)

        # # make and save split_nbr_bool.csv file
        df_for_save['train'] = df_for_save.apply(lambda x : x.set == 'train', axis=1)
        df_for_save['val'] = df_for_save.apply(lambda x : x.set == 'validation', axis=1)
        df_for_save['test'] = df_for_save.apply(lambda x : x.set == 'test', axis=1)
        
        # refine and save
        split_bool_to_save = df_for_save.drop(columns=['set'])
        split_bool_to_save.to_csv(Path(save_path, f'split_{fold}_bool.csv'), index_label=False, index=False)

        # # make and save split_nbr_descriptor.csv file 
        df_for_save = df[[f'fold_{fold+1}', 'slide_id', 'label']]
        df_for_save = df_for_save.rename(columns={f'fold_{fold+1}':'set'})
        
        # create dummy columns needed for later
        df_for_save['train'] = df_for_save.apply(lambda x : x.set == 'train', axis=1)
        df_for_save['val'] = df_for_save.apply(lambda x : x.set == 'validation', axis=1)
        df_for_save['test'] = df_for_save.apply(lambda x : x.set == 'test', axis=1)

        gb = df_for_save.groupby(['label']).agg({'train': 'sum', 'val': 'sum', 'test': 'sum'})
        gb.to_csv(Path(save_path, f'split_{fold}_descriptor.csv'), index_label='class', index=True)


def get_label_to_integer_map(unique_labels:list):
    '''
    Returns a dictionary with keys the string for a label and as value an integer.
    Here using alphabetical order and integers form 0 to len(unique_labels)
    '''
    unique_labels.sort()
    label_to_integer_map = dict.fromkeys(unique_labels)
    for k_index, k in enumerate(label_to_integer_map.keys()):
        label_to_integer_map[k] = k_index
    
    return label_to_integer_map
    

# %% LOAD CONFIGURATION (dev)
        
cfg = omegaconf.OmegaConf.load('/run/media/iulta54/Expansion/Datasets/BTB/SCRIPTS/pre_processing/config/make_csv_for_training.yaml')

# %% MAIN


# @hydra.main(
#     version_base="1.2.0", config_path="config", config_name="make_csv_for_training"
# )
# def main(cfg: DictConfig):


# load BT_csv file
btb_csv = pd.read_csv(cfg.btb_csv_path, encoding="ISO-8859-1")
# make sure we have bools in USE_DURING_ANALYSIS and ACCEPTABLE_IMAGE_QUALITY columns
d = {'True': True, 'False': False, 'UNMATCHED_WSI': 'UNMATCHED_WSI'}
btb_csv['USE_DURING_ANALYSIS'] = btb_csv['USE_DURING_ANALYSIS'].map(d)
d = {'TRUE': True, 'FALSE': False, 'UNMATCHED_WSI': 'UNMATCHED_WSI', 'UNMATCHED':'UNMATCHED'}
btb_csv['ACCEPTABLE_IMAGE_QUALITY'] = btb_csv['ACCEPTABLE_IMAGE_QUALITY'].map(d)

# include only those that are acceptable for the analysis (USE_DURING_ANALYSIS==True & ACCEPTABLE_IMAGE_QUALITY==True)
btb_csv = btb_csv.loc[(btb_csv.USE_DURING_ANALYSIS==True) & (btb_csv.ACCEPTABLE_IMAGE_QUALITY==True)]

# remove class_labels or sites from the dataset if requested
if cfg.classes_to_exclude:
    btb_csv = btb_csv.loc[~btb_csv[cfg.class_column_name].isin(cfg.classes_to_exclude)]
if cfg.site_to_exclude:
    btb_csv = btb_csv.loc[~btb_csv[cfg.site_column_name].isin(cfg.site_to_exclude)]

# create a new Dataframe with only the ANONYMIZED_CODE, CLASS_LABEL and SITE (if needed).
# Use the class_column to get the class_label at the right classification level.
df_for_split = btb_csv[['ANONYMIZED_CODE', cfg.class_column_name]]
df_for_split = df_for_split.rename(columns={'ANONYMIZED_CODE':'slide_id', cfg.class_column_name:'label'})

# check if the slide_ids are available as extracted features
if cfg.check_available_features:
    slide_ids = list(df_for_split.slide_id.values)
    feature_check = [os.path.isfile(os.path.join(cfg.feature_dir, sid+'.pt')) for sid in slide_ids]
    if not all(feature_check):
        print(f'Not all slide_ids have available feature files.\nFound {feature_check.count(True)}.\nMissing {feature_check.count(False)} out of {len(slide_ids)} ({feature_check.count(False) / len(slide_ids) * 100:0.2f}%).')
    
    if cfg.skip_missing_features:
        print(f'ATTENTION! Removing the missing slide_ids (skip_missing_features == {cfg.skip_missing_features})')
        slide_id_with_features = [slide_ids[i] for i, c in enumerate(feature_check) if c]
        df_for_split = df_for_split.loc[df_for_split.slide_id.isin(slide_id_with_features)]

# get case_id (subject id) from the anonymized codes (slide_id). This is needed to perform a per-case/subject split
df_for_split['case_id'] = df_for_split.apply(lambda x : case_id_from_anonymized_code(x.slide_id), axis=1)
# get site_id if site stratification
if cfg.site_stratification:
    df_for_split['site_id'] = df_for_split.apply(lambda x : site_id_from_anonymized_code(x.slide_id), axis=1)

# map the class string to an integer (starts from 0)
label_to_integer_map = get_label_to_integer_map(list(pd.unique(df_for_split.label))) 
df_for_split['label_integer'] = df_for_split.apply(lambda x : label_to_integer_map[x.label], axis=1)

# reset index prior splitting
df_for_split = df_for_split.reset_index()

# start splitting, one split per repetition
for r in range(cfg.number_of_repetitions):
    # create split for this repetition
    df_split = get_repetition_split(cfg, df_for_split, random_seed=cfg.random_seed+r, print_summary=False)

    # save in the format of the specified classification framework
    for f in cfg.framework:
        if f.lower() == 'hipt':
            # save using hipt format
            save_for_hipt(cfg, df_split, repetition_number=r)
        elif f.lower() == 'clam':
            # save using clam format
            save_for_clam(cfg, df_split, repetition_number=r)
        else:
            raise ValueError(f'The given framework is not supported. Given {f}. If need support for this framework, see the definition of save_for_hipt of save_for_clam.')

    # save the raw dataframe for this repetition. This can be used as dataset_description.csv in CLAM
    save_path = Path(cfg.output_dir, cfg.experiment_name, 'dataset_summary', f'repetition_{r}')
    save_path.mkdir(parents=True, exist_ok=True)
    # re-order columns before saving 
    col_order = ['case_id', 'slide_id', 'label', 'label_integer', 'site_id']
    [col_order.append(f'fold_{f+1}') for f in range(cfg.number_of_folds)]
    df_split= df_split[col_order]
    df_split.to_csv(Path(save_path, f'dataset_description_repetition_{r}.csv'), index_label=False, index=False)



# # save file
# # # 
# # run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M")
# run_id = datetime.datetime.now().strftime("%Y-%m-%d")
# # make output directory
# output_dir = Path(cfg.output_dir, cfg.experiment_name, run_id)
# output_dir.mkdir(parents=True, exist_ok=True)

# hs2p_csv.to_csv(os.path.join(output_dir, 'BTB_for_hs2p.csv'), index_label=False, index=False)

# print(f'BTB_hs2p.csv file saved at {output_dir}')

# if __name__ == '__main__':
#     main()
# %%