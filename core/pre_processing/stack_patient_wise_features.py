# %% IMPORTS

'''
This script aggregates the features at UNIQUE CASE ID LEVEL from the single WSI-level features.
A unique case id is identified by the combination of the following columns:
- SUBJECT_ID (not UNIQUE_SUBJECT_ID since this will concatenate WSI from the same patient but at different time points, eg. same diagnosis for N and N_reop).
- TUMOR_CATEGORY
- TUMOR_FAMILY
- TUMOR_TYPE
'''
import torch
import os
import re
import copy
import numpy as np
import pathlib
from tqdm import tqdm
import pandas as pd
import hashlib

# UTILITIES

def get_centre_id_from_anonymised_code(x:str):
    return x.split('_')[1]

def get_subject_id_from_anonymised_code(x:str):
    return x.split('_')[2]

def get_case_id_from_anonymised_code(x:str):
    return x.split('_')[3]

# %% PATHS

ROOT_PER_WSI_FEATURES = '/run/media/iulta54/Expansion/Datasets/BTB/EXTRACTED_FEATURES/wsi_level_features/clam_features_mag_x20_size_224'
PER_SUBJECT_FEATURES_OUTPUT_DIR = '/run/media/iulta54/Expansion/Datasets/BTB/EXTRACTED_FEATURES/subject_case_level_features/clam_features_mag_x20_size_224'
TABULAR_DATA_FILE = '/local/d2/iulta54/Research/P7_BTB_DEEP_LEARNING/dataset_csv_file/BTB_merged_clinical_image_quality_information.xlsx'

# %% LOOAD THE TABULAR DATA 
tabular_data = pd.read_excel(TABULAR_DATA_FILE)

# get unique subject and case IDs
pattern = r"[-_]+"
tabular_data["UNIQUE_SUBJECT_ID"] = tabular_data["SUBJECT_ID"].apply(
    lambda x: (
        int(re.split(pattern, str(x))[0])
        if pd.notnull(x) and re.split(pattern, str(x))[0].isdigit()
        else None
    )
)
 
# add UNIQUE_CASE_ID column as a combination of UNIQUE_SUBJECT_ID and TUMOR_CATEGORY, TUMOR_FAMILY adn TUMOR_TYPE
tabular_data["UNIQUE_CASE_ID"] = tabular_data.apply(
    lambda x: f"{x.SUBJECT_ID}_{x.TUMOR_CATEGORY}_{x.TUMOR_FAMILY}_{x.TUMOR_TYPE}",
    axis=1,
)

# %% ADD THE HASH CODE FOR THE SUBJECT and CASE ID
tabular_data["CENTRE_ID_HASH"] = tabular_data["ANONYMIZED_CODE"].apply(
    lambda x: get_centre_id_from_anonymised_code(str(x))
)

tabular_data["SUBJECT_ID_HASH"] = tabular_data["ANONYMIZED_CODE"].apply( 
    lambda x: get_subject_id_from_anonymised_code(str(x))
)

tabular_data["CASE_ID_HASH"] = tabular_data["ANONYMIZED_CODE"].apply(
    lambda x: get_case_id_from_anonymised_code(str(x))
)

# %% PRINT SUMMARY BEFORE FILTERING
print(f'Before filtering, we have {len(tabular_data)} rows of data.')
print(f'Unique subjects: {len(tabular_data["SUBJECT_ID_HASH"].unique())}')
print(f'Unique cases: {len(tabular_data["CASE_ID_HASH"].unique())}')

# %% APPLY FILTERS TO ONLY WORK ON THOSE FILES THAT ARE RELEVANT)


# build filters to only work on the relevant data, here using the dame filters as per the BTB dataset descriptor
tabular_data = tabular_data.loc[
    tabular_data.MATCH_CLINICAL_WSI_INFO != "UNMATCHED_CLINICAL"
]
 
# filter those that do not have a WSI due to scanning issues (brocken glass or too small tissue)
tabular_data = tabular_data.loc[
    tabular_data.WSI_FILE_PATH != "NOT_AVAILABLE"
]
 
# filter out those rows that do not have a subject ID
tabular_data = tabular_data.loc[
    ~tabular_data["UNIQUE_SUBJECT_ID"].isnull()
]
 
# filter those that do not have a good image quality

tabular_data = tabular_data.loc[
    tabular_data.ACCEPTABLE_IMAGE_QUALITY == True
]
  
# filter out those that are Insufficient material for diagnosis
tabular_data = tabular_data.loc[
    tabular_data.DIAGNOSIS_CLEANED_COHERENT
    != "Insufficient material for diagnosis"
]
 
 
# filter out those that are not tumor cases
str_filter = ["Not tumor", "Inflammation"]
tabular_data = tabular_data.loc[
    ~tabular_data.DIAGNOSIS_CLEANED_COHERENT.isin(str_filter)
]
 
# filter out those that are not malignant
tabular_data = tabular_data.loc[
    tabular_data.DIAGNOSIS_CLEANED_COHERENT != "Not malignant"
]
 
 
# filter those that are not in the CNS
tabular_data = tabular_data.loc[
    tabular_data.SUPRA_INFRA_SPINAL_MENINGI != "NOT_CNS"]

# %% PRINT SUMMARY OF UNIQUE SUBJECTS and CASES
print(f'After filtering, we have {len(tabular_data)} rows of data.')
print(f'Unique subjects: {len(tabular_data["SUBJECT_ID_HASH"].unique())}')
print(f'Unique cases: {len(tabular_data["CASE_ID_HASH"].unique())}')

# %% BUILD A LIST OF TUPLES WITH THE FILES TO STACK USING THE INFORMATION FROM THE TABULAR DATA
# group by the unique case ID
summary_df = tabular_data.groupby('CASE_ID_HASH').agg({'CENTRE_ID_HASH': lambda x : pd.unique(x)[0],'SUBJECT_ID_HASH': lambda x : pd.unique(x)[0],'UNIQUE_CASE_ID': lambda x : pd.unique(x)[0], 'ANONYMIZED_CODE': lambda x : list(x)}).reset_index()
summary_df['NBR_WSI'] = summary_df['ANONYMIZED_CODE'].apply(lambda x : len(x))


# %% BUILD PATHS TO THE FEATURES MASED ON THE FEATURE EXTRACTOR USED
feature_extractor = 'vit_conch'
PER_WSI_FEATURES = os.path.join(ROOT_PER_WSI_FEATURES, feature_extractor, 'pt_files')
# check if the path exists
if not os.path.exists(PER_WSI_FEATURES):
    raise ValueError(f'Path to WSI-level features does not exist: {PER_WSI_FEATURES}')
else:
    print(f'Loading WSI-level features from: {PER_WSI_FEATURES}')

PER_SUBJECT_FEATURES_OUTPUT_DIR = os.path.join(PER_SUBJECT_FEATURES_OUTPUT_DIR, feature_extractor, 'pt_files')
# make output dir if not existing
pathlib.Path(PER_SUBJECT_FEATURES_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
print(f'Saving per-subject-case features at: {PER_SUBJECT_FEATURES_OUTPUT_DIR}')

# %% STACK THE WSI_FEATURE FROM EACH SUBJECT-CASE

for r in tqdm(summary_df.iterrows()):
    # get the row
    s = r[1]

    centre_id = s.CENTRE_ID_HASH
    subject_id = s.SUBJECT_ID_HASH
    case_id = s.CASE_ID_HASH
    anonymised_codes = s.ANONYMIZED_CODE

    # build file name
    output_file_path = os.path.join(PER_SUBJECT_FEATURES_OUTPUT_DIR, '_'.join(['BTB2024', centre_id, subject_id, case_id,])+'.pt')

    # process only if the file does not exist
    if os.path.exists(output_file_path):
        continue
    else:
        # build full file paths based on the anonymised codes
        tensors_to_stack = []
        for f in anonymised_codes:
            tensor_path = os.path.join(PER_WSI_FEATURES, f+'.pt')
            try:
                tensor = torch.load(tensor_path, weights_only=True)
                tensors_to_stack.append(tensor)
            except:
                print(f'Error loading tensor from: {tensor_path}')
        
        # stack tensors
        stacked_tensor = torch.cat(tensors_to_stack, dim=0)

        # # save
        torch.save(stacked_tensor, output_file_path)

# %%