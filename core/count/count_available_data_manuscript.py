# %% 
import os 
import pandas as pd 
import copy
import numpy as np

# %% UTILITIES

def rearrange_dataset_summary(df):
    '''
    Utility that re-arranges the dataset summary dataframe to have one row for each unique subject with the needed 
    information the count.
    '''
    df_rearranged = copy.deepcopy(df)

    # get anonymized subject id from the anonymized code.
    get_subjects_from_anonymized_code = lambda x : x.split('_')[2]
    df_rearranged['SUBJECT_ID_ANONYMIZED'] = df_rearranged.ANONYMIZED_CODE.apply(get_subjects_from_anonymized_code)

    # get anonymized site id from the anonymized code.
    get_site_from_anonymized_code = lambda x : x.split('_')[1]
    df_rearranged['SITE'] = df_rearranged.ANONYMIZED_CODE.apply(get_site_from_anonymized_code)

    # compress
    df_rearranged = df_rearranged.groupby(['SUBJECT_ID_ANONYMIZED']).agg({
    'SITE' :lambda x : pd.unique(x)[0],
    'GENDER': lambda x : pd.unique(x)[0],
    'AGE_YEARS': lambda x : pd.unique(x)[0],
    'LOCATION': lambda x : pd.unique(x)[0], 
    'WHO_TUMOR_CATEGORY': lambda x : pd.unique(x)[0],
    'WHO_TUMOR_FAMILY' : lambda x : pd.unique(x)[0], 
    'WHO_TUMOR_TYPE': lambda x : pd.unique(x)[0], 
    'ANONYMIZED_CODE' : lambda x : len(x),
    })

    # fix gender
    def fix_gender(x):
        if x.GENDER in ('M', 'F'):
            return x.GENDER
        else:
            return 'NotAvailable'
    df_rearranged['GENDER'] = df_rearranged.apply( lambda x: fix_gender(x), axis=1)

    # fix age
    def fix_age(x):
        if x.AGE_YEARS == 'reserv PNR':
            return None
        else:
            return x.AGE_YEARS
    df_rearranged['AGE_YEARS'] = df_rearranged.apply( lambda x: fix_age(x), axis=1)

    return df_rearranged.reset_index()


def print_dataset_counts(df, pm='\u00B1'):
    # re arrange the dataframe to have the one row for each unique subject with the information
    df_temp = rearrange_dataset_summary(df)

    # print subject count
    # # age information overall
    mean_age = df_temp.AGE_YEARS.dropna().astype(int).mean()
    std_age = df_temp.AGE_YEARS.dropna().astype(int).std()
    min_age = df_temp.AGE_YEARS.dropna().astype(int).min()
    max_age = df_temp.AGE_YEARS.dropna().astype(int).max()

    print(f'Found {len(df_temp)} unique subjects (age [y]: {mean_age:0.2f} {pm} {std_age:0.2f}, range [{min_age:0.2f}, {max_age:0.2f}])')

    # # print per gender information
    for g, n in zip(('M', 'F', 'NotAvailable'), ('Male', 'Female', 'NA')):
        aus_df = df_temp.loc[df_temp.GENDER == g]
        subjects = len(aus_df)
        mean_age = aus_df.AGE_YEARS.dropna().astype(int).mean()
        std_age = aus_df.AGE_YEARS.dropna().astype(int).std()
        min_age = aus_df.AGE_YEARS.dropna().astype(int).min()
        max_age = aus_df.AGE_YEARS.dropna().astype(int).max()

        # print
        print(f'    {n:6s}: {subjects} {mean_age:0.2f} {pm} {std_age:0.2f}, range [{min_age:0.2f}, {max_age:0.2f}])')

    # pring the glass count if requested
    nbr_glasses = df_temp.ANONYMIZED_CODE.sum()
    print(f'Found {nbr_glasses} glasses (average {nbr_glasses/len(df_temp):.2f} glasses per subject).')

    # print per site information
    code_to_site = {
            '2233' : 'LUND', 
            '1036' : 'KS',
            '7371' : 'GOT',
            '4812' : 'LK',
            '6218' : 'UMEA',
        }
    per_site_count = df_temp.groupby(['SITE']).agg({'SUBJECT_ID_ANONYMIZED' : lambda x : len(pd.unique(x)), 'ANONYMIZED_CODE': lambda x : len(x), 'GENDER' : lambda x : {'M': sum(x == 'M'), 'F': sum(x=='F'), 'NA': sum(x=='NotAvailable')}})
    
    for c, n in code_to_site.items():
        try:
            subjects = per_site_count.loc[c, "SUBJECT_ID_ANONYMIZED"]
            nbr_male = per_site_count.loc[c, "GENDER"]["M"]
            nbr_female = per_site_count.loc[c, "GENDER"]["F"]
            nbr_NA = per_site_count.loc[c, "GENDER"]["NA"]
            glasses = per_site_count.loc[c, "ANONYMIZED_CODE"]
            print(f'Site: {n:5s}: {subjects:4d} subjects (M: {nbr_male}, F: {nbr_female}, NA: {nbr_NA}) ({glasses:4d} glasses)')
        except:
            continue
# %% PATHS
DATASET_CSV_PATH = '/local/data1/iulta54/Code/BTB_DEEP_LEARNING/dataset_csv_file/BTB_AGGREGATED_CLINICAL_AND_WSI_INFORMATION_KS_LK_GOT_UM_LUND_ANONYM_20240405.csv'
dataset_summary = pd.read_csv(DATASET_CSV_PATH, encoding="ISO-8859-1")
print(f'Found {len(dataset_summary)} entries.')

# %% REFINE TRUE FALSE 
d = {'True': True, 'False': False, 'UNMATCHED_WSI': 'UNMATCHED_WSI'}
dataset_summary['USE_DURING_ANALYSIS'] = dataset_summary['USE_DURING_ANALYSIS'].map(d)
d = {'TRUE': True, 'FALSE': False, 'UNMATCHED_WSI': 'UNMATCHED_WSI', 'UNMATCHED':'UNMATCHED'}
dataset_summary['ACCEPTABLE_IMAGE_QUALITY'] = dataset_summary['ACCEPTABLE_IMAGE_QUALITY'].map(d)
    
# %% REMOVE SUBJECTS THAT SHOULD NOT BE THERE (BTB2024_2233_6329_9960_9231, BTB2024_2233_5081_5038_9231)
to_remove = ('BTB2024_2233_6329_9960_9231', 'BTB2024_2233_5081_5038_9231')
dataset_summary = dataset_summary.loc[~dataset_summary.ANONYMIZED_CODE.isin(to_remove)]

# %% REMOVE WSIs THAT ARE UNMATCHED
to_remove = ['UNMATCHED_WSI']
dataset_summary = dataset_summary.loc[~dataset_summary.MATCH_CLINICAL_WSI_INFO.isin(to_remove)]

# %% PRINT NBR UNIQUE SUBJECTS
print_dataset_counts(dataset_summary)

# %% PRINT MISSING DIAGNOSIS
not_for_analysis = dataset_summary.loc[dataset_summary.USE_DURING_ANALYSIS != True]
dataset_summary = dataset_summary.loc[dataset_summary.USE_DURING_ANALYSIS == True]
print(f'Removing {len(not_for_analysis)} given USE_DURING_ANALYSIS != True')
print('\n############ REMOVING ############\n')
print_dataset_counts(not_for_analysis)
print('\n############ REMAINING ############n\n')
print_dataset_counts(dataset_summary)

# %% PRINT QUALITY CHECK FAIL 
not_for_analysis = dataset_summary.loc[dataset_summary.ACCEPTABLE_IMAGE_QUALITY != True]
dataset_summary = dataset_summary.loc[dataset_summary.ACCEPTABLE_IMAGE_QUALITY == True]
print(f'Removing {len(not_for_analysis)} given ACCEPTABLE_IMAGE_QUALITY != True')
print('\n############ REMOVING ############\n')
print_dataset_counts(not_for_analysis)
print('\n############ REMAINING ############n\n')
print_dataset_counts(dataset_summary)
# %% PRINT DIAGNOSIS COUNTS (at all the levels)
get_subjects_from_anonymized_code = lambda x : x.split('_')[2]
dataset_summary['SUBJECT_ID'] = dataset_summary.ANONYMIZED_CODE.apply(get_subjects_from_anonymized_code)
get_glass_id_from_anonymized_code = lambda x : x.split('_')[3]
dataset_summary['GLASS_ID_CLINICAL'] = dataset_summary.ANONYMIZED_CODE.apply(get_glass_id_from_anonymized_code)

# build counts for each TUMOR cluster (category, family and type)
for_analysis = copy.deepcopy(dataset_summary)

gb = for_analysis.groupby(['WHO_TUMOR_CATEGORY']).agg({'SUBJECT_ID': lambda x : len(pd.unique(x))})
for_analysis = for_analysis.merge(gb, on='WHO_TUMOR_CATEGORY', suffixes=('', '_TUMOR_CATEGORY_COUNT'))
gb = for_analysis.groupby(['WHO_TUMOR_CATEGORY']).agg({'GLASS_ID_CLINICAL': lambda x : len(x)})
for_analysis = for_analysis.merge(gb, on='WHO_TUMOR_CATEGORY', suffixes=('', '_TUMOR_CATEGORY_COUNT'))

gb = for_analysis.groupby(['WHO_TUMOR_CATEGORY', 'WHO_TUMOR_FAMILY']).agg({'SUBJECT_ID': lambda x : len(pd.unique(x))})
for_analysis = for_analysis.merge(gb, on=['WHO_TUMOR_CATEGORY','WHO_TUMOR_FAMILY'], suffixes=('', '_TUMOR_FAMILY_COUNT'))
gb = for_analysis.groupby(['WHO_TUMOR_CATEGORY','WHO_TUMOR_FAMILY']).agg({'GLASS_ID_CLINICAL': lambda x : len(x)})
for_analysis = for_analysis.merge(gb, on=['WHO_TUMOR_CATEGORY','WHO_TUMOR_FAMILY'], suffixes=('', '_TUMOR_FAMILY_COUNT'))

gb = for_analysis.groupby(['WHO_TUMOR_CATEGORY','WHO_TUMOR_FAMILY', 'WHO_TUMOR_TYPE']).agg({'SUBJECT_ID': lambda x : len(pd.unique(x))})
for_analysis = for_analysis.merge(gb, on=['WHO_TUMOR_CATEGORY','WHO_TUMOR_FAMILY', 'WHO_TUMOR_TYPE'], suffixes=('', '_TUMOR_TYPE_COUNT'))
gb = for_analysis.groupby(['WHO_TUMOR_CATEGORY','WHO_TUMOR_FAMILY', 'WHO_TUMOR_TYPE']).agg({'GLASS_ID_CLINICAL': lambda x : len(x)})
for_analysis = for_analysis.merge(gb, on=['WHO_TUMOR_CATEGORY','WHO_TUMOR_FAMILY', 'WHO_TUMOR_TYPE'], suffixes=('', '_TUMOR_TYPE_COUNT'))

for d in ('TUMOR_CATEGORY', 'TUMOR_FAMILY', 'TUMOR_TYPE'):
    print(f'{d}')
    # get unique labels for this level
    unique_labels = pd.unique(for_analysis[f'WHO_{d}'])
    # sort the labels based in the number of subjects
    nbr_subjects = [for_analysis.loc[for_analysis[f'WHO_{d}']==l][f'SUBJECT_ID_{d}_COUNT'].max() for l in unique_labels]
    nbr_glasses = [for_analysis.loc[for_analysis[f'WHO_{d}']==l][f'GLASS_ID_CLINICAL_{d}_COUNT'].max() for l in unique_labels]
    
    unique_labels = [x for (y,x) in sorted(zip(nbr_subjects, unique_labels), key=lambda pair: pair[0])]
    nbr_glasses
    # print stats for each 
    for l in unique_labels:
        subjects = for_analysis.loc[for_analysis[f'WHO_{d}']==l][f'SUBJECT_ID_{d}_COUNT'].max()
        glasses = for_analysis.loc[for_analysis[f'WHO_{d}']==l][f'GLASS_ID_CLINICAL_{d}_COUNT'].max()
        print(f'    {l:70s}: {subjects} subjects, {glasses} glasses')

# 
# tumor_category_family_type_aggregation = for_analysis.groupby(['WHO_TUMOR_CATEGORY','WHO_TUMOR_FAMILY', 'WHO_TUMOR_TYPE'], dropna=False).agg({'SUBJECT_ID_TUMOR_CATEGORY_COUNT': lambda x : max(x), 
#                                                                                                                                              'GLASS_ID_CLINICAL_TUMOR_CATEGORY_COUNT': lambda x : max(x),
#                                                                                                                                              'SUBJECT_ID_TUMOR_FAMILY_COUNT': lambda x : max(x),
#                                                                                                                                              'GLASS_ID_CLINICAL_TUMOR_FAMILY_COUNT': lambda x : max(x),
#                                                                                                                                              'SUBJECT_ID_TUMOR_TYPE_COUNT': lambda x : max(x),
#                                                                                                                                              'GLASS_ID_CLINICAL_TUMOR_TYPE_COUNT': lambda x : max(x),
#                                                                                                                                              })
# print(tumor_category_family_type_aggregation)

# %% PRINT THOSE THAT HAVE EXTRACTED FEATURES

PATH_TO_PATCHES = '/run/media/iulta54/Expansion/Datasets/BTB/SCRIPTS/pre_processing/outputs/clam/BTB_patch_extraction_x20_224/2024-04-19/patches'
PATH_TO_FEATURES = '/local/data2/iulta54/Data/BTB/histology_features/clam_features_mag_x20_size_224/vit_hipt/pt_files'

patch_list = [i.split('.')[0] for i in os.listdir(PATH_TO_PATCHES)]
feature_file_list = [i.split('.')[0] for i in os.listdir(PATH_TO_FEATURES)]

ids_without_patches = dataset_summary.loc[~dataset_summary.ANONYMIZED_CODE.isin(patch_list)]
ids_without_features = dataset_summary.loc[~dataset_summary.ANONYMIZED_CODE.isin(feature_file_list)]

print(f'Glasses with missing patching: {len(ids_without_patches)}')
print(f'Glasses with missing features: {len(ids_without_features)}')