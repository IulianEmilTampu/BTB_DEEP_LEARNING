# %%
import os
import pandas as pd
import copy
import numpy as np
from datetime import datetime

# %% UTILITIES


def rearrange_dataset_summary(df):
    """
    Utility that re-arranges the dataset summary dataframe to have one row for each unique subject-case with the needed
    information the count.
    """
    df_rearranged = copy.deepcopy(df)

    # get anonymized subject id from the anonymized code.
    """
    From the anonymization code, index n. specifies:
        0: Dataset name (BTB2024)
        1: site/centre
        2: subject code (stripped from any '_'. Thus, if original case number is 154_M1 -> 154)
        3: diagnosis. This is the hex hash code for the string containing tumor category, family and type information
        4: pad number
        5: glass id code
    """
    # this selects all the unique subject-diagnosis pairs. There can be subjects which glasses are diagnosed differently.
    get_subjects_from_anonymized_code = lambda x: "_".join(x.split("_")[2:4])
    df_rearranged["SUBJECT_DIAGNOSIS_ID_ANONYMIZED"] = (
        df_rearranged.ANONYMIZED_CODE.apply(get_subjects_from_anonymized_code)
    )

    # only subject (not considering multiple diagnosis)
    get_subjects_from_anonymized_code = lambda x: "_".join(x.split("_")[2:3])
    df_rearranged["SUBJECT_ID_ANONYMIZED"] = df_rearranged.ANONYMIZED_CODE.apply(
        get_subjects_from_anonymized_code
    )

    # get anonymized site id from the anonymized code.
    get_site_from_anonymized_code = lambda x: x.split("_")[1]
    df_rearranged["SITE"] = df_rearranged.ANONYMIZED_CODE.apply(
        get_site_from_anonymized_code
    )

    # compress based on subject id only (no duble diagnosis)
    df_rearranged = df_rearranged.groupby(["SUBJECT_DIAGNOSIS_ID_ANONYMIZED"]).agg(
        {
            "SUBJECT_ID_ANONYMIZED": lambda x: pd.unique(x)[0],
            "SITE": lambda x: pd.unique(x)[0],
            "GENDER": lambda x: pd.unique(x)[0],
            "AGE_YEARS": lambda x: pd.unique(x)[0],
            "LOCATION": lambda x: pd.unique(x)[0],
            "WHO_TUMOR_CATEGORY": lambda x: pd.unique(x)[0],
            "WHO_TUMOR_FAMILY": lambda x: pd.unique(x)[0],
            "WHO_TUMOR_TYPE": lambda x: pd.unique(x)[0],
            "ANONYMIZED_CODE": lambda x: len(x),
        }
    )

    # fix gender
    def fix_gender(x):
        if x.GENDER in ("M", "F"):
            return x.GENDER
        else:
            return "NotAvailable"

    df_rearranged["GENDER"] = df_rearranged.apply(lambda x: fix_gender(x), axis=1)

    # fix age
    def fix_age(x):
        if x.AGE_YEARS == "reserv PNR":
            return None
        else:
            return x.AGE_YEARS

    df_rearranged["AGE_YEARS"] = df_rearranged.apply(lambda x: fix_age(x), axis=1)

    return df_rearranged.reset_index()


def print_dataset_counts(df, pm="\u00B1"):
    # re arrange the dataframe to have the one row for each unique subject with the information
    df_temp = rearrange_dataset_summary(df)

    # print subject count
    # # age information overall
    mean_age = df_temp.AGE_YEARS.dropna().astype(int).mean()
    std_age = df_temp.AGE_YEARS.dropna().astype(int).std()
    min_age = df_temp.AGE_YEARS.dropna().astype(int).min()
    max_age = df_temp.AGE_YEARS.dropna().astype(int).max()

    print(
        f"Found {len(pd.unique(df_temp.SUBJECT_ID_ANONYMIZED))} subjects [{len(df_temp)} subjects-diagnosis pairs [SDPs]] (age [y]: {mean_age:0.2f} {pm} {std_age:0.2f}, range [{min_age:0.2f}, {max_age:0.2f}])"
    )

    # # print per gender information
    for g, n in zip(("M", "F", "NotAvailable"), ("Male", "Female", "NA")):
        aus_df = df_temp.loc[df_temp.GENDER == g]
        subjects_diagnosis = len(aus_df)
        subjects = len(pd.unique(aus_df.SUBJECT_ID_ANONYMIZED))
        mean_age = aus_df.AGE_YEARS.dropna().astype(int).mean()
        std_age = aus_df.AGE_YEARS.dropna().astype(int).std()
        min_age = aus_df.AGE_YEARS.dropna().astype(int).min()
        max_age = aus_df.AGE_YEARS.dropna().astype(int).max()

        # print
        print(
            f"    {n:6s}: {subjects:4d} subjects [{subjects_diagnosis:4d} SDPs] {mean_age:0.2f} {pm} {std_age:0.2f}, range [{min_age:0.2f}, {max_age:0.2f}])"
        )

    # pring the glass count if requested
    nbr_glasses = df_temp.ANONYMIZED_CODE.sum()
    print(
        f"Found {nbr_glasses} glasses (average {nbr_glasses/len(df_temp):.2f} glasses per subject)."
    )

    # print per site information
    code_to_site = {
        "5e4761c2": "LUND",
        "fc173989": "KS",
        "103f236b": "GOT",
        "6c730372": "LK",
        "9a2a64c4": "UMEA",
        "9fb809d6": "UPPSALA",
    }
    len_characters_site = max([len(s) for s in code_to_site.keys()])
    per_site_count = df_temp.groupby(["SITE"]).agg(
        {
            "SUBJECT_ID_ANONYMIZED": lambda x: len(pd.unique(x)),
            "SUBJECT_DIAGNOSIS_ID_ANONYMIZED": lambda x: len(pd.unique(x)),
            "ANONYMIZED_CODE": lambda x: sum(x),
            "GENDER": lambda x: {
                "M": sum(x == "M"),
                "F": sum(x == "F"),
                "NA": sum(x == "NotAvailable"),
            },
        }
    )

    for c, n in code_to_site.items():
        # try:
        # filter df to only get the data from this site
        aus_df = df_temp.loc[df_temp.SITE == c]
        # get the remaining information
        ## number of subjects and subjects-diagnosis pairs
        subject_diagnosis = len(pd.unique(aus_df.SUBJECT_DIAGNOSIS_ID_ANONYMIZED))
        subjects = len(pd.unique(aus_df.SUBJECT_ID_ANONYMIZED))

        ## number of males (subjects and subjects-diagnosis pairs)
        nbr_male_subjects_diagnosis = len(aus_df.loc[aus_df.GENDER == "M"])
        nbr_male_subjects = len(
            pd.unique(aus_df.loc[aus_df.GENDER == "M"].SUBJECT_ID_ANONYMIZED)
        )

        ## number of females (subjects and subjects-diagnosis pairs)
        nbr_female_subjects_diagnosis = len(aus_df.loc[aus_df.GENDER == "F"])
        nbr_female_subjects = len(
            pd.unique(aus_df.loc[aus_df.GENDER == "F"].SUBJECT_ID_ANONYMIZED)
        )

        ## number of NA (subjects and subjects-diagnosis pairs)
        nbr_NA_subjects_diagnosis = len(aus_df.loc[aus_df.GENDER == "NotAvailable"])
        nbr_NA_subjects = len(
            pd.unique(aus_df.loc[aus_df.GENDER == "NotAvailable"].SUBJECT_ID_ANONYMIZED)
        )

        ## number of WSIs/glasses
        glasses = len(aus_df)

        print(
            f"Site: {n:{len_characters_site}s}: {subjects:4d} subjects [{subject_diagnosis:4d} SDPs] (M: {nbr_male_subjects} [{nbr_male_subjects_diagnosis} SDPs], F: {nbr_female_subjects} [{nbr_female_subjects_diagnosis} SDPs], NA: {nbr_NA_subjects} [{nbr_NA_subjects_diagnosis} SDPs]) ({glasses:4d} glasses)"
        )
    # except:
    #     continue

    # for c, n in code_to_site.items():
    #     try:
    #         subject_diagnosis = per_site_count.loc[c, "SUBJECT_DIAGNOSIS_ID_ANONYMIZED"]
    #         subjects = per_site_count.loc[c, "SUBJECT_ID_ANONYMIZED"]
    #         nbr_male_subjects_diagnosis = per_site_count.loc[c, "GENDER"]["M"]
    #         nbr_female_subjects_diagnosis = per_site_count.loc[c, "GENDER"]["F"]
    #         nbr_NA_subjects_diagnosis = per_site_count.loc[c, "GENDER"]["NA"]
    #         glasses = per_site_count.loc[c, "ANONYMIZED_CODE"]
    #         print(f'Site: {n:{len_characters_site}s}: {subjects:4d} subjects, {subject_diagnosis:4d} SDPs (M: ToDo [{nbr_male_subjects_diagnosis} SDPs], F: ToDo [{nbr_female_subjects_diagnosis} SDPs], NA: ToDo [{nbr_NA_subjects_diagnosis} SDPs]) ({glasses:4d} glasses)')
    #     except:
    #         continue


# %% PATHS
DATASET_CSV_PATH = "/local/data1/iulta54/Code/BTB_DEEP_LEARNING/dataset_csv_file/BTB_AGGREGATED_CLINICAL_AND_WSI_INFORMATION_KS_LK_GOT_UM_LUND_UPP_ANONYM_20240704_UPDATED_20241205.csv"
# DATASET_CSV_PATH = "/local/data1/iulta54/Code/BTB_DEEP_LEARNING/dataset_csv_file/BTB_AGGREGATED_CLINICAL_AND_WSI_INFORMATION_KS_LK_GOT_UM_LUND_UPP_ANONYM_20240704.csv"
dataset_summary = pd.read_csv(DATASET_CSV_PATH, encoding="ISO-8859-1")
print(f"Found {len(dataset_summary)} entries.")

# %% DEFINE WHERE TO SAVE ALL THE REMOVED CASES FROM THE ANALYSIS
# This will be useful for refining the information of the excluded cases and re-run the experiments

SAVE_PATH_REMOVED = "/local/data1/iulta54/Code/BTB_DEEP_LEARNING/dataset_csv_file"
removed = []

# %% REFINE TRUE FALSE
d = {
    "True": True,
    "False": False,
    "UNMATCHED_WSI": "UNMATCHED_WSI",
    "TRUE": True,
    "FALSE": False,
    True: True,
    False: False,
}
dataset_summary["USE_DURING_ANALYSIS"] = dataset_summary["USE_DURING_ANALYSIS"].map(d)
dataset_summary["ACCEPTABLE_IMAGE_QUALITY"] = dataset_summary[
    "ACCEPTABLE_IMAGE_QUALITY"
].map(d)

# %% REMOVE SUBJECTS THAT SHOULD NOT BE THERE
"""
BTB2024_5e4761c2_0a51bcc55ec3_c6636fd9_917f8a98_2ee04c52: case number 163
BTB2024_5e4761c2_44972dc52dd6_c6636fd9_b0f1699f_2ee04c52: case number 193
"""
to_remove = (
    "BTB2024_5e4761c2_0a51bcc55ec3_c6636fd9_917f8a98_2ee04c52",
    "BTB2024_5e4761c2_44972dc52dd6_c6636fd9_b0f1699f_2ee04c52",
)
dataset_summary = dataset_summary.loc[~dataset_summary.ANONYMIZED_CODE.isin(to_remove)]

# %% REMOVE WSIs THAT ARE UNMATCHED (NO WSI files for these cases)
to_remove = ["UNMATCHED_WSI"]
not_for_analysis = dataset_summary.loc[
    dataset_summary.MATCH_CLINICAL_WSI_INFO.isin(to_remove)
]
dataset_summary = dataset_summary.loc[
    ~dataset_summary.MATCH_CLINICAL_WSI_INFO.isin(to_remove)
]

# save removed cases
# Add flag for why removed
not_for_analysis["WHY_REMOVED"] = "UNMATCHED_WSI"
removed.append(not_for_analysis)
# %% PRINT NBR UNIQUE SUBJECTS
print_dataset_counts(dataset_summary)

# %% PRINT MISSING DIAGNOSIS
not_for_analysis = dataset_summary.loc[dataset_summary.USE_DURING_ANALYSIS != True]
dataset_summary = dataset_summary.loc[dataset_summary.USE_DURING_ANALYSIS == True]

# there are cases where a unique subject-diagnosis pair appears multiple times, but it is excluded partially.
# Get those that still are present and count them. These should be excluded from the not_for_analysis counts
sdp_to_exclude = list(
    pd.unique(
        rearrange_dataset_summary(not_for_analysis).SUBJECT_DIAGNOSIS_ID_ANONYMIZED
    )
)
sdp_to_include = list(
    pd.unique(
        rearrange_dataset_summary(dataset_summary).SUBJECT_DIAGNOSIS_ID_ANONYMIZED
    )
)

# get anonymized ID for those cases
excluded_still_available = [s for s in sdp_to_include if s in sdp_to_exclude]
excluded_still_available = [
    ID
    for ID in list(not_for_analysis.ANONYMIZED_CODE)
    if any(map(ID.__contains__, excluded_still_available))
]
# remove from the not_for_analysis
not_for_analysis = not_for_analysis.loc[
    ~not_for_analysis.ANONYMIZED_CODE.isin(excluded_still_available)
]

print(
    f"\n\nRemoving {len(pd.unique(rearrange_dataset_summary(not_for_analysis).SUBJECT_ID_ANONYMIZED))} subjects [{len(rearrange_dataset_summary(not_for_analysis))} SDPs] given USE_DURING_ANALYSIS != True"
)
print("############ REMOVING ############\n")
print_dataset_counts(not_for_analysis)
print("\n############ REMAINING ############\n")
print_dataset_counts(dataset_summary)

# save removed cases
# Add flag for why removed
not_for_analysis["WHY_REMOVED"] = "MISSING_DIAGNOSIS"
removed.append(not_for_analysis)

# %% REMOVE WSIs THAT ARE UNMATCHED (NO WSI files for these cases)
to_remove = ["UNMATCHED_CLINICAL"]
not_for_analysis = dataset_summary.loc[
    dataset_summary.MATCH_CLINICAL_WSI_INFO.isin(to_remove)
]
dataset_summary = dataset_summary.loc[
    ~dataset_summary.MATCH_CLINICAL_WSI_INFO.isin(to_remove)
]

print(
    f"\n\nRemoving {len(pd.unique(rearrange_dataset_summary(not_for_analysis).SUBJECT_ID_ANONYMIZED))} subjects [{len(rearrange_dataset_summary(not_for_analysis))} SDPs] given UNMATCHED_CLINICAL != True"
)
print("############ REMOVING ############\n")
print_dataset_counts(not_for_analysis)
print("\n############ REMAINING ############\n")
print_dataset_counts(dataset_summary)

# save removed cases
# Add flag for why removed
not_for_analysis["WHY_REMOVED"] = "UNMATCHED_CLINICAL"
removed.append(not_for_analysis)

# %% PRINT QUALITY CHECK FAIL
not_for_analysis = dataset_summary.loc[dataset_summary.ACCEPTABLE_IMAGE_QUALITY != True]
dataset_summary = dataset_summary.loc[dataset_summary.ACCEPTABLE_IMAGE_QUALITY == True]

# there are cases where a unique subject-diagnosis pair appears multiple times, but it is excluded partially.
# Get those that still are present and count them. These should be excluded from the not_for_analysis counts
sdp_to_exclude = list(
    pd.unique(
        rearrange_dataset_summary(not_for_analysis).SUBJECT_DIAGNOSIS_ID_ANONYMIZED
    )
)
sdp_to_include = list(
    pd.unique(
        rearrange_dataset_summary(dataset_summary).SUBJECT_DIAGNOSIS_ID_ANONYMIZED
    )
)

# get anonymized ID for those cases
excluded_still_available = [s for s in sdp_to_include if s in sdp_to_exclude]
excluded_still_available = [
    ID
    for ID in list(not_for_analysis.ANONYMIZED_CODE)
    if any(map(ID.__contains__, excluded_still_available))
]
# remove from the not_for_analysis
not_for_analysis = not_for_analysis.loc[
    ~not_for_analysis.ANONYMIZED_CODE.isin(excluded_still_available)
]

print(
    f"\n\nRemoving {len(pd.unique(rearrange_dataset_summary(not_for_analysis).SUBJECT_ID_ANONYMIZED))} subjects [{len(rearrange_dataset_summary(not_for_analysis))} SDPs] given ACCEPTABLE_IMAGE_QUALITY != True"
)
print("############ REMOVING ############\n")
print_dataset_counts(not_for_analysis)
print("\n############ REMAINING ############n\n")
print_dataset_counts(dataset_summary)

# save removed cases
# Add flag for why removed
not_for_analysis["WHY_REMOVED"] = "QUALITY_CHECK_FAIL"
removed.append(not_for_analysis)

# %% PRINT THOSE THAT HAVE HAVE EXTRACTED FEATURES
PATH_TO_PATCHES = "/local/data2/iulta54/Data/BTB/histology_features/wsi_level_features/clam_features_mag_x20_size_224/vit_uni/h5_files"
PATH_TO_FEATURES = "/local/data2/iulta54/Data/BTB/histology_features/wsi_level_features/clam_features_mag_x20_size_224/vit_uni/pt_files"

patch_list = [i.split(".")[0] for i in os.listdir(PATH_TO_PATCHES)]
feature_file_list = [i.split(".")[0] for i in os.listdir(PATH_TO_FEATURES)]

ids_without_patches = dataset_summary.loc[
    ~dataset_summary.ANONYMIZED_CODE.isin(patch_list)
]
dataset_summary = dataset_summary.loc[dataset_summary.ANONYMIZED_CODE.isin(patch_list)]

# there are cases where a unique subject-diagnosis pair appears multiple times, but it is excluded partially.
# Get those that still are present and count them. These should be excluded from the not_for_analysis counts
sdp_to_exclude = list(
    pd.unique(
        rearrange_dataset_summary(ids_without_patches).SUBJECT_DIAGNOSIS_ID_ANONYMIZED
    )
)
sdp_to_include = list(
    pd.unique(
        rearrange_dataset_summary(dataset_summary).SUBJECT_DIAGNOSIS_ID_ANONYMIZED
    )
)

# get anonymized ID for those cases
excluded_still_available = [s for s in sdp_to_include if s in sdp_to_exclude]
excluded_still_available = [
    ID
    for ID in list(ids_without_patches.ANONYMIZED_CODE)
    if any(map(ID.__contains__, excluded_still_available))
]
# remove from the not_for_analysis
ids_without_patches = ids_without_patches.loc[
    ~ids_without_patches.ANONYMIZED_CODE.isin(excluded_still_available)
]

print(f"\n\nRemoving {len(ids_without_patches)} given no PATCHES were created")
print("############ REMOVING ############\n")
print_dataset_counts(ids_without_patches)
print("\n############ REMAINING ############n\n")
print_dataset_counts(dataset_summary)
print("\n\n")

# ids_without_features = dataset_summary.loc[~dataset_summary.ANONYMIZED_CODE.isin(feature_file_list)]
# dataset_summary = dataset_summary.loc[dataset_summary.ANONYMIZED_CODE.isin(feature_file_list)]
# print(f'Removing {len(ids_without_features)} given no FEATURES were created')
# print('\n############ REMOVING ############\n')
# print_dataset_counts(ids_without_features)
# print('\n############ REMAINING ############n\n')
# print_dataset_counts(dataset_summary)


# save removed cases
# Add flag for why removed
ids_without_patches["WHY_REMOVED"] = "NO_PATCHING"
removed.append(ids_without_patches)

# %% PRINT DIAGNOSIS COUNTS (at all the levels)
# subject-diagnosis pairs
get_subjects_from_anonymized_code = lambda x: "_".join(x.split("_")[2:4])
dataset_summary["SUBJECT_DIAGNOSIS_ID"] = dataset_summary.ANONYMIZED_CODE.apply(
    get_subjects_from_anonymized_code
)

# only subject (not considering multiple diagnosis)
get_subjects_from_anonymized_code = lambda x: "_".join(x.split("_")[2:3])
dataset_summary["SUBJECT_ID"] = dataset_summary.ANONYMIZED_CODE.apply(
    get_subjects_from_anonymized_code
)

# WSI/glass id
get_glass_id_from_anonymized_code = lambda x: x.split("_")[4]
dataset_summary["GLASS_ID_CLINICAL"] = dataset_summary.ANONYMIZED_CODE.apply(
    get_glass_id_from_anonymized_code
)

# build counts for each TUMOR cluster (category, family and type)
for_analysis = copy.deepcopy(dataset_summary)

# print summary for each of the classification granularities
for d in ("WHO_TUMOR_CATEGORY", "WHO_TUMOR_FAMILY", "WHO_TUMOR_TYPE"):
    # group df based on the granularity
    gb = for_analysis.groupby([d]).agg(
        {
            "SUBJECT_DIAGNOSIS_ID": lambda x: len(pd.unique(x)),
            "SUBJECT_ID": lambda x: len(pd.unique(x)),
            "GLASS_ID_CLINICAL": lambda x: len(x),
        }
    )
    # order based on the number of subject-diagnosis pair
    gb = gb.sort_values(by=["SUBJECT_DIAGNOSIS_ID"])

    # print stats
    # print stats for each
    for l in list(gb.index):
        subjects_diagnosis = gb.loc[l].SUBJECT_DIAGNOSIS_ID
        subjects = gb.loc[l].SUBJECT_ID
        glasses = gb.loc[l].GLASS_ID_CLINICAL
        print(
            f"    {l:70s}: {subjects_diagnosis:4d} SDPs [{subjects:4d} subjects], {glasses:4d} glasses"
        )

    # print total count of subjects_diagnosis
    print(f'    {"#"*5}')
    print(f"    Total subject diagnosis pairs: {gb.SUBJECT_DIAGNOSIS_ID.sum():4d}")
    print(f'    {"#"*5}')
    # reset counter
    spds_count = 0


# %% PRINT BY FILTERING BASED ON MIN NBR SUBJECTS
print("\n\n\n")
min_nbr_subjects = 10

excluded = {"WHO_TUMOR_CATEGORY": 0, "WHO_TUMOR_FAMILY": 0, "WHO_TUMOR_TYPE": 0}

spds_count = 0
glasses_count = 0
# print summary for each of the classification granularities
for d in ("WHO_TUMOR_CATEGORY", "WHO_TUMOR_FAMILY", "WHO_TUMOR_TYPE"):
    print(f"{d} (with nbr. subjects-diagnosis pairs >= {min_nbr_subjects})")
    # group df based on the granularity
    gb = for_analysis.groupby([d]).agg(
        {
            "SUBJECT_DIAGNOSIS_ID": lambda x: len(pd.unique(x)),
            "SUBJECT_ID": lambda x: len(pd.unique(x)),
            "GLASS_ID_CLINICAL": lambda x: len(x),
            "GENDER": lambda x: {
                "M": sum(x == "M"),
                "F": sum(x == "F"),
                "NA": sum(x == "NotAvailable"),
            },
        }
    )
    # order based on the number of subject-diagnosis pair
    gb = gb.sort_values(by=["SUBJECT_DIAGNOSIS_ID"])

    # print stats
    # print stats for each
    for l in list(gb.index):
        subjects_diagnosis = gb.loc[l].SUBJECT_DIAGNOSIS_ID

        # get genders (sdps level)
        gender_list = (
            for_analysis.loc[for_analysis[d] == l]
            .groupby(["SUBJECT_DIAGNOSIS_ID"])
            .agg({"GENDER": lambda x: pd.unique(x)[0]})
            .GENDER.values
        )
        sdp_male = sum(gender_list == "M")
        sdp_female = sum(gender_list == "F")
        sdp_NA = sum(gender_list == None)

        subjects = gb.loc[l].SUBJECT_ID
        glasses = gb.loc[l].GLASS_ID_CLINICAL
        if subjects_diagnosis >= min_nbr_subjects:
            print(
                f"    {l:70s}: {subjects_diagnosis:4d} SDPs [{subjects:4d} subjects, M: {sdp_male}, F: {sdp_female}, NA: {sdp_NA}], {glasses:4d} glasses"
            )
            # count spds_count
            spds_count += subjects_diagnosis
            glasses_count += glasses
        else:
            # aggregate information for those that were excluded
            excluded[d] += subjects_diagnosis

    # print total count of subjects_diagnosis
    print(f'    {"#"*5}')
    print(f"    Total subject diagnosis pairs: {spds_count} ({glasses_count})")
    print(f'    {"#"*5}')
    # reset counter
    spds_count = 0
    glasses_count = 0

for d, v in excluded.items():
    print(f"{d}: excluded {v} subjects-diagnosis pairs")

# %% PLOT SUNBURST OF TUMOR DIAGNOSIS
import plotly.express as px
import numpy as np

# built df
df = (
    for_analysis.groupby(["WHO_TUMOR_CATEGORY", "WHO_TUMOR_FAMILY", "WHO_TUMOR_TYPE"])
    .count()
    .reset_index(level=[0, 1, 2])
)
df = df.loc[df.GENDER >= 10]
df["FRACTION"] = df.apply(lambda x: f"{x.GENDER / df.GENDER.sum() * 100:0.2f}", axis=1)

fig = fig = px.sunburst(
    df,
    path=["WHO_TUMOR_CATEGORY", "WHO_TUMOR_FAMILY", "WHO_TUMOR_TYPE"],
    values="FRACTION",
)
fig.show()

# %% SAVE FILE TO BE USED FOR PRE_PROCESSING
dataset_summary.to_csv(
    os.path.join(
        SAVE_PATH_REMOVED,
        f'BTB_AGGREGATED_CLINICAL_AND_WSI_INFORMATION_KS_LK_GOT_UM_LUND_UPP_ANONYM_{datetime.now().strftime("%Y%m%d")}.csv',
    ),
    index=False,
)
# %% SAVE LIST OF EXCLUDED TO CSV
removed = pd.concat(removed)
removed = removed.reset_index().rename(columns={"index": "INDEX_FOR_UPDATE"})
removed.to_csv(
    os.path.join(
        SAVE_PATH_REMOVED,
        f'BTB_REMOVED_FROM_ANALYSIS_{datetime.now().strftime("%Y%m%d")}.csv',
    ),
    index=False,
)
