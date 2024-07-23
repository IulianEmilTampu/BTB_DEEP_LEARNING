# %% IMPORTS
import h5py
import torch
import os
import numpy as np
import pathlib
from tqdm import tqdm

# %% PATHS
PER_WSI_FEATURES = '/local/data2/iulta54/Data/BTB/histology_features/clam_features_mag_x20_size_224/vit_conch/pt_files'
PER_SUBJECT_FEATURES_OUTPUT_DIR = '/local/data2/iulta54/Data/BTB/histology_features/clam_per_subject_features_mag_x20_size_224/vit_conch/pt_files'

# make output dir if not existing
pathlib.Path(PER_SUBJECT_FEATURES_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
print(f'Saving per-subject features at: {PER_SUBJECT_FEATURES_OUTPUT_DIR}')

# %% DEFINE HOW TO GET THE SUBJECT DI FROM THE ANONYMIZED CODE
get_subject_id = lambda x : os.path.basename(x).split('_')[2]

# %% GET THE LIST OF WSI-LEVEL FEATURES and SUBJECT IDS
tensor_files = os.listdir(PER_WSI_FEATURES)
subjects_ids = [get_subject_id(i) for i in tensor_files]
subjects_ids = list(dict.fromkeys(subjects_ids))

# build a list of tuples where in [0] we have the subject ID and in [1] the list of tensor files to stack
subject_to_WSI = [(s, [f for f in tensor_files if get_subject_id(f)==s]) for s in subjects_ids]

print(f'Found {len(tensor_files)} WSI-level feature files belonging to {len(subjects_ids)} unique subjects.')
# %% STACK THE WSI_FEATURE FROM EACH PATIENT

for s in tqdm(subject_to_WSI):
    # load and stack all the tensor files
    tensors_to_stack = []
    for f in s[1]:
        tensor_path = os.path.join(PER_WSI_FEATURES, f)
        tensor = torch.load(tensor_path)
        tensors_to_stack.append(tensor)
    
    # stack tensors
    stacked_tensor = torch.cat(tensors_to_stack, dim=0)

    # build file name
    output_file_path = os.path.join(PER_SUBJECT_FEATURES_OUTPUT_DIR, '_'.join(s[1][0].split('_')[0:3])+'.pt')

    # save
    torch.save(stacked_tensor, output_file_path)

# %%