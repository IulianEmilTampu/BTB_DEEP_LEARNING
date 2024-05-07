# %% 
'''
Utility that given the summary file (overall_summary_data_with_wsi_paths_date.csv), creates a 
.csv file that is needed to run the hsp2 tissue segmentation and patching.
'''

import os
import hydra
import datetime
import pandas as pd

from pathlib import Path
from omegaconf import DictConfig

@hydra.main(
    version_base="1.2.0", config_path="../../configs/pre_processing", config_name="BTB_csv_2_hs2p"
)
def main(cfg: DictConfig):

    # load BT_csv file
    btb_csv = pd.read_csv(cfg.btb_csv_path, encoding="ISO-8859-1")
    # make sure we have bools in USE_DURING_ANALYSIS and ACCEPTABLE_IMAGE_QUALITY columns
    d = {'True': True, 'False': False, 'UNMATCHED_WSI': 'UNMATCHED_WSI', 1: True, 0: False}
    btb_csv['USE_DURING_ANALYSIS'] = btb_csv['USE_DURING_ANALYSIS'].map(d)
    d = {'TRUE': True, 'FALSE': False, 'UNMATCHED_WSI': 'UNMATCHED_WSI', 'UNMATCHED':'UNMATCHED', 1: True, 0: False}
    btb_csv['ACCEPTABLE_IMAGE_QUALITY'] = btb_csv['ACCEPTABLE_IMAGE_QUALITY'].map(d)

    # include only those that are acceptable for the analysis (USE_DURING_ANALYSIS==True & ACCEPTABLE_IMAGE_QUALITY==True)
    btb_csv = btb_csv.loc[(btb_csv.USE_DURING_ANALYSIS==True) & (btb_csv.ACCEPTABLE_IMAGE_QUALITY==True)]

    # filter based on the patch_extraction_configuration. This is useful to create .csv for those files that have problems
    if cfg.patch_extraction_config:
        # get all the settings
        print(f'Creating .csv including only files with {cfg.patch_extraction_config} patch extraction configuration.')
        btb_csv = btb_csv.loc[btb_csv.PATCH_EXTRACTION_SETTINGS.isin(cfg.patch_extraction_config)]
        
    # create a new dataframe with only the ANONYMIZED_CODE (which will be the slide_id) and the WSI_FILE_PATH
    hs2p_csv = btb_csv[['ANONYMIZED_CODE', 'WSI_FILE_PATH']]
    # rename to match the hs2p requirements
    hs2p_csv = hs2p_csv.rename(columns={'ANONYMIZED_CODE':'slide_id', 'WSI_FILE_PATH':'slide_path'})

    # build the fill path to the slides using the cfg.WSIs_dir
    hs2p_csv['slide_path'] = hs2p_csv.apply(lambda x : os.path.join(cfg.WSIs_dir, x.slide_path), axis=1)

    # check if files exist (if requested)
    if cfg.skip_not_available_WSI:
        # check if the slide_path points to an existing file.
        # Remove those that are not existing.
        hs2p_csv['is_file'] = hs2p_csv.apply(lambda x : os.path.isfile(x.slide_path), axis=1)
        
        # print findings
        print(f'Found {len(hs2p_csv.loc[hs2p_csv.is_file==True])} files out of {len(hs2p_csv)}.')
        print(f'Removing the not existing files from the list ({len(hs2p_csv) - len(hs2p_csv.loc[hs2p_csv.is_file==True])}).')

        # remove files not existing
        hs2p_csv = hs2p_csv.loc[hs2p_csv.is_file==True]
        hs2p_csv.drop(columns=['is_file'])
    
    # check WSI have been already pre-processed (if requested)
    if cfg.skip_already_processed_WSI:
        if os.path.isdir(cfg.path_to_processed_WSIs):
            # check which files have already been processed and exclude
            list_preprocessed_WSIs = os.listdir(cfg.path_to_processed_WSIs)
            # remove file extension
            list_preprocessed_WSIs = [Path(v).stem for v in list_preprocessed_WSIs]
            print(f'Found {len(list_preprocessed_WSIs)} pre-processed WSIs. Removing them from the .csv file.')
            
            # remove entries in the dataframe
            hs2p_csv = hs2p_csv.loc[~hs2p_csv.slide_id.isin(list_preprocessed_WSIs)]
        else:
            raise ValueError(f'WSIs already pre-processed should be removed, but the path to where these are located is not available. GIven {cfg.path_to_processed_WSIs}. Check inputs.')

    # save file

    # run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M")
    run_id = datetime.datetime.now().strftime("%Y-%m-%d")
    # make output directory
    output_dir = Path(cfg.output_dir, cfg.experiment_name, run_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    hs2p_csv.to_csv(os.path.join(output_dir, 'BTB_for_hs2p.csv'), index_label=False, index=False)

    print(f'{len(hs2p_csv)} WSIs flagged for processing.')
    print(f'BTB_hs2p.csv file saved at {output_dir}')

if __name__ == '__main__':
    main()