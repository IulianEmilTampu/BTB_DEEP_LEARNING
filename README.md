# BTB project 
## Check tissue segmentation and patch extraction
Here we are using the functionalities provided by **hs2p** (https://github.com/clemsgrs/hs2p). The first thing to do is to create the required .csv file formatted like below: 
| slide_id   | slide_path              |
| ---------- | ----------------------- |
| slide_id_1 | path_to_slide_id_1.ndpi |
| slide_id_2 | path_to_slide_id_2.ndpi |

This can be achieved by running the BTB_create_csv_for_hsp2.py script that uses the summary file (*overall_summary_data_with_wsi_paths_**date**.csv*). The selection of which files to include in the .csv file for hs2p can be set using the configuration .yaml file (config/BT_csv_2_hsp2.yaml).

Ones the .csv file is created, move to the hsp2 folder and run
```bash
python3 patch_extraction.py --config-name *name_of_the_configuration_file_to_use*
```
Given that not all the slides share the same tissue segmentation configuration, one can specify which configurations to use (available: default, missing_tissue (changed the threshold value and the kernel size), small_tissue_pieces, stripes. In general, those that can't be processed by the default configuration can be processed using any of the other configurations). 

One might want to not save teh patches at first, but run the script using hs2p.config.default.flags.patch=False to only save the tissue segmentation results. Thus, one can check first if the tissue segmentation is satisfactory and then run the same code with hs2p.config.default.flags.patch=True to save the patches.

**Modifications**
The hs2p patch extraction utility has been modified to save the extracted patches using the slide_id name provided in the .csv file (this was not the case). Additionally, resuming of an experiment was also modified (previously not working since the output_dir that was created never matched the old one given the time stamp).
Search for **# IET** in the patch_extraction.py and source/utils.py files.

## Create csv for tumor classification
To run tumor classification, both HIPT and CLAM require a .csv file(s) where the slide_id and the class are provided for each of the training folds. Below you can find the specification for the format of HIPT and CLAM.

To obtain the .csv files needed for training of HIPT or CLAM on the BTB dataset, set the values in the make_csv_for_training.yaml configuration file and then run the make_csv_for_training.py.

### HIPT
HIPT requires one file for training (train.csv) and one file for validation (tune.csv) for each of the folds to run. The files for each fold need to be saved in a separate folder, named fold_**nbr**. 
| slide_id [str] | label [int] |
| -------------- | ----------- |
| slide_id_1     | 0           |
| slide_id_2     | 1           |

### CLAM
CLAM requires three different files for each of the folds (called splits, starting from 0): split_**nbr**.csv, split\_**nbr**\_bool.csv, and split\_**nbr**\_descriptor.csv . In addition, CLAM also needs a description of the class for each of the files which is provided by the dataset_description.csv file.
The description of each of these is provided below:
##### split_**nbr**.csv
Contains the slide_id for the training, validation and test sets for a given split.
| train [str] | val [str]   | test [str]  |
| ----------- | ----------- | ----------- |
| slide_id_1  | slide_id_3  | slide_id_15 |
| slide_id_2  | slide_id_65 | slide_id_89 |
| slide_id_25 | slide_id_65 |             |
| slide_id_67 |             |             |

##### split_**nbr**_bool.csv
Contains all the slide_ids in the dataset along with a bool value specifying if that slide belongs to the train, val or test set.
| slide_id [str] | train [bool] | val [bool] | test [bool] |
| -------------- | ------------ | ---------- | ----------- |
| slide_id_1     | True         | False      | False       |
| slide_id_2     | True         | False      | False       |
| slide_id_3     | False        | True       | False       |
| slide_id_89    | False        | False      | True        |

##### split_**nbr**_descriptor.csv
Contains a summary of the number of cases for each class and for each set.
| class [str] | train [int] | val [int] | test [int] |
| ----------- | ----------- | --------- | ---------- |
| Class_1     | 100         | 20        | 30         |
| Class_2     | 60          | 10        | 15         |

##### dataset_description.csv
Provides the class information for each of the files.
| case_id [str] | slide_id [int] | label [str] |
| ------------- | -------------- | ----------- |
| case_1        | slide_id_89    | class_1     |
| case_2        | slide_id_63    | class_2     |


## HIPT
Paper [CVPR2022](https://openaccess.thecvf.com/content/CVPR2022/html/Chen_Scaling_Vision_Transformers_to_Gigapixel_Images_via_Hierarchical_Self-Supervised_Learning_CVPR_2022_paper.html)
Original implementation [GitHub](https://github.com/mahmoodlab/HIPT?tab=readme-ov-file)
Radboud re-implementation (the one used here) [GitHub](https://github.com/clemsgrs/hipt)

### Feature extraction
The feature extraction is performed on the 4096 x 4096 patches extracted using the method described in [patch extraction](#check-tissue-segmentation-and-patch-extraction). The pre-trained ViT models extracting features at patch-level (Vit$_{256-16}$ 16x16 on patches of 256x256) and region level (Vit$_{4096-256}$on 256x256 on patches of 4096X4096) are saved in the pre_trained_feature_extractor folder. This folder will be the storage for all the pre-trained model used in the project.
To run the feature extraction, configure the classification/hipt/config/feature_extraction/default.yaml file with the path to the extracted patches, and run
```bash
python3 feature_extraction.py --config-name default
```
This will create a folder in the specified output folder with the .py files for each of the slide_id whose patches are found. 


### Tumor classification

### Visualization

## CLAM
Paper [Nature Biomedical Engineering](https://doi.org/10.1038/s41551-020-00682-w)
Original implementation [GitHub](https://github.com/mahmoodlab/CLAM)
### Patch extraction
##### Modifications to the original implementation
1. The original code has been refactored to allow the use of .yaml configuration files.
2. A *.csv* file can be provided as source (previously only a folder) where the path to the different slides to process are specified by column *slide_path*. 
3. If a *.csv* file is provided with a *slide_id* column, the patched slides will be saved not using the original slide name, but the *slide_id*. This is convenient in the case of BTB since the original files are not renamed (no need for a second copy of the dataset). Ones the features are generated, the original files (with the original name) are not needed, while the features are anonymized.

Search for **# IET** to find where the code was modified.

To run the patch extraction (fast processing: fp), first set the configurations in the CLAM/config/path_extraction/default.yaml file, and run

```bash
python3 create_patches_fp.py --config-name default
```
This will create a folder in the save_dir (../save_dir/experiment_name/time_stamp) where the masks and the patches (.h5 files) are saved. 

### Feature extraction
##### Modifications to the original implementation
1. The original code has been refactored to allow the use of .yaml configuration files.
2. A *.csv* file can be provided as source (previously only a folder) where the path to the different slides to process are specified by column *slide_path*. 
3. Two ViT pre-trained models (vit_hipt and vit_uni) are now available along with the ResNet50 feature extractor. These can be selected using the *model_type* setting in the .yaml file.

Search for **# IET** to find where the code was modified.

To run the feature extraction (fast processing: fp), set the configurations in the CLAM/config/feature_extraction/default.yaml, then run
```bash
python3 extract_features_fp.py --config-name default
```
This will create a folder in the feat_dir (../feat_dir/experiment_name/time_stamp) where .h5 and .pt feature files are saved.

### Tumor classification
##### Modifications to the original implementation
1. The original code has been refactored to allow the use of .yaml configuration files.
2. Addition of the ABMIL (simple gated attention based pooling - CLAM without the clustering)
3. 

### Visualization
##### Modifications to the original implementation

