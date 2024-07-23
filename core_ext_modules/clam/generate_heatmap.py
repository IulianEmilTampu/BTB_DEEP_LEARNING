# %% IMPORTS

'''
Script that given a subject id and a model, loads the features for the subject WSIs and generates heatmaps for each of the slides.  
'''

# %%
from __future__ import print_function

import numpy as np

import argparse
import pathlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import pdb
import os
import pandas as pd
from utils.utils import *
from math import floor
from utils.eval_utils import initiate_model as initiate_model
from models.model_clam import CLAM_MB, CLAM_SB
from models.model_abmil import ABMIL
from models.resnet_custom import resnet50_baseline
from types import SimpleNamespace
from collections import namedtuple
import h5py
import yaml
from wsi_core.wsi_utils import get_closest_downsample_level
from wsi_core.batch_process_utils import initialize_df
from vis_utils.heatmap_utils import initialize_wsi, drawHeatmap, compute_from_patches
from wsi_core.wsi_utils import sample_rois
from utils.file_utils import save_hdf5

import hydra
import omegaconf
from omegaconf import DictConfig, open_dict

# %% UTILITIES

def infer_single_slide(model, features, label, reverse_label_dict, k=1):
	features = features.to(device)
	with torch.no_grad():
		if isinstance(model, (CLAM_SB, CLAM_MB, ABMIL)):
			model_results_dict = model(features)
			logits, Y_prob, Y_hat, A, _ = model(features)
			Y_hat = Y_hat.item()

			if isinstance(model, (CLAM_MB,)):
				A = A[Y_hat]

			A = A.view(-1, 1).cpu().numpy()

		else:
			raise NotImplementedError

		print('Y_hat: {}, Y: {}, Y_prob: {}'.format(reverse_label_dict[Y_hat], label, ["{:.4f}".format(p) for p in Y_prob.cpu().flatten()]))	
		
		probs, ids = torch.topk(Y_prob, k)
		probs = probs[-1].cpu().numpy()
		ids = ids[-1].cpu().numpy()
		preds_str = np.array([reverse_label_dict[idx] for idx in ids])

	return ids, preds_str, probs, A

def load_params(df_entry, params):
	for key in params.keys():
		if key in df_entry.index:
			dtype = type(params[key])
			val = df_entry[key] 
			val = dtype(val)
			if isinstance(val, str):
				if len(val) > 0:
					params[key] = val
			elif not np.isnan(val):
				params[key] = val
			else:
				pdb.set_trace()

	return params

# %% PATHS

PATH_TO_WSIs = '/run/media/iulta54/Expansion/Datasets/BTB/PER_SITE_WSIs'
PATH_TO_FEATURES = '/local/data2/iulta54/Data/BTB/histology_features/wsi_level_features/clam_features_mag_x20_size_224/vit_uni' # where the features for the subjects are going to be searched for.
PATH_TO_SEGMENTATION_SETTINGS = '/local/data2/iulta54/Data/BTB/patch_extraction_csv_files/refined_segmentation_settings.csv'
SAVE_PATH = '/local/data1/iulta54/Code/BTB_DEEP_LEARNING/outputs'
SAVE_PATH = pathlib.Path(SAVE_PATH, 'heatmap_generation_conch')
SAVE_PATH.mkdir(parents=True, exist_ok=True)

# model for evaluation
'''
UNI - ABMIL 
- tumor category: median at repetition 77
- tumor family: median at repetition 115
- tumor type: median at repetition 9

CONCH - ABMIL 
- tumor category: median at repetition 130
- tumor family: median at repetition 107
- tumor type: median at repetition 6

ResNET - CLAM 
- tumor category: median at repetition 149
- tumor family: median at repetition 32
- tumor type: median at repetition 44
'''
# # UNI
# MODEL_INFORMATION = [
# 	{
# 		'model_path': '/local/data1/iulta54/Code/BTB_DEEP_LEARNING/outputs/2024_07_07_classification_models_and_results/tumor_type_all_vit_uni_mag_20_ps_224_agg_abmil_none_lr_1E-04_sch_cosine_opt_adamW_bgl_ce_7E-01_cls_True_svm_8_t083322',
# 		'repetition': 9,
# 		'classification_type':'tumor_type',
# 		'model_description': 'vit_uni_abmil'
# 	},
# 	{
# 		'model_path': '/local/data1/iulta54/Code/BTB_DEEP_LEARNING/outputs/2024_07_07_classification_models_and_results/tumor_category_all_vit_uni_mag_20_ps_224_agg_abmil_none_lr_1E-04_sch_cosine_opt_adamW_bgl_ce_7E-01_cls_True_svm_8_t070930',
# 		'repetition': 77,
# 		'classification_type': 'tumor_category',
# 		'model_description': 'vit_uni_abmil'
# 	},
# 	{
# 		'model_path': '/local/data1/iulta54/Code/BTB_DEEP_LEARNING/outputs/2024_07_07_classification_models_and_results/tumor_family_all_vit_uni_mag_20_ps_224_agg_abmil_none_lr_1E-04_sch_cosine_opt_adamW_bgl_ce_7E-01_cls_True_svm_8_t073358',
# 		'repetition': 115,
# 		'classification_type':'tumor_family' ,
# 		'model_description': 'vit_uni_abmil'
# 	}
# ]

# CONCH
MODEL_INFORMATION = [
	{
		'model_path': '/local/data1/iulta54/Code/BTB_DEEP_LEARNING/outputs/2024_07_07_classification_models_and_results/tumor_type_all_vit_uni_mag_20_ps_224_agg_abmil_none_lr_1E-04_sch_cosine_opt_adamW_bgl_ce_7E-01_cls_True_svm_8_t083322',
		'repetition': 130,
		'classification_type':'tumor_type',
		'model_description': 'vit_conch_abmil'
	},
	{
		'model_path': '/local/data1/iulta54/Code/BTB_DEEP_LEARNING/outputs/2024_07_07_classification_models_and_results/tumor_category_all_vit_uni_mag_20_ps_224_agg_abmil_none_lr_1E-04_sch_cosine_opt_adamW_bgl_ce_7E-01_cls_True_svm_8_t070930',
		'repetition': 107,
		'classification_type': 'tumor_category',
		'model_description': 'vit_conch_abmil'
	},
	{
		'model_path': '/local/data1/iulta54/Code/BTB_DEEP_LEARNING/outputs/2024_07_07_classification_models_and_results/tumor_family_all_vit_uni_mag_20_ps_224_agg_abmil_none_lr_1E-04_sch_cosine_opt_adamW_bgl_ce_7E-01_cls_True_svm_8_t073358',
		'repetition': 6,
		'classification_type':'tumor_family' ,
		'model_description': 'vit_conch_abmil'
	}
]

# # ResNet50
# MODEL_INFORMATION = [
# 	{
# 		'model_path': '/local/data1/iulta54/Code/BTB_DEEP_LEARNING/outputs/2024_07_07_classification_models_and_results/tumor_type_all_vit_uni_mag_20_ps_224_agg_abmil_none_lr_1E-04_sch_cosine_opt_adamW_bgl_ce_7E-01_cls_True_svm_8_t083322',
# 		'repetition': 149,
# 		'classification_type':'tumor_type',
# 		'model_description': 'resnet50_clam'
# 	},
# 	{
# 		'model_path': '/local/data1/iulta54/Code/BTB_DEEP_LEARNING/outputs/2024_07_07_classification_models_and_results/tumor_category_all_vit_uni_mag_20_ps_224_agg_abmil_none_lr_1E-04_sch_cosine_opt_adamW_bgl_ce_7E-01_cls_True_svm_8_t070930',
# 		'repetition': 32,
# 		'classification_type': 'tumor_category',
# 		'model_description': 'resnet50_clam'
# 	},
# 	{
# 		'model_path': '/local/data1/iulta54/Code/BTB_DEEP_LEARNING/outputs/2024_07_07_classification_models_and_results/tumor_family_all_vit_uni_mag_20_ps_224_agg_abmil_none_lr_1E-04_sch_cosine_opt_adamW_bgl_ce_7E-01_cls_True_svm_8_t073358',
# 		'repetition': 44,
# 		'classification_type':'tumor_family' ,
# 		'model_description': 'resnet50_clam'
# 	}
# ]


# paths to the list of subjects to evaluate
SUBJECT_TO_EVALUATE = '/local/data2/iulta54/Data/BTB/experiments_csv_files/patient_level_features/splits_tumor_category_npb_patient_features/dataset_descriptor.csv'
subjects_to_evaluate = pd.read_csv(SUBJECT_TO_EVALUATE, encoding="ISO-8859-1")
subjects_to_evaluate = list(subjects_to_evaluate.slide_id)

# path to the dataset description .csv file
DATASET_CSV_PATH = '/local/data1/iulta54/Code/BTB_DEEP_LEARNING/dataset_csv_file/BTB_AGGREGATED_CLINICAL_AND_WSI_INFORMATION_KS_LK_GOT_UM_LUND_UPP_ANONYM_20240704.csv'
dataset_summary = pd.read_csv(DATASET_CSV_PATH, encoding="ISO-8859-1")

# load segmentation settings
segmentation_settings = pd.read_csv(PATH_TO_SEGMENTATION_SETTINGS, encoding="ISO-8859-1")
# define default segmentation parameters 
default_seg_params = {
    'seg_level': -1,
    'sthresh' : 9,
    'mthresh' : 7,
    'close' : 5,
    'use_otsu' : False,
    'keep_ids' : 'none',
    'exclude_ids' : 'none',
}

default_filter_params = {
	'a_t' : 100,
    'a_h' : 16,
    'max_n_holes' : 8,
}

# %% LOOP THROUGH ALL THE SUBJECTS
for subject_id in subjects_to_evaluate:
	# find all the slides for this subject ID
	subject_information = dataset_summary.loc[dataset_summary.ANONYMIZED_CODE.str.contains(subject_id)]

	subject_information['feature_path'] = pd.Series()
	subject_information['patch_path'] = pd.Series()
	subject_information['wsi_path'] = pd.Series()

	# find alsto the segmetnation settings 
	subject_segmentation_settigs = segmentation_settings.loc[segmentation_settings.anonymized_slide_id.str.contains(subject_id)]

	# check if the subject does have the features for each of the WSIs available
	for index, s in subject_information.iterrows():
		# check feature
		feature_file_path = os.path.join(PATH_TO_FEATURES, 'pt_files', s.ANONYMIZED_CODE+'.pt')
		if os.path.isfile(feature_file_path):
			subject_information.loc[index, 'feature_path'] = feature_file_path

		# check patch
		patch_file_path = os.path.join(PATH_TO_FEATURES, 'h5_files', s.ANONYMIZED_CODE+'.h5')
		if os.path.isfile(patch_file_path):
			subject_information.loc[index, 'patch_path'] = patch_file_path
		
		# check WSI
		wsi_file_path = os.path.join(PATH_TO_WSIs, s.WSI_FILE_PATH)
		if os.path.isfile(wsi_file_path):
			subject_information.loc[index, 'wsi_path'] = wsi_file_path

	# print findings
	print(f'Found {len(subject_information) - subject_information.feature_path.isna().sum()} feature files, {len(subject_information) - subject_information.patch_path.isna().sum()} patch files, and {len(subject_information) - subject_information.wsi_path.isna().sum()} wsi files.')
	print(f'Missing {subject_information.feature_path.isna().sum()} feature files, {subject_information.patch_path.isna().sum()} patch files and {subject_information.wsi_path.isna().sum()} wsi files.')

	# drop WSI with do not have features
	subject_information = subject_information.dropna(subset=['feature_path'])

	# % LOAD ALL THE FEATURES AND PUT INTO ONE VECTOR. KEEP TRACK OF THE DIFFERENT INDEXES SINCE NEEDED TO BUILD HEATHMAPS
	features = []
	per_wsi_feature_size = [0]
	for f in list(subject_information.feature_path):
		wsi_features = torch.load(f)
		features.append(wsi_features)
		per_wsi_feature_size.append(per_wsi_feature_size[-1] + wsi_features.shape[0])

	# concatenate features
	features = torch.vstack(features)

	# % LOAD MODEL AND INFARE
	for model_infromation in MODEL_INFORMATION:
		# check the given trained model path
		if not os.path.isdir(model_infromation['model_path']):
			raise ValueError(f'The path to the trained model folder does not exist. Give {model_infromation["model_path"]}.')
		else:
			# check that the .pt file of the trained model is available for the specified fold.
			if not os.path.isfile(pathlib.Path(model_infromation['model_path'], f's_{model_infromation["repetition"]}_checkpoint.pt')):
				raise ValueError(f'The .pt file of the model checkpoint for the specified fold could not be found.')
			else:
				model_checkpoint_path = pathlib.Path(model_infromation['model_path'], f's_{model_infromation["repetition"]}_checkpoint.pt')
			# check that the hydra configuration file is also available (needed to parse information about the model and the task that was trained for)
			if not os.path.isfile(pathlib.Path(model_infromation['model_path'], f'hydra_config.yaml')):
				raise ValueError(f'The hydra configuration file for the trained model could not be found. In future implementations, the model information can be given or parsed from the trained model name.')
			else:
				# load the training configuration file
				training_cfg = omegaconf.OmegaConf.load(pathlib.Path(model_infromation['model_path'], f'hydra_config.yaml'))

		print('\ninitializing classification model from checkpoint.')
		# build model initialization arguments using the trained model configuration
		model_args = omegaconf.DictConfig({
			'model_type' : training_cfg.model_type,
			'model_size' : training_cfg.model_size,
			'drop_out' : training_cfg.drop_out,
			'n_classes' : training_cfg.task.n_classes,
			'feature_encoding_size' : training_cfg.encoding_size,
		})

		print('\nckpt path: {}'.format(model_checkpoint_path))
		model =  initiate_model(model_args, model_checkpoint_path)

		# % INFER SUBJECT and GET ATTENTION SCORED 

		class_labels = list(training_cfg.task.label_dict.keys())
		class_encodings = list(training_cfg.task.label_dict.values())
		reverse_label_dict = {class_encodings[i]: class_labels[i] for i in range(len(class_labels))} 
		n_classes = len(class_labels)

		# get label for the subject
		str_label = pd.unique(subject_information[f'WHO_{training_cfg.task.description.upper()}'])[0]
		try:
			categorical_label = training_cfg.task.label_dict[str_label]
		except:
			print(f'Model not trained to classify this class!')
			categorical_label = 100

		# infer 
		Y_hats, Y_hats_str, Y_probs, A = infer_single_slide(model, features, str_label, reverse_label_dict, n_classes)

		# % BUILD HEATMAP FOR EACH SLIDE

		for idx, (wsi_id, patch_file, wsi_file) in enumerate(zip(list(subject_information.ANONYMIZED_CODE), list(subject_information.patch_path), list(subject_information.wsi_path))):
			# open .h5 file 
			file = h5py.File(patch_file, 'r')
			coord_dset = file['coords']
			# get the index of the attention scores to use
			scores = A[per_wsi_feature_size[idx]:per_wsi_feature_size[idx+1]]
			coords = coord_dset[:]
			file.close()

			# get segmentation settings for this slide 
			wsi_seg_params = subject_segmentation_settigs.loc[subject_segmentation_settigs.anonymized_slide_id == wsi_id]

			if len(wsi_seg_params) == 0:
				print('Segmentation parameters not found for this slide. Using default values.')
				seg_params = default_seg_params
				filter_params = default_filter_params
			else:
				seg_params = {
						'seg_level': wsi_seg_params.seg_level.values[0],
						'sthresh' : wsi_seg_params.sthresh.values[0],
						'mthresh' : wsi_seg_params.mthresh.values[0],
						'close' : wsi_seg_params.close.values[0],
						'use_otsu' : bool(wsi_seg_params.use_otsu.values[0]),
						'keep_ids' : wsi_seg_params.keep_ids.values[0],
						'exclude_ids' : wsi_seg_params.exclude_ids.values[0],
				}

				filter_params = {
					'a_t' : wsi_seg_params.a_t.values[0],
					'a_h' : wsi_seg_params.a_h.values[0],
					'max_n_holes' : wsi_seg_params.max_n_holes.values[0],
				}

				
			
			print('Initializing WSI object')
			mask_save_name = f'{wsi_id}_mask.tiff'
			wsi_object = initialize_wsi(wsi_file, seg_mask_path=os.path.join(SAVE_PATH, mask_save_name), seg_params=seg_params, filter_params=filter_params)
			print('Done!') 

			# fix other parametres
			heatmap_vis_args = {'convert_to_percentiles': True, 'vis_level': -2, 'blur': True, 'custom_downsample': 1}
			vis_patch_size = tuple((np.array(224) * np.array(wsi_object.level_downsamples[get_closest_downsample_level(wsi_object, 20)]) * 1).astype(int))
			

			# generate heat map
			heatmap_save_name = f'{wsi_id}_{model_infromation["model_description"]}_{model_infromation["classification_type"]}_gt_{categorical_label}_pred_{Y_hats[0]}.png'

			heatmap = drawHeatmap(scores, coords, wsi_file, wsi_object=wsi_object, cmap='jet', alpha=0.3, use_holes=True, binarize=False, blank_canvas=False,
							thresh=-1, patch_size = vis_patch_size, overlap=0.5, segment=False,**heatmap_vis_args)

			# heatmap_vis_args = {'convert_to_percentiles': True, 'vis_level': 1, 'blur': False, 'custom_downsample': 1}
			# drawHeatmap(scores, coords, slide_path, 
			# 							wsi_object=wsi_object,  
			# 							cmap=cfg.heatmap_arguments.cmap, 
			# 							alpha=cfg.heatmap_arguments.alpha, 
			# 							**heatmap_vis_args, 
			# 							binarize=cfg.heatmap_arguments.binarize, 
			# 							blank_canvas=cfg.heatmap_arguments.blank_canvas,
			# 							thresh=cfg.heatmap_arguments.binary_thresh,  
			# 							patch_size = vis_patch_size,
			# 							overlap=patch_args.overlap, 
			# 							top_left = top_left, 
			# 							bot_right = bot_right)

			heatmap.save(os.path.join(SAVE_PATH, heatmap_save_name))

# %%
