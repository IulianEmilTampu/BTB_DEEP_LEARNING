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

# %% PATH TO THE CONFIGURATION FILE
CFG_PATH = '/local/data1/iulta54/Code/BTB_DEEP_LEARNING/core_ext_modules/clam/config/heatmap/default.yaml'
cfg = omegaconf.OmegaConf.load(CFG_PATH)

# %% BUILD MAIN
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

# check what is the process_list (.csv or directory)
if cfg.process_list:
	# check if file 
	if os.path.isfile(cfg.process_list):
		print('Running heatmap generation using information from the .csv process_list file')
	elif os.path.isdir(cfg.process_list):
		print('Running heatmap generation on the WSIs available in the process_list (patching and feature extraction included).')
	else:
		ValueError(f'The given process_list file does not exist. Given {cfg.process_list}.')
	
else:
	raise ValueError(f'The given process_list file does not exist. Given {cfg.process_list}.')

# check the given trained model path
if not os.path.isdir(cfg.trained_model):
	raise ValueError(f'The path to the trained model folder does not exist. Give {cfg.trained_model}.')
else:
	# check that the .pt file of the trained model is available for the specified fold.
	if not os.path.isfile(pathlib.Path(cfg.trained_model, f's_{cfg.fold}_checkpoint.pt')):
		raise ValueError(f'The .pt file of the model checkpoint for the specified fold could not be found.')
	else:
		model_checkpoint_path = pathlib.Path(cfg.trained_model, f's_{cfg.fold}_checkpoint.pt')
	# check that the hydra configuration file is also available (needed to parse information about the model and the task that was trained for)
	if not os.path.isfile(pathlib.Path(cfg.trained_model, f'hydra_config.yaml')):
		raise ValueError(f'The hydra configuration file for the trained model could not be found. In future implementations, the model information can be given or parsed from the trained model name.')
	else:
		# load the training configuration file
		training_cfg = omegaconf.OmegaConf.load(pathlib.Path(cfg.trained_model, f'hydra_config.yaml'))

# try to load the trained models dataset_descriptor.csv file to get information about the labels. 
# If not successful and the process_list (if a csv file) does not have a label column, the labels for each slide will the set to Unspecified.

if os.path.isfile(training_cfg.task.csv_path):
	training_dataset_descriptor = pd.read_csv(training_cfg.task.csv_path)
else:
	training_dataset_descriptor = None
	print('Dataset descriptor for the trained model could not be loaded.')

# check if a patch folder is provided
run_patching = False
if not cfg.h5_folder:
	print('A patch directory (with .h5 files for each slide_id in the process_list file) was not provided.\nAll slides will be patched and feature extracted.\nFeatures and patches will be saved in the corresponding output directory.')
	run_patching = True
elif not os.path.isdir(cfg.h5_folder):
	raise ValueError(f'The given patch folder is not valid. Given {cfg.h5_folder}')

run_feature_extraction = False	
if not cfg.feature_folder:
	print('A feature directory (with .pt files for each slide_id in the process_list file) was not provided.\nFeature extraction will be performed and the features will be saved in the corresponding output directory.')
	run_feature_extraction = True	
elif not os.path.isdir(cfg.feature_folder):
	raise ValueError(f'The given feature folder is not valid. Given {cfg.h5_folder}')	

if not run_patching and not run_feature_extraction:
	print('Both patch coordinates and features are provided.\nEnsure that the patches and the features are aligned.')

# %% PARSE PARAMETERS AND BUILD DEFAULTS (TODO check that the parameters match the trained model configuration)

patch_args = cfg.patching_arguments

# data_args = cfg.data_arguments
# model_args = cfg.model_arguments

# exp_args = cfg.exp_arguments
# heatmap_args = cfg.heatmap_arguments
# sample_args = cfg.sample_arguments

patch_size = tuple([patch_args.patch_size, patch_args.patch_size])
step_size = tuple((np.array(patch_size) * (1-patch_args.overlap)).astype(int))
print(f'patch_size: {patch_size}, with {patch_args.overlap:.2f} overlap -> step size is {step_size}')

# initialise list of files to process along with the settings

if os.path.isdir(cfg.process_list):
	if isinstance(cfg.process_list, list):
		slides = []
		for data_dir in cfg.process_list:
			slides.extend(os.listdir(data_dir))
	else:
		slides = sorted(os.listdir(cfg.process_list))
	slides = [slide for slide in slides if cfg.slide_ext in slide]
	df = initialize_df(slides, cfg.patching_arguments.seg_params, cfg.patching_arguments.filter_params, cfg.patching_arguments.vis_params, cfg.patching_arguments.patch_params, use_heatmap_args=False)
	
else:
	df = pd.read_csv(cfg.process_list)
	df = initialize_df(df, cfg.patching_arguments.seg_params, cfg.patching_arguments.filter_params, cfg.patching_arguments.vis_params, cfg.patching_arguments.patch_params, use_heatmap_args=False, use_anonymized_slide_ids=True, anonymized_slide_ids=list(df.slide_id))


mask = df['process'] == 1
process_stack = df[mask].reset_index(drop=True)
total = len(process_stack)
print(f'\nFound {len(process_stack)} slides to process.')

# %% INITIALIZE CLASSIFICATION MODEL
print('\ninitializing classification model from checkpoint.')
# build model initialization arguments using the trained model configuration
model_args = omegaconf.DictConfig({
	'model_type' : training_cfg.model_type,
	'model_size' : training_cfg.model_size,
	'drop_out' : training_cfg.drop_out,
	'n_classes' : training_cfg.task.n_classes,
})

print('\nckpt path: {}'.format(model_checkpoint_path))
model =  initiate_model(model_args, model_checkpoint_path)

# %% INITIALIZE FEATURE EXTRACTOR MODEL

print('\nInitializing feature extraction model from checkpoint.')
if training_cfg.feature_extractor == 'resnet50':
	print('ResNet50')
	feature_extractor = resnet50_baseline(pretrained=True)
	feature_extractor = feature_extractor.to(device)
elif training_cfg.feature_extractor == 'vit_hipt':
	print('ViT HIPT')
	# import HIPT ViT pre-trained model
	feature_extractor = get_vit256(pretrained_weights=os.path.join(training_cfg.pre_trained_model_archive, 'vit256_small_dino.pth')).to(device)
elif training_cfg.feature_extractor == 'vit_uni':
	print('ViT UNI')
	# import UNI ViT pre-trained model
	import timm
	from torchvision import transforms
	
	feature_extractor = timm.create_model(
	"vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
		)
	feature_extractor.load_state_dict(torch.load(os.path.join(cfg.feature_extraction_arguments.pre_trained_model_archive, "vit224_large_uni_dino.bin"), map_location=device), strict=True)
	feature_extractor = feature_extractor.to(device)
elif training_cfg.feature_extractor == 'vit_conch':
	print('ViT CONCH')
	# import CONCH ViT pre-trained model (using the conch factory.py utility)
	from models.conch_open_clip_custom import create_model_from_pretrained
	feature_extractor, _ = create_model_from_pretrained(model_cfg='conch_ViT-B-16', checkpoint_path=os.path.join(cfg.feature_extraction_arguments.pre_trained_model_archive, "vit224_large_conch.bin"), device=device)
	feature_extractor.forward = partial(feature_extractor.encode_image, proj_contrast=False, normalize=False)
	feature_extractor = feature_extractor.to(device)

if torch.cuda.device_count() > 1:
	device_ids = list(range(torch.cuda.device_count()))
	feature_extractor = nn.DataParallel(feature_extractor, device_ids=device_ids).to('cuda:0')
else:
	feature_extractor = feature_extractor.to(device)
print('Done!')

# %% GET AND FIX LABELS
label_dict =  training_cfg.task.label_dict
class_labels = list(label_dict.keys())
class_encodings = list(label_dict.values())
reverse_label_dict = {class_encodings[i]: class_labels[i] for i in range(len(class_labels))} 
n_classes = len(class_labels)

# %% DEFINE SAVE PATHS
temp_output_folder = os.path.join(cfg.output_dir, 'heatmaps', os.path.basename(cfg.trained_model))

production_save_dir = pathlib.Path(temp_output_folder, 'production_save_dir')
production_save_dir.mkdir(parents=True, exist_ok=True)

raw_save_dir = pathlib.Path(temp_output_folder, 'raw_save_dir')
raw_save_dir.mkdir(parents=True, exist_ok=True)

# %% LOOP THROUGH THE SLIDES IN THE PROCESS STACK
for i in range(len(process_stack)):
	slide_name = process_stack.loc[i, 'slide_id']
	if cfg.slide_ext not in slide_name:
		slide_name+=cfg.slide_ext
	print('\nprocessing: ', slide_name)	

	slide_id = slide_name.replace(cfg.slide_ext, '')

	# try to infire the label
	try:
		label = process_stack.loc[i, 'label']
	except KeyError:
		if training_dataset_descriptor is not None:
			# try to get the label for this slide id using the training dataset descriptor. 
			try:
				label = training_dataset_descriptor.loc[training_dataset_descriptor.slide_id==slide_id].label.values[0]
			except IndexError:
				label = 'Unspecified' 

	if not isinstance(label, str):
		grouping = reverse_label_dict[label]
	else:
		grouping = label

	p_slide_save_dir = pathlib.Path(production_save_dir, str(grouping), slide_id)
	p_slide_save_dir.mkdir(parents=True, exist_ok=True)

	r_slide_save_dir = pathlib.Path(raw_save_dir, str(grouping), slide_id)
	r_slide_save_dir.mkdir(parents=True, exist_ok=True)

	#  ###

	if cfg.heatmap_arguments.use_roi:
			x1, x2 = process_stack.loc[i, 'x1'], process_stack.loc[i, 'x2']
			y1, y2 = process_stack.loc[i, 'y1'], process_stack.loc[i, 'y2']
			top_left = (int(x1), int(y1))
			bot_right = (int(x2), int(y2))
	else:
		top_left = None
		bot_right = None
		
	print('slide id: ', slide_id)
	print('top left: ', top_left, ' bot right: ', bot_right)

	if os.path.isdir(cfg.process_list):
		slide_path = os.path.join(cfg.process_list, slide_name)
	else:
		slide_path = process_stack.loc[i, 'slide_path']

	# ### build mask file path (to be moved)
	mask_file = os.path.join(r_slide_save_dir, slide_id+'_mask.pkl')

	# ###
	seg_params = cfg.patching_arguments.seg_params.copy()
	filter_params = cfg.patching_arguments.filter_params.copy()
	vis_params = cfg.patching_arguments.vis_params.copy()

	seg_params = load_params(process_stack.loc[i], seg_params)
	filter_params = load_params(process_stack.loc[i], filter_params)
	vis_params = load_params(process_stack.loc[i], vis_params)

	keep_ids = str(seg_params['keep_ids'])
	if len(keep_ids) > 0 and keep_ids != 'none':
		seg_params['keep_ids'] = np.array(keep_ids.split(',')).astype(int)
	else:
		seg_params['keep_ids'] = []

	exclude_ids = str(seg_params['exclude_ids'])
	if len(exclude_ids) > 0 and exclude_ids != 'none':
		seg_params['exclude_ids'] = np.array(exclude_ids.split(',')).astype(int)
	else:
		seg_params['exclude_ids'] = []

	if cfg.debug:
		for key, val in seg_params.items():
			print('{}: {}'.format(key, val))

		for key, val in filter_params.items():
			print('{}: {}'.format(key, val))

		for key, val in vis_params.items():
			print('{}: {}'.format(key, val))
	
	print('Initializing WSI object')
	wsi_object = initialize_wsi(slide_path, seg_mask_path=mask_file, seg_params=seg_params, filter_params=filter_params)
	print('Done!') 
	
	index_level_downsample = get_closest_downsample_level(wsi_object, training_cfg.magnification)
	wsi_ref_downsample = wsi_object.level_downsamples[index_level_downsample]

	# the actual patch size for heatmap visualization should be the patch size * downsample factor * custom downsample factor
	vis_patch_size = tuple((np.array(patch_size) * np.array(wsi_ref_downsample) * patch_args.custom_downsample).astype(int))

	block_map_save_path = os.path.join(r_slide_save_dir, '{}_blockmap.h5'.format(slide_id))
	mask_path = os.path.join(r_slide_save_dir, '{}_mask.jpg'.format(slide_id))
	if cfg.patching_arguments.vis_params['vis_level'] < 0:
		best_level = wsi_object.wsi.get_best_level_for_downsample(32)
		cfg.patching_arguments.vis_params['vis_level'] = best_level
	mask = wsi_object.visWSI(**vis_params, number_contours=True)
	mask.save(mask_path)
	
	features_path = os.path.join(r_slide_save_dir, slide_id+'.pt')
	h5_path = os.path.join(r_slide_save_dir, slide_id+'.h5')

	# some setting
	blocky_wsi_kwargs = {'top_left': None, 'bot_right': None, 'patch_size': patch_size, 'step_size': patch_size, 
'custom_downsample':cfg.patching_arguments.custom_downsample, 'level': index_level_downsample, 'use_center_shift': cfg.heatmap_arguments.use_center_shift}

	break
# %% PERFORM FEATURE EXTRACTION IF FEATURES NOT AVAILABLE FOR THIS SLIDE.
	##### check if h5_features_file exists ######
	if not (cfg.h5_folder and os.path.isfile(pathlib.Path(cfg.h5_folder, slide_id+'.h5'))):
		print(f'Patching...', end='')
		h5_path = os.path.join(r_slide_save_dir, slide_id+'.h5')
		_, _, wsi_object = compute_from_patches(wsi_object=wsi_object, 
										model=model, 
										feature_extractor=feature_extractor, 
										batch_size=512, **blocky_wsi_kwargs, 
										attn_save_path=None, feat_save_path=h5_path, 
										ref_scores=None)
# %% PREDICT THIS SLIDE
	##### check if pt_features_file exists ######
	if not os.path.isfile(features_path):
		file = h5py.File(h5_path, "r")
		features = torch.tensor(file['features'][:])
		torch.save(features, features_path)
		file.close()

	# load features 
	features = torch.load(features_path)
	process_stack.loc[i, 'bag_size'] = len(features)
	
	wsi_object.saveSegmentation(mask_file)
	Y_hats, Y_hats_str, Y_probs, A = infer_single_slide(model, features, label, reverse_label_dict, n_classes)
	del features

# %% GET THE TOP THREE PREDICTIONS
	if not os.path.isfile(block_map_save_path): 
				file = h5py.File(h5_path, "r")
				coords = file['coords'][:]
				file.close()
				asset_dict = {'attention_scores': A, 'coords': coords}
				block_map_save_path = save_hdf5(block_map_save_path, asset_dict, mode='w')
		
	# save top 3 predictions
	for c in range(n_classes):
		process_stack.loc[i, 'Pred_{}'.format(c)] = Y_hats_str[c]
		process_stack.loc[i, 'p_{}'.format(c)] = Y_probs[c]

# %%  SAVE PROCESS STACK 
		process_stack.to_csv(pathlib.Path(temp_output_folder, 'process_stack.csv'), index=False)

# %%    LOAD ATTENTION SCORES FOR EACH PATCH AND CORRESPONDING COORDINATES
		file = h5py.File(block_map_save_path, 'r')
		dset = file['attention_scores']
		coord_dset = file['coords']
		scores = dset[:]
		coords = coord_dset[:]
		file.close()

		# save cf.sample_arguments
		sample_arguments = cfg.representative_patches_arguments
		if sample_arguments.save_samples:
			tag = "label_{}_pred_{}".format(label, Y_hats[0])
			sample_save_dir =  os.path.join(p_slide_save_dir, 'sampled_patches', str(tag), sample_arguments['name'])
			os.makedirs(sample_save_dir, exist_ok=True)
			print('sampling {}'.format(sample_arguments['name']))
			sample_results = sample_rois(scores, coords, k=sample_arguments['k'], mode=sample_arguments['mode'], seed=sample_arguments['seed'], 
				score_start=0.45, score_end=0.55)
			for idx, (s_coord, s_score) in enumerate(zip(sample_results['sampled_coords'], sample_results['sampled_scores'])):
				print('coord: {} score: {:.3f}'.format(s_coord, s_score))
				patch = wsi_object.wsi.read_region(tuple(s_coord), index_level_downsample, (patch_size[0], patch_size[1])).convert('RGB')
				patch.save(os.path.join(sample_save_dir, '{}_{}_x_{}_y_{}_a_{:.3f}.png'.format(idx, slide_id, s_coord[0], s_coord[1], s_score)))

# %% SAVE RAW UN-SMOOTHED ATTENTION MAP

		wsi_kwargs = {'top_left': top_left, 'bot_right': bot_right, 'patch_size': patch_size, 'step_size': step_size, 
		'custom_downsample':cfg.patching_arguments.custom_downsample, 'level': index_level_downsample, 'use_center_shift': cfg.heatmap_arguments.use_center_shift}

		heatmap_save_name = '{}_blockmap.tiff'.format(slide_id)
		if os.path.isfile(os.path.join(r_slide_save_dir, heatmap_save_name)):
			pass
		else:
			heatmap = drawHeatmap(scores, coords, slide_path, wsi_object=wsi_object, cmap=cfg.heatmap_arguments.cmap, alpha=cfg.heatmap_arguments.alpha, use_holes=True, binarize=False, vis_level=-1, blank_canvas=False,
							thresh=-1, patch_size = vis_patch_size, convert_to_percentiles=True)
		
			heatmap.save(os.path.join(r_slide_save_dir, '{}_blockmap.png'.format(slide_id)))
			del heatmap

# %% COMPUTE HEATMAP ON OVERLAPPING PATCHES - THIS HELPS IN MAKING THE ATTENTION MAP SMOOTHER.
	save_path = os.path.join(r_slide_save_dir, '{}_{}_roi_{}.h5'.format(slide_id, cfg.patching_arguments.overlap, cfg.heatmap_arguments.use_roi))

	if cfg.heatmap_arguments.use_ref_scores:
		ref_scores = scores
	else:
		ref_scores = None

	ref_scores = None
	if cfg.heatmap_arguments.calc_heatmap:
		compute_from_patches(wsi_object=wsi_object, clam_pred=Y_hats[0], model=model, feature_extractor=feature_extractor, batch_size=512, **wsi_kwargs, 
							attn_save_path=save_path,  ref_scores=ref_scores)

# %% SAVE RAW SMOOTHED ATTENTION MAP
	file = h5py.File(save_path, 'r')
	dset = file['attention_scores']
	coord_dset = file['coords']
	scores = dset[:]
	coords = coord_dset[:]
	file.close()

	cfg.heatmap_arguments.blur = True
	heatmap_vis_args = {'convert_to_percentiles': True, 'vis_level': cfg.heatmap_arguments.vis_level, 'blur': cfg.heatmap_arguments.blur, 'custom_downsample': cfg.heatmap_arguments.custom_downsample}
	if cfg.heatmap_arguments.use_ref_scores:
		heatmap_vis_args['convert_to_percentiles'] = False

	heatmap_save_name = '{}_{}_roi_{}_blur_{}_rs_{}_bc_{}_a_{}_l_{}_bi_{}_{}.{}'.format(slide_id, float(patch_args.overlap), int(cfg.heatmap_arguments.use_roi),
																					int(cfg.heatmap_arguments.blur), 
																					int(cfg.heatmap_arguments.use_ref_scores), int(cfg.heatmap_arguments.blank_canvas), 
																					float(cfg.heatmap_arguments.alpha), int(cfg.heatmap_arguments.vis_level), 
																					int(cfg.heatmap_arguments.binarize), float(cfg.heatmap_arguments.binary_thresh), cfg.heatmap_arguments.save_ext)

	# if os.path.isfile(os.path.join(p_slide_save_dir, heatmap_save_name)):
	if not True:
			pass
	else:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
		heatmap = drawHeatmap(scores, coords, slide_path, 
								wsi_object=wsi_object,  
								cmap=cfg.heatmap_arguments.cmap, 
								alpha=cfg.heatmap_arguments.alpha, 
								**heatmap_vis_args, 
								binarize=cfg.heatmap_arguments.binarize, 
								blank_canvas=cfg.heatmap_arguments.blank_canvas,
								thresh=cfg.heatmap_arguments.binary_thresh,  
								patch_size = vis_patch_size,
								overlap=patch_args.overlap, 
								top_left = top_left, 
								bot_right = bot_right)
		if cfg.heatmap_arguments.save_ext == 'jpg':
			heatmap.save(os.path.join(p_slide_save_dir, heatmap_save_name), quality=20)
		else:
			heatmap.save(os.path.join(p_slide_save_dir, heatmap_save_name))
# %% 
	if not os.path.isfile(save_path):
		print('heatmap {} not found'.format(save_path))
		if heatmap_args.use_roi:
			save_path_full = os.path.join(r_slide_save_dir, '{}_{}_roi_False.h5'.format(slide_id, patch_args.overlap))
			print('found heatmap for whole slide')
			save_path = save_path_full
		# else:
		# 	continue

	file = h5py.File(save_path, 'r')
	dset = file['attention_scores']
	coord_dset = file['coords']
	scores = dset[:]
	coords = coord_dset[:]
	file.close()
# %% 
# %%
# # %% MAIN
# @hydra.main(
#     version_base="1.2.0", config_path=os.path.join('config', 'heatmaps'), config_name='default'
# )

# def main(cfg:DictConfig):

# 	for key, value in config_dict.items():
# 		if isinstance(value, dict):
# 			print('\n'+key)
# 			for value_key, value_value in value.items():
# 				print (value_key + " : " + str(value_value))
# 		else:
# 			print ('\n'+key + " : " + str(value))
			
# 	decision = input('Continue? Y/N ')
# 	if decision in ['Y', 'y', 'Yes', 'yes']:
# 		pass
# 	elif decision in ['N', 'n', 'No', 'NO']:
# 		exit()
# 	else:
# 		raise NotImplementedError

# 	# patch_args = argparse.Namespace(**args['patching_arguments'])
# 	# data_args = argparse.Namespace(**args['data_arguments'])
# 	# model_args = args['model_arguments']
# 	# model_args.update({'n_classes': args['exp_arguments']['n_classes']})
# 	# model_args = argparse.Namespace(**model_args)
# 	# exp_args = argparse.Namespace(**args['exp_arguments'])
# 	# heatmap_args = argparse.Namespace(**args['heatmap_arguments'])
# 	# sample_args = argparse.Namespace(**args['sample_arguments'])

# 	patch_args = cfg.patching_arguments
# 	data_args = cfg.data_arguments
# 	model_args = cfg.model_arguments
# 	exp_args = cfg.exp_arguments
# 	heatmap_args = cfg.heatmap_arguments
# 	sample_args = cfg.sample_arguments

# 	patch_size = tuple([patch_args.patch_size for i in range(2)])
# 	step_size = tuple((np.array(patch_size) * (1-patch_args.overlap)).astype(int))
# 	print('patch_size: {} x {}, with {:.2f} overlap, step size is {} x {}'.format(patch_size[0], patch_size[1], patch_args.overlap, step_size[0], step_size[1]))

# 	preset = data_args.preset
# 	def_seg_params = {'seg_level': -1, 'sthresh': 15, 'mthresh': 11, 'close': 2, 'use_otsu': False, 
# 					  'keep_ids': 'none', 'exclude_ids':'none'}
# 	def_filter_params = {'a_t':50.0, 'a_h': 8.0, 'max_n_holes':10}
# 	def_vis_params = {'vis_level': -1, 'line_thickness': 250}
# 	def_patch_params = {'use_padding': True, 'contour_fn': 'four_pt'}

# 	if preset is not None:
# 		preset_df = pd.read_csv(preset)
# 		for key in def_seg_params.keys():
# 			def_seg_params[key] = preset_df.loc[0, key]

# 		for key in def_filter_params.keys():
# 			def_filter_params[key] = preset_df.loc[0, key]

# 		for key in def_vis_params.keys():
# 			def_vis_params[key] = preset_df.loc[0, key]

# 		for key in def_patch_params.keys():
# 			def_patch_params[key] = preset_df.loc[0, key]


# 	if data_args.process_list is None:
# 		if isinstance(data_args.data_dir, list):
# 			slides = []
# 			for data_dir in data_args.data_dir:
# 				slides.extend(os.listdir(data_dir))
# 		else:
# 			slides = sorted(os.listdir(data_args.data_dir))
# 		slides = [slide for slide in slides if data_args.slide_ext in slide]
# 		df = initialize_df(slides, def_seg_params, def_filter_params, def_vis_params, def_patch_params, use_heatmap_args=False)
		
# 	else:
# 		df = pd.read_csv(data_args.process_list)
# 		df = initialize_df(df, def_seg_params, def_filter_params, def_vis_params, def_patch_params, use_heatmap_args=False)

	# mask = df['process'] == 1
	# process_stack = df[mask].reset_index(drop=True)
	# total = len(process_stack)
	# print('\nlist of slides to process: ')
	# print(process_stack.head(len(process_stack)))

	# print('\ninitializing model from checkpoint')
	# ckpt_path = model_args.ckpt_path
	# print('\nckpt path: {}'.format(ckpt_path))
	
	# if model_args.initiate_fn == 'initiate_model':
	# 	model =  initiate_model(model_args, ckpt_path)
	# else:
	# 	raise NotImplementedError


	# feature_extractor = resnet50_baseline(pretrained=True)
	# feature_extractor.eval()
	# device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
	# print('Done!')

	# label_dict =  training_cfg.task.label_dict
	# class_labels = list(label_dict.keys())
	# class_encodings = list(label_dict.values())
	# reverse_label_dict = {class_encodings[i]: class_labels[i] for i in range(len(class_labels))} 

	# if torch.cuda.device_count() > 1:
	# 	device_ids = list(range(torch.cuda.device_count()))
	# 	feature_extractor = nn.DataParallel(feature_extractor, device_ids=device_ids).to('cuda:0')
	# else:
	# 	feature_extractor = feature_extractor.to(device)

	# os.makedirs(exp_args.production_save_dir, exist_ok=True)
	# os.makedirs(exp_args.raw_save_dir, exist_ok=True)
	# blocky_wsi_kwargs = {'top_left': None, 'bot_right': None, 'patch_size': patch_size, 'step_size': patch_size, 
	# 'custom_downsample':patch_args.custom_downsample, 'level': patch_args.patch_level, 'use_center_shift': heatmap_args.use_center_shift}

	# for i in range(len(process_stack)):
	# 	slide_name = process_stack.loc[i, 'slide_id']
	# 	if data_args.slide_ext not in slide_name:
	# 		slide_name+=data_args.slide_ext
	# 	print('\nprocessing: ', slide_name)	

	# 	try:
	# 		label = process_stack.loc[i, 'label']
	# 	except KeyError:
	# 		label = 'Unspecified'

	# 	slide_id = slide_name.replace(data_args.slide_ext, '')

	# 	if not isinstance(label, str):
	# 		grouping = reverse_label_dict[label]
	# 	else:
	# 		grouping = label

	# 	p_slide_save_dir = os.path.join(exp_args.production_save_dir, exp_args.save_exp_code, str(grouping))
	# 	os.makedirs(p_slide_save_dir, exist_ok=True)

	# 	r_slide_save_dir = os.path.join(exp_args.raw_save_dir, exp_args.save_exp_code, str(grouping),  slide_id)
	# 	os.makedirs(r_slide_save_dir, exist_ok=True)

	# 	if heatmap_args.use_roi:
	# 		x1, x2 = process_stack.loc[i, 'x1'], process_stack.loc[i, 'x2']
	# 		y1, y2 = process_stack.loc[i, 'y1'], process_stack.loc[i, 'y2']
	# 		top_left = (int(x1), int(y1))
	# 		bot_right = (int(x2), int(y2))
	# 	else:
	# 		top_left = None
	# 		bot_right = None
		
	# 	print('slide id: ', slide_id)
	# 	print('top left: ', top_left, ' bot right: ', bot_right)

	# 	if isinstance(data_args.data_dir, str):
	# 		slide_path = os.path.join(data_args.data_dir, slide_name)
	# 	elif isinstance(data_args.data_dir, dict):
	# 		data_dir_key = process_stack.loc[i, data_args.data_dir_key]
	# 		slide_path = os.path.join(data_args.data_dir[data_dir_key], slide_name)
	# 	else:
	# 		raise NotImplementedError

	# 	mask_file = os.path.join(r_slide_save_dir, slide_id+'_mask.pkl')
		
		# # Load segmentation and filter parameters
		# seg_params = def_seg_params.copy()
		# filter_params = def_filter_params.copy()
		# vis_params = def_vis_params.copy()

		# seg_params = load_params(process_stack.loc[i], seg_params)
		# filter_params = load_params(process_stack.loc[i], filter_params)
		# vis_params = load_params(process_stack.loc[i], vis_params)

		# keep_ids = str(seg_params['keep_ids'])
		# if len(keep_ids) > 0 and keep_ids != 'none':
		# 	seg_params['keep_ids'] = np.array(keep_ids.split(',')).astype(int)
		# else:
		# 	seg_params['keep_ids'] = []

		# exclude_ids = str(seg_params['exclude_ids'])
		# if len(exclude_ids) > 0 and exclude_ids != 'none':
		# 	seg_params['exclude_ids'] = np.array(exclude_ids.split(',')).astype(int)
		# else:
		# 	seg_params['exclude_ids'] = []

		# for key, val in seg_params.items():
		# 	print('{}: {}'.format(key, val))

		# for key, val in filter_params.items():
		# 	print('{}: {}'.format(key, val))

		# for key, val in vis_params.items():
		# 	print('{}: {}'.format(key, val))
		
		# print('Initializing WSI object')
		# wsi_object = initialize_wsi(slide_path, seg_mask_path=mask_file, seg_params=seg_params, filter_params=filter_params)
		# print('Done!')

		# wsi_ref_downsample = wsi_object.level_downsamples[patch_args.patch_level]

		# # the actual patch size for heatmap visualization should be the patch size * downsample factor * custom downsample factor
		# vis_patch_size = tuple((np.array(patch_size) * np.array(wsi_ref_downsample) * patch_args.custom_downsample).astype(int))

		# block_map_save_path = os.path.join(r_slide_save_dir, '{}_blockmap.h5'.format(slide_id))
		# mask_path = os.path.join(r_slide_save_dir, '{}_mask.jpg'.format(slide_id))
		# if vis_params['vis_level'] < 0:
		# 	best_level = wsi_object.wsi.get_best_level_for_downsample(32)
		# 	vis_params['vis_level'] = best_level
		# mask = wsi_object.visWSI(**vis_params, number_contours=True)
		# mask.save(mask_path)
		
		# features_path = os.path.join(r_slide_save_dir, slide_id+'.pt')
		# h5_path = os.path.join(r_slide_save_dir, slide_id+'.h5')
	

		# ##### check if h5_features_file exists ######
		# if not os.path.isfile(h5_path) :
		# 	_, _, wsi_object = compute_from_patches(wsi_object=wsi_object, 
		# 									model=model, 
		# 									feature_extractor=feature_extractor, 
		# 									batch_size=exp_args.batch_size, **blocky_wsi_kwargs, 
		# 									attn_save_path=None, feat_save_path=h5_path, 
		# 									ref_scores=None)				
		
		# ##### check if pt_features_file exists ######
		# if not os.path.isfile(features_path):
		# 	file = h5py.File(h5_path, "r")
		# 	features = torch.tensor(file['features'][:])
		# 	torch.save(features, features_path)
		# 	file.close()

		# # load features 
		# features = torch.load(features_path)
		# process_stack.loc[i, 'bag_size'] = len(features)
		
		# wsi_object.saveSegmentation(mask_file)
		# Y_hats, Y_hats_str, Y_probs, A = infer_single_slide(model, features, label, reverse_label_dict, exp_args.n_classes)
		# del features
		
		# if not os.path.isfile(block_map_save_path): 
		# 	file = h5py.File(h5_path, "r")
		# 	coords = file['coords'][:]
		# 	file.close()
		# 	asset_dict = {'attention_scores': A, 'coords': coords}
		# 	block_map_save_path = save_hdf5(block_map_save_path, asset_dict, mode='w')
		
		# # save top 3 predictions
		# for c in range(exp_args.n_classes):
		# 	process_stack.loc[i, 'Pred_{}'.format(c)] = Y_hats_str[c]
		# 	process_stack.loc[i, 'p_{}'.format(c)] = Y_probs[c]

		# os.makedirs('heatmaps/results/', exist_ok=True)
		# if data_args.process_list is not None:
		# 	process_stack.to_csv('heatmaps/results/{}.csv'.format(data_args.process_list.replace('.csv', '')), index=False)
		# else:
		# 	process_stack.to_csv('heatmaps/results/{}.csv'.format(exp_args.save_exp_code), index=False)
		
		# file = h5py.File(block_map_save_path, 'r')
		# dset = file['attention_scores']
		# coord_dset = file['coords']
		# scores = dset[:]
		# coords = coord_dset[:]
		# file.close()

		# # samples = sample_args.samples
		# for sample in samples:
		# 	if sample['sample']:
		# 		tag = "label_{}_pred_{}".format(label, Y_hats[0])
		# 		sample_save_dir =  os.path.join(exp_args.production_save_dir, exp_args.save_exp_code, 'sampled_patches', str(tag), sample['name'])
		# 		os.makedirs(sample_save_dir, exist_ok=True)
		# 		print('sampling {}'.format(sample['name']))
		# 		sample_results = sample_rois(scores, coords, k=sample['k'], mode=sample['mode'], seed=sample['seed'], 
		# 			score_start=sample.get('score_start', 0), score_end=sample.get('score_end', 1))
		# 		for idx, (s_coord, s_score) in enumerate(zip(sample_results['sampled_coords'], sample_results['sampled_scores'])):
		# 			print('coord: {} score: {:.3f}'.format(s_coord, s_score))
		# 			patch = wsi_object.wsi.read_region(tuple(s_coord), patch_args.patch_level, (patch_args.patch_size, patch_args.patch_size)).convert('RGB')
		# 			patch.save(os.path.join(sample_save_dir, '{}_{}_x_{}_y_{}_a_{:.3f}.png'.format(idx, slide_id, s_coord[0], s_coord[1], s_score)))

		# wsi_kwargs = {'top_left': top_left, 'bot_right': bot_right, 'patch_size': patch_size, 'step_size': step_size, 
		# 'custom_downsample':patch_args.custom_downsample, 'level': patch_args.patch_level, 'use_center_shift': heatmap_args.use_center_shift}

		# heatmap_save_name = '{}_blockmap.tiff'.format(slide_id)
		# if os.path.isfile(os.path.join(r_slide_save_dir, heatmap_save_name)):
		# 	pass
		# else:
		# 	heatmap = drawHeatmap(scores, coords, slide_path, wsi_object=wsi_object, cmap=heatmap_args.cmap, alpha=heatmap_args.alpha, use_holes=True, binarize=False, vis_level=-1, blank_canvas=False,
		# 					thresh=-1, patch_size = vis_patch_size, convert_to_percentiles=True)
		
		# 	heatmap.save(os.path.join(r_slide_save_dir, '{}_blockmap.png'.format(slide_id)))
		# 	del heatmap

		# save_path = os.path.join(r_slide_save_dir, '{}_{}_roi_{}.h5'.format(slide_id, patch_args.overlap, heatmap_args.use_roi))

		# if heatmap_args.use_ref_scores:
		# 	ref_scores = scores
		# else:
		# 	ref_scores = None
		
		# if heatmap_args.calc_heatmap:
		# 	compute_from_patches(wsi_object=wsi_object, clam_pred=Y_hats[0], model=model, feature_extractor=feature_extractor, batch_size=exp_args.batch_size, **wsi_kwargs, 
		# 						attn_save_path=save_path,  ref_scores=ref_scores)

		if not os.path.isfile(save_path):
			print('heatmap {} not found'.format(save_path))
			if heatmap_args.use_roi:
				save_path_full = os.path.join(r_slide_save_dir, '{}_{}_roi_False.h5'.format(slide_id, patch_args.overlap))
				print('found heatmap for whole slide')
				save_path = save_path_full
			else:
				continue

		# file = h5py.File(save_path, 'r')
		# dset = file['attention_scores']
		# coord_dset = file['coords']
		# scores = dset[:]
		# coords = coord_dset[:]
		# file.close()

		# heatmap_vis_args = {'convert_to_percentiles': True, 'vis_level': heatmap_args.vis_level, 'blur': heatmap_args.blur, 'custom_downsample': heatmap_args.custom_downsample}
		# if heatmap_args.use_ref_scores:
		# 	heatmap_vis_args['convert_to_percentiles'] = False

		# heatmap_save_name = '{}_{}_roi_{}_blur_{}_rs_{}_bc_{}_a_{}_l_{}_bi_{}_{}.{}'.format(slide_id, float(patch_args.overlap), int(heatmap_args.use_roi),
		# 																				int(heatmap_args.blur), 
		# 																				int(heatmap_args.use_ref_scores), int(heatmap_args.blank_canvas), 
		# 																				float(heatmap_args.alpha), int(heatmap_args.vis_level), 
		# 																				int(heatmap_args.binarize), float(heatmap_args.binary_thresh), heatmap_args.save_ext)


		# if os.path.isfile(os.path.join(p_slide_save_dir, heatmap_save_name)):
		# 	pass
		
		# else:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
		# 	heatmap = drawHeatmap(scores, coords, slide_path, wsi_object=wsi_object,  
		# 				          cmap=heatmap_args.cmap, alpha=heatmap_args.alpha, **heatmap_vis_args, 
		# 				          binarize=heatmap_args.binarize, 
		# 				  		  blank_canvas=heatmap_args.blank_canvas,
		# 				  		  thresh=heatmap_args.binary_thresh,  patch_size = vis_patch_size,
		# 				  		  overlap=patch_args.overlap, 
		# 				  		  top_left=top_left, bot_right = bot_right)
		# 	if heatmap_args.save_ext == 'jpg':
		# 		heatmap.save(os.path.join(p_slide_save_dir, heatmap_save_name), quality=100)
		# 	else:
		# 		heatmap.save(os.path.join(p_slide_save_dir, heatmap_save_name))
		
		if heatmap_args.save_orig:
			if heatmap_args.vis_level >= 0:
				vis_level = heatmap_args.vis_level
			else:
				vis_level = vis_params['vis_level']
			heatmap_save_name = '{}_orig_{}.{}'.format(slide_id,int(vis_level), heatmap_args.save_ext)
			if os.path.isfile(os.path.join(p_slide_save_dir, heatmap_save_name)):
				pass
			else:
				heatmap = wsi_object.visWSI(vis_level=vis_level, view_slide_only=True, custom_downsample=heatmap_args.custom_downsample)
				if heatmap_args.save_ext == 'jpg':
					heatmap.save(os.path.join(p_slide_save_dir, heatmap_save_name), quality=100)
				else:
					heatmap.save(os.path.join(p_slide_save_dir, heatmap_save_name))

	with open(os.path.join(exp_args.raw_save_dir, exp_args.save_exp_code, 'config.yaml'), 'w') as outfile:
		yaml.dump(config_dict, outfile, default_flow_style=False)

if __name__ == '__main__':
	main()


