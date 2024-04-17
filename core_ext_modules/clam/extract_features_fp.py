# local imports
from utils.utils import print_network, collate_features
from utils.file_utils import save_hdf5
from datasets.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from models.resnet_custom import resnet50_baseline
from models.hipt_model_utils import get_vit256, get_vit4k

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import datetime
from math import floor
import os
import random
import numpy as np
import pdb
import time
from PIL import Image
import h5py
import openslide
import hydra
from omegaconf import DictConfig
import pathlib
import tqdm
import pandas as pd
from functools import partial


# UTILITIES

def compute_w_loader(file_path, output_path, wsi, model, device,
 	batch_size = 8, verbose = 0, print_every=20, pretrained=True, 
	custom_downsample=1, target_patch_size=-1, is_conch_model:bool=False):
	"""
	args:
		file_path: directory of bag (.h5 file)
		output_path: directory to save computed features (.h5 file)
		model: pytorch model
		batch_size: batch_size for computing features in batches
		verbose: level of feedback
		pretrained: use weights pretrained on imagenet
		custom_downsample: custom defined downscale factor of image patches
		target_patch_size: custom defined, rescaled image size before embedding
		# IET
		is_conch_model: bool
			Specify if this is the vit conch pretrained model. Since this is a vision language model, the encode_image method should be used.
		# END
	"""
	dataset = Whole_Slide_Bag_FP(file_path=file_path, wsi=wsi, pretrained=pretrained, 
		custom_downsample=custom_downsample, target_patch_size=target_patch_size)
	x, y = dataset[0]
	kwargs = {'num_workers': 4, 'pin_memory': True} if device.type == "cuda" else {}
	loader = DataLoader(dataset=dataset, batch_size=batch_size, **kwargs, collate_fn=collate_features)

	if verbose > 0:
		print('processing {}: total of {} batches'.format(file_path,len(loader)))

	mode = 'w'
	for count, (batch, coords) in enumerate(loader):
		with torch.no_grad():	
			if count % print_every == 0:
				print('batch {}/{}, {} files processed'.format(count, len(loader), count * batch_size))
			batch = batch.to(device, non_blocking=True)
			
			# IET
			if is_conch_model:
				features = model.encode_image(batch)
			else:
				features = model(batch)
			# END
				
			features = features.cpu().numpy()

			asset_dict = {'features': features, 'coords': coords}
			save_hdf5(output_path, asset_dict, attr_dict= None, mode=mode)
			mode = 'a'
	
	return output_path

# %% MAIN

@hydra.main(
    version_base="1.2.0", config_path=os.path.join('config', 'feature_extraction'), config_name='default'
)
def main(cfg:DictConfig):

	print('Feature extraction')
	# set up

	# # run id
	# run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M")
	run_id = datetime.datetime.now().strftime("%Y-%m-%d")

	# # select device on which to perform feature extraction
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	print(f'Running feature extraction on {device}.')

	# # Check if resuming experiment, if not create paths to where to save the model
	if cfg.resume:
		# check that the given path of the experiment to resume exists
		if not os.path.isdir(cfg.stopped_experiment_dir):
			raise ValueError(f'Resume experiment is set to True, but the given experiment directory does not exist.\nGIven {cfg.stopped_experiment_dir}.')
		else:
			save_dir = cfg.stopped_experiment_dir
		print(f'Resuming feature extraction (at {cfg.stopped_experiment_dir})')
	else:
		save_dir = pathlib.Path(os.path.join(cfg.feat_dir, cfg.experiment_name, run_id))
		save_dir.mkdir(parents=True, exist_ok=True)
		os.makedirs(os.path.join(save_dir, 'pt_files'), exist_ok=True)
		os.makedirs(os.path.join(save_dir, 'h5_files'), exist_ok=True)
	
	# # get list of .pt files in the save directory
	# dest_files = os.listdir(os.path.join(save_dir, 'pt_files'))

	# IET
	# refine to only contain.pt files
	dest_files = [f for f in os.listdir(os.path.join(save_dir, 'pt_files')) if f.endswith(".pt")]
	print(f'Found {len(dest_files)} .pt files in the feat_dir.')
	# END
	
	# # Dataset 
	print('Initializing dataset')
	csv_path = cfg.csv_path
	print('CSV FILE PATH ', csv_path)
	if csv_path is None:
		raise NotImplementedError

	bags_dataset = Dataset_All_Bags(csv_path)

	# IET
	# load .csv file that contains the file paths. This is to allow the feature extraciton to work 
	# not just on files in one folder.
	if os.path.isfile(cfg.data_slide_dir) and os.path.splitext(cfg.data_slide_dir)[1]=='.csv':
		df_file_paths = pd.read_csv(cfg.data_slide_dir, encoding="ISO-8859-1")
	# END
	

	print('Loading model checkpoint')
	if cfg.model_type == 'resnet50':
		model = resnet50_baseline(pretrained=True)
		model = model.to(device)
	elif cfg.model_type == 'vit_hipt':
		# import HIPT ViT pre-trained model
		model = get_vit256(pretrained_weights=os.path.join(cfg.pre_trained_model_archive, 'vit256_small_dino.pth')).to(device)
	elif cfg.model_type == 'vit_uni':
		# import UNI ViT pre-trained model
		import timm
		from torchvision import transforms
		
		model = timm.create_model(
    "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
		)
		model.load_state_dict(torch.load(os.path.join(cfg.pre_trained_model_archive, "vit224_large_uni_dino.bin"), map_location=device), strict=True)
		model = model.to(device)
	elif cfg.model_type == 'vit_conch':
		# import CONCH ViT pre-trained model (using the conch factory.py utility)
		from models.conch_open_clip_custom import create_model_from_pretrained
		model, _ = create_model_from_pretrained(model_cfg='conch_ViT-B-16', checkpoint_path=os.path.join(cfg.pre_trained_model_archive, "vit224_large_conch.bin"), device=device)
		model.forward = partial(model.encode_image, proj_contrast=False, normalize=False)
		model = model.to(device)

	if torch.cuda.device_count() > 1:
		model = nn.DataParallel(model)
	
	# perform feature extraction
	model.eval()
	total = len(bags_dataset)

	skipped_slide_ids = []

	for bag_candidate_idx in range(total):
		slide_id = bags_dataset[bag_candidate_idx].split(cfg.slide_ext)[0]
		bag_name = slide_id +'.h5'
		h5_file_path = os.path.join(cfg.data_h5_dir, 'patches', bag_name)

		# IET 
		# skip the missing .h5 patch files
		if cfg.skip_missing_h5_patches and not os.path.isfile(h5_file_path):
			print(f'ATTENTION!!!\nSkipping slide id since the .h5 patch file does not exist {slide_id}')
			skipped_slide_ids.append(skipped_slide_ids)
			continue
		# END

		# IET
		if os.path.isdir(cfg.data_slide_dir):
			slide_file_path = os.path.join(cfg.data_slide_dir, slide_id+cfg.slide_ext)
		elif os.path.isfile(cfg.data_slide_dir) and os.path.splitext(cfg.data_slide_dir)[1]=='.csv':
			# get file path from the given .csv file (slide_path column)
			slide_file_path = df_file_paths.loc[df_file_paths.slide_id==slide_id].slide_path.values[0]
		else:
			raise NotImplementedError
		# END
		print(slide_file_path)

		print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
		print(slide_id)

		if cfg.auto_skip and slide_id+'.pt' in dest_files:
			print('skipped {}'.format(slide_id))
			continue 

		output_path = os.path.join(save_dir, 'h5_files', bag_name)
		time_start = time.time()
		wsi = openslide.open_slide(slide_file_path)
		output_file_path = compute_w_loader(h5_file_path, output_path, wsi, 
		model = model, device=device, batch_size = cfg.batch_size, verbose = 1, print_every = 20, 
		custom_downsample=cfg.custom_downsample, target_patch_size=cfg.target_patch_size)
		time_elapsed = time.time() - time_start
		print('\ncomputing features for {} took {} s'.format(output_file_path, time_elapsed))
		file = h5py.File(output_file_path, "r")

		features = file['features'][:]
		print('features size: ', features.shape)
		print('coordinates size: ', file['coords'].shape)
		features = torch.from_numpy(features)
		bag_base, _ = os.path.splitext(bag_name)
		torch.save(features, os.path.join(save_dir, 'pt_files', bag_base+'.pt'))
	
	# print the skipped slide_ids
	if skipped_slide_ids:
		print(f'In total, {len(skipped_slide_ids)} were skipped since .h5 patch file was not found.')
		[print(sid) for sid in skipped_slide_ids]
	
	print('\nFeature extraction completed.')


if __name__ == '__main__':
	main()


