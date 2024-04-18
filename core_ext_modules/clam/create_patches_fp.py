# internal imports
from wsi_core.WholeSlideImage import WholeSlideImage
from wsi_core.wsi_utils import StitchCoords
from wsi_core.batch_process_utils import initialize_df

# other imports
import os
import datetime
import numpy as np
import time
import argparse
import pdb
import pandas as pd
import hydra
from omegaconf import DictConfig
import pathlib

# %% UTILITIES
def stitching(file_path, wsi_object, downscale = 64):
	start = time.time()
	heatmap = StitchCoords(file_path, wsi_object, downscale=downscale, bg_color=(0,0,0), alpha=-1, draw_grid=False)
	total_time = time.time() - start
	
	return heatmap, total_time

def segment(WSI_object, seg_params = None, filter_params = None, mask_file = None):
	### Start Seg Timer
	start_time = time.time()
	# Use segmentation file
	if mask_file is not None:
		WSI_object.initSegmentation(mask_file)
	# Segment	
	else:
		WSI_object.segmentTissue(**seg_params, filter_params=filter_params)

	### Stop Seg Timers
	seg_time_elapsed = time.time() - start_time   
	return WSI_object, seg_time_elapsed

def patching(WSI_object, **kwargs):
	### Start Patch Timer
	start_time = time.time()

	# Patch
	file_path = WSI_object.process_contours(**kwargs)


	### Stop Patch Timer
	patch_time_elapsed = time.time() - start_time
	return file_path, patch_time_elapsed

def get_closest_downsample_level(wsi, base_magnification:float=20.0, downsample:int=1, print_info:bool=True):
	'''
	Utility that given a wsi (openslide object), looks at the available levels and their downsample ratios 
	and finds the one closes to the combination of requested base_magnification and downsample.
	Eg.
		wsi objective_magnification=40, base_magnification=20, downsample=2.
		patch_level_downsample = closet wsi downsamples to objective_magnification / (base_magnification/downsample)
	'''

	# print levels, downsamples and objective power
	levels, downsamples = wsi.getWSIlevels()
	objective_power = wsi.getObjectivePower()
	if print_info:
		print(f'Available levels: {levels}')
		print(f'Corresponding downsample factors: {downsamples}')
		print(f'Objective power: {objective_power}')

	# update the base_magnification based on the downsample
	ref_base_magnification = base_magnification / downsample

	# # Compute what is the down sampling that should be performed given the original and the requested base_magnification.
	# # Use this to find the level at which the slide should be opened at by looking at the downsample of each available level (the closes should be selected).

	downsample_ratio = round(objective_power / ref_base_magnification)
	# find the closes in the list of downsamples
	if downsample_ratio < 1:
		raise ValueError(f'The requested base magnification is higher than the highest available for this slide ({slide}). Objective magnification < requested magnification. {objective_power} < {patch_reference_object_magnification}')
	else:
		downsamples_difference = [abs(d - downsample_ratio) for d in downsamples]
		index_level_downsample = downsamples_difference.index(min(downsamples_difference))
		# adjust the level at which tp perform the patching
		if print_info:
			print(f'Extracting patches at level {index_level_downsample} which has a downsample of {downsamples[index_level_downsample]}')
	
	return index_level_downsample


def seg_and_patch(source, save_dir, patch_save_dir, mask_save_dir, stitch_save_dir, 
				  patch_size = 256, step_size = 256, 
				  seg_params = {'seg_level': -1, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False,
				  'keep_ids': 'none', 'exclude_ids': 'none'},
				  filter_params = {'a_t':100, 'a_h': 16, 'max_n_holes':8}, 
				  vis_params = {'vis_level': -1, 'line_thickness': 500},
				  patch_params = {'use_padding': True, 'contour_fn': 'four_pt'},
				  patch_reference_object_magnification=20.0,
				  patch_downsample = None,
				  use_default_params = False, 
				  seg = False, save_mask = True, 
				  stitch= False, 
				  patch = False, auto_skip=True, process_list = None,
				  save_using_anonymized_slide_id:bool=False, 
				  anonymized_slide_ids:list=None):

	# IET 
	if os.path.isfile(source):
		# loading the .csv file with the paths to the WSIs to process
		aus_df = pd.read_csv(source)
		slides = list(aus_df.slide_path)
		# slides = sorted(list(slides.slide_path))
		if save_using_anonymized_slide_id:
			anonymized_slide_ids = list(aus_df.slide_id)
	else:
		slides = sorted(os.listdir(source))
		if save_using_anonymized_slide_id:
			if not anonymized_slide_ids:
				raise ValueError(f'Requested to save the processed WSIs using an anonymized slide_id, but none were given.')
	# END
		
	slides = [slide for slide in slides if os.path.isfile(os.path.join(source, slide))]
	if process_list is None:
		df = initialize_df(slides, seg_params, filter_params, vis_params, patch_params, use_anonymized_slide_ids=save_using_anonymized_slide_id, anonymized_slide_ids=anonymized_slide_ids)
	else:
		df = pd.read_csv(process_list)
		df = initialize_df(df, seg_params, filter_params, vis_params, patch_params, use_anonymized_slide_ids=save_using_anonymized_slide_id, anonymized_slide_ids=anonymized_slide_ids)

	mask = df['process'] == 1
	process_stack = df[mask]

	total = len(process_stack)

	legacy_support = 'a' in df.keys()
	if legacy_support:
		print('detected legacy segmentation csv file, legacy support enabled')
		df = df.assign(**{'a_t': np.full((len(df)), int(filter_params['a_t']), dtype=np.uint32),
		'a_h': np.full((len(df)), int(filter_params['a_h']), dtype=np.uint32),
		'max_n_holes': np.full((len(df)), int(filter_params['max_n_holes']), dtype=np.uint32),
		'line_thickness': np.full((len(df)), int(vis_params['line_thickness']), dtype=np.uint32),
		'contour_fn': np.full((len(df)), patch_params['contour_fn'])})

	seg_times = 0.
	patch_times = 0.
	stitch_times = 0.

	for i in range(total):
		df.to_csv(os.path.join(save_dir, 'process_list_autogen.csv'), index=False)
		idx = process_stack.index[i]
		slide = process_stack.loc[idx, 'slide_id']
		print("\n\nprogress: {:.2f}, {}/{}".format(i/total, i, total))
		print('processing {}'.format(slide))
		
		df.loc[idx, 'process'] = 0

		# IET
		# slide_id, _ = os.path.splitext(slide)
		if save_using_anonymized_slide_id:
			slide_id = process_stack.loc[idx, 'anonymized_slide_id']
		else:
			slide_id, _ = os.path.basename(os.path.splitext(slide))
		# END

		if auto_skip and os.path.isfile(os.path.join(patch_save_dir, slide_id + '.h5')):
			print('{} already exist in destination location, skipped'.format(slide_id))
			df.loc[idx, 'status'] = 'already_exist'
			continue

		# Inialize WSI
		# IET
		if os.path.isdir(source):
			full_path = os.path.join(source, slide)
		else:
			full_path = slide
		# END
			
		WSI_object = WholeSlideImage(full_path)

		# IET
		if save_using_anonymized_slide_id:
			WSI_object.name = slide_id
		# END

		if use_default_params:
			current_vis_params = vis_params.copy()
			current_filter_params = filter_params.copy()
			current_seg_params = seg_params.copy()
			current_patch_params = patch_params.copy()
			
		else:
			current_vis_params = {}
			current_filter_params = {}
			current_seg_params = {}
			current_patch_params = {}


			for key in vis_params.keys():
				if legacy_support and key == 'vis_level':
					df.loc[idx, key] = -1
				current_vis_params.update({key: df.loc[idx, key]})

			for key in filter_params.keys():
				if legacy_support and key == 'a_t':
					old_area = df.loc[idx, 'a']
					seg_level = df.loc[idx, 'seg_level']
					scale = WSI_object.level_downsamples[seg_level]
					adjusted_area = int(old_area * (scale[0] * scale[1]) / (512 * 512))
					current_filter_params.update({key: adjusted_area})
					df.loc[idx, key] = adjusted_area
				current_filter_params.update({key: df.loc[idx, key]})

			for key in seg_params.keys():
				if legacy_support and key == 'seg_level':
					df.loc[idx, key] = -1
				current_seg_params.update({key: df.loc[idx, key]})

			for key in patch_params.keys():
				current_patch_params.update({key: df.loc[idx, key]})

		if current_vis_params['vis_level'] < 0:
			if len(WSI_object.level_dim) == 1:
				current_vis_params['vis_level'] = 0
			
			else:	
				wsi = WSI_object.getOpenSlide()
				best_level = wsi.get_best_level_for_downsample(64)
				current_vis_params['vis_level'] = best_level

		if current_seg_params['seg_level'] < 0:
			if len(WSI_object.level_dim) == 1:
				current_seg_params['seg_level'] = 0
			
			else:
				wsi = WSI_object.getOpenSlide()
				best_level = wsi.get_best_level_for_downsample(64)
				current_seg_params['seg_level'] = best_level

		keep_ids = str(current_seg_params['keep_ids'])
		if keep_ids != 'none' and len(keep_ids) > 0:
			str_ids = current_seg_params['keep_ids']
			current_seg_params['keep_ids'] = np.array(str_ids.split(',')).astype(int)
		else:
			current_seg_params['keep_ids'] = []

		exclude_ids = str(current_seg_params['exclude_ids'])
		if exclude_ids != 'none' and len(exclude_ids) > 0:
			str_ids = current_seg_params['exclude_ids']
			current_seg_params['exclude_ids'] = np.array(str_ids.split(',')).astype(int)
		else:
			current_seg_params['exclude_ids'] = []

		w, h = WSI_object.level_dim[current_seg_params['seg_level']] 
		if w * h > 1e8:
			print('level_dim {} x {} is likely too large for successful segmentation, aborting'.format(w, h))
			df.loc[idx, 'status'] = 'failed_seg'
			continue

		df.loc[idx, 'vis_level'] = current_vis_params['vis_level']
		df.loc[idx, 'seg_level'] = current_seg_params['seg_level']


		seg_time_elapsed = -1
		if seg:
			WSI_object, seg_time_elapsed = segment(WSI_object, current_seg_params, current_filter_params) 

		if save_mask:
			mask = WSI_object.visWSI(**current_vis_params)
			mask_path = os.path.join(mask_save_dir, slide_id+'.jpg')
			mask.save(mask_path)

		patch_time_elapsed = -1 # Default time
		if patch:
			# update level at which to perform the patching 
			updated_patch_level = get_closest_downsample_level(WSI_object, patch_reference_object_magnification, )
			current_patch_params.update({'patch_level': updated_patch_level, 'patch_size': patch_size, 'step_size': step_size, 
										 'save_path': patch_save_dir})
			file_path, patch_time_elapsed = patching(WSI_object = WSI_object,  **current_patch_params,)
		
		stitch_time_elapsed = -1
		if stitch:
			file_path = os.path.join(patch_save_dir, slide_id+'.h5')
			if os.path.isfile(file_path):
				heatmap, stitch_time_elapsed = stitching(file_path, WSI_object, downscale=64)
				stitch_path = os.path.join(stitch_save_dir, slide_id+'.jpg')
				heatmap.save(stitch_path)

		print("segmentation took {} seconds".format(seg_time_elapsed))
		print("patching took {} seconds".format(patch_time_elapsed))
		print("stitching took {} seconds".format(stitch_time_elapsed))
		df.loc[idx, 'status'] = 'processed'

		seg_times += seg_time_elapsed
		patch_times += patch_time_elapsed
		stitch_times += stitch_time_elapsed

	seg_times /= total
	patch_times /= total
	stitch_times /= total

	df.to_csv(os.path.join(save_dir, 'process_list_autogen.csv'), index=False)
	print("average segmentation time in s per slide: {}".format(seg_times))
	print("average patching time in s per slide: {}".format(patch_times))
	print("average stiching time in s per slide: {}".format(stitch_times))
		
	return seg_times, patch_times

# %% MAIN 

@hydra.main(
    version_base="1.2.0", config_path=os.path.join('config', 'patch_extraction'), config_name='default'
)
def main(cfg: DictConfig):

	# set up
	# # run id
	# run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M")
	run_id = datetime.datetime.now().strftime("%Y-%m-%d")

	# # check if resume is true, else create dir to where to save the .h5 patch files.
	if cfg.resume:
		# check that the given path of the experiment to resume exists
		if not os.path.isdir(cfg.stopped_experiment_dir):
			raise ValueError(f'Resume experiment is set to True, but the given experiment directory does not exist.\nGIven {cfg.stopped_experiment_dir}.')
		else:
			save_dir = cfg.stopped_experiment_dir

		print(f'Resuming patching at {cfg.stopped_experiment_dir}')
	else:
		save_dir = pathlib.Path(os.path.join(cfg.save_dir, cfg.experiment_name, run_id))
		save_dir.mkdir(parents=True, exist_ok=True)

	# # build path to all the other folders
	patch_save_dir = os.path.join(save_dir, 'patches')
	mask_save_dir = os.path.join(save_dir, 'masks')
	stitch_save_dir = os.path.join(save_dir, 'stitches')

	if cfg.process_list:
		process_list = os.path.join(save_dir, cfg.process_list)

	else:
		process_list = None

	print('source: ', cfg.source)
	print('patch_save_dir: ', patch_save_dir)
	print('mask_save_dir: ', mask_save_dir)
	print('stitch_save_dir: ', stitch_save_dir)
	
	directories = {'source': cfg.source, 
				   'save_dir': save_dir,
				   'patch_save_dir': patch_save_dir, 
				   'mask_save_dir' : mask_save_dir, 
				   'stitch_save_dir': stitch_save_dir} 

	for key, val in directories.items():
		print("{} : {}".format(key, val))
		if key not in ['source']:
			os.makedirs(val, exist_ok=True)

	# seg_params = {'seg_level': -1, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False,
	# 			  'keep_ids': 'none', 'exclude_ids': 'none'}
	# filter_params = {'a_t':100, 'a_h': 16, 'max_n_holes':8}
	# vis_params = {'vis_level': -1, 'line_thickness': 250}
	# patch_params = {'use_padding': True, 'contour_fn': 'four_pt'}

	if cfg.preset:
		preset_df = pd.read_csv(os.path.join('presets', cfg.preset))
		for key in cfg.seg_params.keys():
			cfg.seg_params[key] = preset_df.loc[0, key]

		for key in cfg.filter_params.keys():
			cfg.filter_params[key] = preset_df.loc[0, key]

		for key in cfg.vis_params.keys():
			cfg.vis_params[key] = preset_df.loc[0, key]

		for key in cfg.patch_params.keys():
			cfg.patch_params[key] = preset_df.loc[0, key]
	
	parameters = {'seg_params': cfg.seg_params,
				  'filter_params': cfg.filter_params,
	 			  'patch_params': cfg.patch_params,
				  'vis_params': cfg.vis_params}

	print(parameters)

	seg_times, patch_times = seg_and_patch(**directories, **parameters,
											patch_size = cfg.patch_size, step_size=cfg.step_size, 
											seg = cfg.seg,  use_default_params=False, save_mask = cfg.save_mask, 
											stitch= cfg.stitch, 
											patch = cfg.patch,
											patch_reference_object_magnification=cfg.base_magnification,
											patch_downsample = cfg.downsample,
											process_list = process_list, 
											auto_skip=cfg.no_auto_skip, 
											save_using_anonymized_slide_id=cfg.save_using_anonymized_slide_ids)

if __name__ == '__main__':
	main()
