import os
import numpy as np
import matplotlib.pyplot as plt
import time
import cv2

import pydrake


# torch
import torch


from key_dynam.utils.utils import get_project_root, save_yaml, load_yaml, get_current_YYYY_MM_DD_hh_mm_ss_ms, get_data_root
from key_dynam.dense_correspondence.dc_drake_sim_episode_reader import DCDrakeSimEpisodeReader

import dense_correspondence_manipulation.utils.utils as pdc_utils
from dense_correspondence.training.train_integral_heatmap_3d import train_dense_descriptors
from dense_correspondence_manipulation.utils import dev_utils
from dense_correspondence_manipulation.utils.utils import set_cuda_visible_devices

set_cuda_visible_devices([0])



# load config

# specify the dataset
dataset_name = "dps_box_on_side_600"
# prepare folders
data_dir = os.path.join(get_data_root(), 'dev/experiments/18/data', dataset_name)

start_time = time.time()

multi_episode_dict = DCDrakeSimEpisodeReader.load_dataset(dataset_root=data_dir)

# placeholder for now
config = dev_utils.load_integral_heatmap_3d_config()
config['dataset']['name'] = dataset_name
config['dataset']['camera_names'] = ['camera_angled', 'camera_angled_rotated']
config['dataset']['train_valid_ratio'] = 0.83


model_name = "3D_loss_%s_%s" %("camera_angled", get_current_YYYY_MM_DD_hh_mm_ss_ms())

train_dir = os.path.join(get_data_root(), 'dev/experiments/18/trained_models/perception/dense_descriptor',
                         model_name)

train_dense_descriptors(config,
                        train_dir=train_dir,
                        multi_episode_dict=multi_episode_dict,
                        verbose=False)