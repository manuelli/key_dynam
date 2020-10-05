import os
import transforms3d
import numpy as np

import torch

from key_dynam.utils.utils import get_data_root, get_data_ssd_root, load_yaml
from key_dynam.dataset.episode_dataset import MultiEpisodeDataset
from key_dynam.utils import transform_utils
from key_dynam.utils.utils import random_sample_in_range

TRAIN_VALID_RATIO = 0.83

def box_on_side_dataset_root():
    dataset_name = "dps_box_on_side_600"
    # dataset_root = os.path.join(get_data_root(), "dev/experiments/18/data", dataset_name)
    dataset_root = os.path.join(get_data_ssd_root(), 'dataset', dataset_name)
    config = load_yaml(os.path.join(dataset_root, 'config.yaml'))

    dense_descriptor_model_chkpt = "/home/manuelli/data/key_dynam/dev/experiments/drake_pusher_slider_box_on_side/dataset_dps_box_on_side_600/trained_models/perception/dense_descriptor/3D_loss_camera_angled_2020-05-13-23-39-35-818188/net_best_dy_model.pth"

    return {'dataset_name': dataset_name,
            'dataset_root': dataset_root,
            'config': config,
            'main_camera_name': 'camera_angled',
            'dense_descriptor_model_chkpt': dense_descriptor_model_chkpt,
            }

def get_dataset_paths(dataset_name):
    if dataset_name == "dps_box_on_side_600":
        return box_on_side_dataset_root()
    else:
        raise ValueError("unknown dataset:", dataset_name)

