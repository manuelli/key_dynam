import os
import transforms3d
import numpy as np

import torch

from key_dynam.utils.utils import get_data_root, get_data_ssd_root, load_yaml, get_project_root
from key_dynam.dataset.episode_dataset import MultiEpisodeDataset
from key_dynam.utils import transform_utils
from key_dynam.utils.utils import random_sample_in_range



def mugs_random_colors_1000():
    dataset_name = "mugs_random_colors_1000"

    dataset_root = os.path.join(get_data_ssd_root(), 'dataset', dataset_name)
    config = load_yaml(os.path.join(dataset_root, 'config.yaml'))

    return {'dataset_name': dataset_name,
            'dataset_root': dataset_root,
            'config': config,
            'main_camera_name': 'camera_1_top_down',
            }

def mugs_correlle_mug_mall_1000():
    dataset_name = "mugs_correlle_mug-small_1000"

    dataset_root = os.path.join(get_data_ssd_root(), 'dataset', dataset_name)
    config = load_yaml(os.path.join(dataset_root, 'config.yaml'))

    return {'dataset_name': dataset_name,
            'dataset_root': dataset_root,
            'config': config,
            'main_camera_name': 'camera_1_top_down',
            }


def correlle_mug_small_single_color_600():
    dataset_name = "correlle_mug-small_single_color_600"

    dataset_root = os.path.join(get_data_ssd_root(), 'dataset', dataset_name)
    config = load_yaml(os.path.join(dataset_root, 'config.yaml'))

    return {'dataset_name': dataset_name,
            'dataset_root': dataset_root,
            'config': config,
            'main_camera_name': 'camera_1_top_down',
            }


def single_corelle_mug_600():
    dataset_name = "single_corelle_mug_600"

    dataset_root = os.path.join(get_data_ssd_root(), 'dataset', dataset_name)
    config = load_yaml(os.path.join(dataset_root, 'config.yaml'))

    return {'dataset_name': dataset_name,
            'dataset_root': dataset_root,
            'config': config,
            'main_camera_name': 'camera_1_top_down',
            }

def correlle_mug_small_many_colors_600():
    dataset_name = "correlle_mug-small_many_colors_600"

    dataset_root = os.path.join(get_data_ssd_root(), 'dataset', dataset_name)
    config = load_yaml(os.path.join(dataset_root, 'config.yaml'))

    dense_descriptor_model_chkpt = "/home/manuelli/data/key_dynam/dev/experiments/20/dataset_correlle_mug-small_many_colors_600/trained_models/perception/dense_descriptors/data_aug_2020-06-03-16-41-29-740641/net_best_model.pth"

    return {'dataset_name': dataset_name,
            'dataset_root': dataset_root,
            'config': config,
            'main_camera_name': 'camera_1_top_down',
            "dense_descriptor_model_chkpt": dense_descriptor_model_chkpt,
            }


def get_dataset_paths(dataset_name):
    if dataset_name == "correlle_mug-small_single_color_600":
        return correlle_mug_small_single_color_600()
    elif dataset_name == "single_corelle_mug_600":
        return single_corelle_mug_600()
    elif dataset_name == "correlle_mug-small_many_colors_600":
        return correlle_mug_small_many_colors_600()
    else:
        raise ValueError("unknown dataset:", dataset_name)


def get_env_config():
    return load_yaml(os.path.join(get_project_root(), 'experiments/exp_29_mugs/config.yaml'))
