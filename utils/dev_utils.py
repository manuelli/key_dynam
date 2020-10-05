"""
Useful helper functions for loading dataset during development
"""

# system
import os

# key_dynam
from key_dynam.utils.utils import get_data_root, load_yaml, get_project_root
from key_dynam.dataset.drake_sim_episode_reader import DrakeSimEpisodeReader



def load_drake_pusher_slider_episodes(**kwargs):
    """
    Helper function for loading drake pusher slider dataset
    :param descriptor_images_root:
    :type descriptor_images_root:
    :return:
    :rtype:
    """
    # DATASET_NAME = "top_down_rotated"
    DATASET_NAME = "2019-12-05-15-58-48-462834_top_down_rotated_250"
    dataset_root = os.path.join(get_data_root(), "dev/experiments/05/data", DATASET_NAME)

    multi_episode_dict = DrakeSimEpisodeReader.load_dataset(dataset_root,
                                                            **kwargs)

    return multi_episode_dict


def load_default_config():
    """
    Loads the experiments/05/config.yaml
    :return:
    :rtype:
    """
    config_file = os.path.join(get_project_root(), 'experiments/05/config.yaml')
    config = load_yaml(config_file)
    return config


def load_simple_config():
    config_file = os.path.join(get_project_root(), 'config/simple_config.yaml')
    config = load_yaml(config_file)
    return config


