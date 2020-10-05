import os
import transforms3d
import numpy as np

import torch

from key_dynam.utils.utils import get_data_root, get_data_ssd_root, load_yaml
from key_dynam.dataset.episode_dataset import MultiEpisodeDataset
from key_dynam.utils import transform_utils
from key_dynam.utils.utils import random_sample_in_range



def top_down_dataset_root():
    dataset_name = "2020-04-20-14-58-21-418302_T_aug_random_velocity_1000"
    # dataset_root = os.path.join(get_data_root(), "dev/experiments/09/data", dataset_name)
    dataset_root = os.path.join(get_data_ssd_root(), 'dataset', dataset_name)
    config = load_yaml(os.path.join(dataset_root, 'config.yaml'))

    return {'dataset_name': dataset_name,
            'dataset_root': dataset_root,
            'config': config,
            'main_camera_name': 'camera_1_top_down'}


def angled_cam_dataset_root():
    dataset_name = "2020-04-23-20-45-12-697915_T_aug_random_velocity_1000_angled_cam"

    # prepare folders
    # dataset_root = os.path.join(get_data_root(), 'dev/experiments/10/data', dataset_name)
    dataset_root = os.path.join(get_data_ssd_root(), 'dataset', dataset_name)
    config = load_yaml(os.path.join(dataset_root, 'config.yaml'))

    return {'dataset_name': dataset_name,
            'dataset_root': dataset_root,
            'config': config,
            'main_camera_name': 'camera_angled',
            }


def get_dataset_paths(dataset_name):
    if dataset_name == "2020-04-20-14-58-21-418302_T_aug_random_velocity_1000":
        return top_down_dataset_root()
    elif dataset_name == "2020-04-23-20-45-12-697915_T_aug_random_velocity_1000_angled_cam":
        return angled_cam_dataset_root()
    elif dataset_name == "box_push_1000_top_down":
        dataset_root = os.path.join(get_data_ssd_root(), 'dataset', "box_push_1000")
        config = load_yaml(os.path.join(dataset_root, 'config.yaml'))
        transporter_model_chkpt = "/home/manuelli/data/key_dynam/dev/experiments/drake_pusher_slider_v2/dataset_box_push_1000_top_down/trained_models/perception/transporter/transporter_standard_2020-06-14-22-29-31-256422/train_nKp6_invStd10.0/net_best.pth"

        dense_descriptor_model_chkpt = "/home/manuelli/data/key_dynam/dev/experiments/drake_pusher_slider_v2/dataset_box_push_1000_top_down/trained_models/perception/dense_descriptors/data_aug_2020-06-14-21-47-52-389769/net_best_model.pth"


        return {'dataset_name': dataset_name,
                'dataset_root': dataset_root,
                'config': config,
                'main_camera_name': 'camera_1_top_down',
                'dense_descriptor_camera_list': ['camera_1_top_down', 'camera_2_top_down_rotated'],
                'transporter_model_chkpt': transporter_model_chkpt,
                'dense_descriptor_model_chkpt': dense_descriptor_model_chkpt,
                }
    elif dataset_name == "box_push_1000_angled":
        dataset_root = os.path.join(get_data_ssd_root(), 'dataset', "box_push_1000")
        config = load_yaml(os.path.join(dataset_root, 'config.yaml'))
        transporter_model_chkpt = "/home/manuelli/data/key_dynam/dev/experiments/drake_pusher_slider_v2/dataset_box_push_1000_angled/trained_models/perception/transporter/transporter_standard_2020-06-15-18-35-52-478769/train_nKp6_invStd10.0/net_best.pth"

        dense_descriptor_model_chkpt = "/home/manuelli/data/key_dynam/dev/experiments/drake_pusher_slider_v2/dataset_box_push_1000_angled/trained_models/perception/dense_descriptors/data_aug_2020-06-15-15-39-24-127276/net_best_model.pth"

        return {'dataset_name': dataset_name,
                'dataset_root': dataset_root,
                'config': config,
                'main_camera_name': 'camera_angled',
                'dense_descriptor_camera_list': ['camera_angled', 'camera_angled_rotated'],
                'transporter_model_chkpt': transporter_model_chkpt,
                'dense_descriptor_model_chkpt': dense_descriptor_model_chkpt,
                }
    else:
        raise ValueError("unknown dataset:", dataset_name)



def sample_slider_position(T_aug=None):
    # always at the origin basically
    pos = np.array([0, 0, 0.03])

    yaw_min = np.array([0])
    yaw_max = np.array([2 * np.pi])
    yaw = random_sample_in_range(yaw_min, yaw_max)

    quat = transforms3d.euler.euler2quat(0, 0, yaw)

    T_O_slider = transform_utils.transform_from_pose(pos, quat)

    T_W_slider = None
    if T_aug is not None:
        T_W_slider = T_aug @ T_O_slider
    else:
        T_W_slider = T_O_slider

    pose_dict = transform_utils.matrix_to_dict(T_W_slider)

    # note the quat/pos orderining
    q = np.concatenate((pose_dict['quaternion'], pose_dict['position']))

    return q


def sample_pusher_position_and_velocity(T_aug=None):
    low = np.array([-0.16, -0.08])
    high = np.array([-0.15, 0.08])
    q_pusher = random_sample_in_range(low, high)

    q_pusher_3d_homog = np.array([0, 0, 0, 1.0])
    q_pusher_3d_homog[:2] = q_pusher


    # sample pusher velocity
    vel_min = np.array([0.15])
    vel_max = np.array([0.25])
    magnitude = random_sample_in_range(vel_min, vel_max)
    v_pusher_3d = magnitude * np.array([1, 0, 0])

    if T_aug is not None:
        v_pusher_3d = T_aug[:3, :3] @ v_pusher_3d
        q_pusher_3d_homog = T_aug @ q_pusher_3d_homog

    v_pusher = v_pusher_3d[:2]
    q_pusher = q_pusher_3d_homog[:2]

    return {'q_pusher': q_pusher,
            'v_pusher': v_pusher,
            }


def generate_initial_condition(config=None,
                               T_aug_enabled=True,
                               N=10):
    T_aug = None
    if T_aug_enabled:
        pos_min = config['eval']['T_aug']['pos_min']
        pos_max = config['eval']['T_aug']['pos_max']
        yaw_min = config['eval']['T_aug']['yaw_min']
        yaw_max = config['eval']['T_aug']['yaw_max']
        T_aug = MultiEpisodeDataset.sample_T_aug(pos_min=pos_min,
                                                 pos_max=pos_max,
                                                 yaw_min=yaw_min,
                                                 yaw_max=yaw_max, )

    q_slider = sample_slider_position(T_aug=T_aug)
    d = sample_pusher_position_and_velocity(T_aug=T_aug)

    action_sequence = torch.Tensor(d['v_pusher']).unsqueeze(0).expand([N, -1])

    return {'q_slider': q_slider,
            'q_pusher': d['q_pusher'],
            'v_slider': d['v_pusher'],
            'action_sequence': action_sequence,
            }