import os

# dense_correspondence
from dense_correspondence_manipulation.utils import utils as pdc_utils

# key_dynam
from key_dynam.utils.utils import load_yaml, get_project_root, get_data_ssd_root

def get_dataset_paths(dataset_name):
    if dataset_name == "push_box_hardware":

        episodes_root = os.path.join(get_data_ssd_root(), "dataset/push_box_hardware")
        episodes_config = load_yaml(os.path.join(get_project_root(), 'experiments/exp_22_push_box_hardware/push_box_hardware_episodes_config.yaml'))

        transporter_model_chkpt = None
        dense_descriptor_model_chkpt = "/home/manuelli/data/key_dynam/dev/experiments/22/dataset_push_box_string_pull/trained_models/perception/dense_descriptors/data_aug_2020-07-02-02-39-27-400442/net_best_model.pth"

        return {'dataset_name': dataset_name,
                'dataset_root': episodes_root,
                'episodes_config': episodes_config,
                'main_camera_name': 'd415_01',
                'dense_descriptor_camera_list': ['d415_01', 'd415_02'],
                'transporter_model_chkpt': transporter_model_chkpt,
                'dense_descriptor_model_chkpt': dense_descriptor_model_chkpt,
                }
    elif dataset_name == "push_box_string_pull":
        episodes_root = os.path.join(get_data_ssd_root(), "dataset/push_box_string_pull")
        episodes_config = load_yaml(os.path.join(get_project_root(),
                                                 'experiments/exp_22_push_box_hardware/push_box_string_pull_episodes_config.yaml'))

        transporter_model_chkpt = None

        dense_descriptor_model_chkpt = "/home/manuelli/data/key_dynam/dev/experiments/22/dataset_push_box_string_pull/trained_models/perception/dense_descriptors/data_aug_2020-07-02-02-39-27-400442/net_best_model.pth"

        return {'dataset_name': dataset_name,
                'dataset_root': episodes_root,
                'episodes_config': episodes_config,
                'main_camera_name': 'd415_01',
                'dense_descriptor_camera_list': ['d415_01', 'd415_02'],
                'transporter_model_chkpt': transporter_model_chkpt,
                'dense_descriptor_model_chkpt': dense_descriptor_model_chkpt,
                }

    else:
        raise ValueError("unknown dataset:", dataset_name)