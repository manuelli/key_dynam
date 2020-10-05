import os
# pydrake
import pydrake

# torch
import torch

# pdc
from dense_correspondence.dataset.dynamic_drake_sim_dataset import DynamicDrakeSimDataset
from dense_correspondence.dataset.dynamic_spartan_episode_reader import DynamicSpartanEpisodeReader as DCDynamicSpartanEpisodeReader
from dense_correspondence_manipulation.utils.utils import getDenseCorrespondenceSourceDir, getDictFromYamlFilename

# dense correspondence manipulation
from dense_correspondence_manipulation.visualization.heatmap_visualization import HeatmapVisualization
from dense_correspondence_manipulation.utils import dev_utils

# key_dynam
from key_dynam.experiments.exp_22_push_box_hardware import utils as exp_utils
from key_dynam.utils.torch_utils import get_freer_gpu, set_cuda_visible_devices
from key_dynam.utils.utils import get_data_root



def main():
    dataset_name = "push_box_hardware"
    dataset_name = "push_box_string_pull"

    model_file = None

    if dataset_name == "push_box_hardware":
        model_file = "/home/manuelli/data_ssd/key_dynam/trained_models/perception/dense_descriptors/2020-02-27-00-32-58_3D_loss_resnet50__dataset_real_push_box/net_dy_epoch_3_iter_1000_model.pth"
    elif dataset_name == "push_box_string_pull":
        # model_name = "data_aug_2020-07-02-02-39-27-400442"
        # model_file = os.path.join(get_data_root(), 'dev/experiments/22/dataset_push_box_string_pull/net_')

        model_file = "/home/manuelli/data/key_dynam/dev/experiments/22/dataset_push_box_string_pull/trained_models/perception/dense_descriptors/data_aug_2020-07-02-02-39-27-400442/net_best_model.pth"


    # dataset_paths = exp_utils.get_dataset_paths(dataset_name)
    dataset_paths = exp_utils.get_dataset_paths("push_box_hardware")
    episodes_config = dataset_paths['episodes_config']
    episodes_root = dataset_paths['dataset_root']

    multi_episode_dict = DCDynamicSpartanEpisodeReader.load_dataset(config=episodes_config,
                                                                        episodes_root=episodes_root)
    camera_list = ['d415_01']
    target_camera_list = ['d415_01']

    config = dev_utils.load_dataset_config()

    model_config_file = os.path.join(os.path.dirname(model_file), 'config.yaml')
    model_config = getDictFromYamlFilename(model_config_file)

    # model_config['dataset']['data_augmentation'] = False
    config['dataset']['data_augmentation'] = False
    dataset = DynamicDrakeSimDataset(config, multi_episode_dict, phase="train") # could also use train data

    model = torch.load(model_file)
    model = model.cuda()
    model = model.eval()


    heatmap_vis = HeatmapVisualization(config,
                                       dataset,
                                       model,
                                       visualize_3D=False,
                                       camera_list=camera_list,
                                       verbose=True,
                                       sample_same_episode=False,
                                       display_confidence_value=True,
                                       use_custom_target_image_func=False,
                                       target_camera_names=target_camera_list)
    heatmap_vis.run()


if __name__ == "__main__":
    set_cuda_visible_devices([get_freer_gpu()])
    main()

