import os
# pydrake
import pydrake

# pdc
from dense_correspondence_manipulation.utils.utils import set_cuda_visible_devices

GPU_LIST = [1]
set_cuda_visible_devices(GPU_LIST)



# torch
import torch

# pdc
from dense_correspondence.dataset.dynamic_drake_sim_dataset import DynamicDrakeSimDataset
from dense_correspondence_manipulation.utils.utils import getDenseCorrespondenceSourceDir, getDictFromYamlFilename

# dense correspondence manipulation
from dense_correspondence_manipulation.visualization.heatmap_visualization import HeatmapVisualization
from dense_correspondence_manipulation.utils import dev_utils

# key_dynam
from key_dynam.utils.utils import get_data_root, get_project_root
from key_dynam.experiments.exp_20_mugs import utils as exp_utils
from key_dynam.dense_correspondence.dc_drake_sim_episode_reader import DCDrakeSimEpisodeReader


def get_DD_model_file():

    # pth_file = "net_dy_epoch_7_iter_0_model.pth"
    pth_file = "net_best_model.pth"
    dataset_name = "mugs_random_colors_1000"
    model_name = "standard_2020-06-01-15-59-32-621007"



    pth_file = "net_best_model.pth"
    model_name = "data_aug_2020-06-01-19-26-37-655086"
    dataset_name = "mugs_correlle_mug-small_1000"


    pth_file = "net_best_model.pth"
    dataset_name = "correlle_mug-small_single_color_600"
    model_name = "data_aug_2020-06-01-23-14-05-694265"

    pth_file = "net_best_model.pth"
    dataset_name = "correlle_mug-small_single_color_600"
    model_name = "standard_2020-06-02-14-31-28-600354"

    pth_file = "net_best_model.pth"
    dataset_name = "single_corelle_mug_600"
    model_name = "data_aug_2020-06-02-20-56-54-192430"

    pth_file = "net_best_model.pth"
    dataset_name = "correlle_mug-small_single_color_600"
    model_name = "data_aug_2020-06-03-00-27-50-738970"

    pth_file = "net_best_model.pth"
    dataset_name = "correlle_mug-small_many_colors_600"
    model_name = "data_aug_2020-06-03-16-41-29-740641"


    model_file = os.path.join(get_data_root(),
                              "dev/experiments/20/dataset_%s/trained_models/perception/dense_descriptors/%s/%s" % (dataset_name, model_name, pth_file))

    return model_file, dataset_name



def main():

    model_file, dataset_name = get_DD_model_file()

    dataset_paths = exp_utils.get_dataset_paths(dataset_name)
    dataset_root = dataset_paths['dataset_root']
    multi_episode_dict = DCDrakeSimEpisodeReader.load_dataset(dataset_root, max_num_episodes=None)


    camera_list = ['camera_1_top_down']

    model_config_file = os.path.join(os.path.dirname(model_file), 'config.yaml')
    model_config = getDictFromYamlFilename(model_config_file)


    model_config['dataset']['data_augmentation'] = False
    dataset = DynamicDrakeSimDataset(model_config, multi_episode_dict, phase="valid") # could also use train data

    model = torch.load(model_file)
    model = model.cuda()
    model = model.eval()


    heatmap_vis = HeatmapVisualization(model_config,
                                       dataset,
                                       model,
                                       visualize_3D=False,
                                       camera_list=camera_list,
                                       verbose=True,
                                       sample_same_episode=False,
                                       display_confidence_value=False,
                                       use_custom_target_image_func=True)
    heatmap_vis.run()


if __name__ == "__main__":
    main()

