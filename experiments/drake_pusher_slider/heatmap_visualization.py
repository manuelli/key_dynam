import os
# pydrake
import pydrake

# pdc
from dense_correspondence_manipulation.utils.utils import set_cuda_visible_devices




# torch
import torch

# pdc
from dense_correspondence.dataset.dynamic_drake_sim_dataset import DynamicDrakeSimDataset
from dense_correspondence_manipulation.utils.utils import getDenseCorrespondenceSourceDir, getDictFromYamlFilename

# dense correspondence manipulation
from dense_correspondence_manipulation.visualization.heatmap_visualization import HeatmapVisualization

# key_dynam
from key_dynam.utils.utils import get_data_root, get_project_root
from key_dynam.utils.torch_utils import get_freer_gpu
from key_dynam.experiments.drake_pusher_slider import utils as exp_utils
from key_dynam.dense_correspondence.dc_drake_sim_episode_reader import DCDrakeSimEpisodeReader



def get_DD_model_file():

    # pth_file = "net_dy_epoch_7_iter_0_model.pth"
    pth_file = "net_best_model.pth"
    dataset_name = "box_push_1000_top_down"
    model_name = "data_aug_2020-06-14-21-47-52-389769"

    model_file = os.path.join(get_data_root(),
                              "dev/experiments/drake_pusher_slider_v2/dataset_%s/trained_models/perception/dense_descriptors/%s/%s" % (dataset_name, model_name, pth_file))

    return model_file, dataset_name



def main():
    set_cuda_visible_devices([get_freer_gpu()])
    #
    # model_file, dataset_name = get_DD_model_file()

    dataset_name = "box_push_1000_top_down"
    dataset_name = "box_push_1000_angled"
    dataset_paths = exp_utils.get_dataset_paths(dataset_name)
    dataset_root = dataset_paths['dataset_root']
    model_file = dataset_paths['dense_descriptor_model_chkpt']



    multi_episode_dict = DCDrakeSimEpisodeReader.load_dataset(dataset_root, max_num_episodes=None)


    camera_list = [dataset_paths['main_camera_name']]
    target_camera_names = [dataset_paths['main_camera_name']]
    # camera_list = ['camera_1_top_down']
    # target_camera_names = ['camera_1_top_down']

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
                                       target_camera_names=target_camera_names,
                                       verbose=True,
                                       sample_same_episode=False)
    heatmap_vis.run()


if __name__ == "__main__":
    set_cuda_visible_devices([1])
    main()

