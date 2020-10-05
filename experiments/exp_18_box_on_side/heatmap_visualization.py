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

# key_dynam
from key_dynam.experiments.exp_18_box_on_side.utils import get_dataset_paths
from key_dynam.dense_correspondence.dc_drake_sim_episode_reader import DCDrakeSimEpisodeReader


def main():

    model_file = "/home/manuelli/data/key_dynam/dev/experiments/drake_pusher_slider_box_on_side/dataset_dps_box_on_side_600/trained_models/perception/dense_descriptor/3D_loss_camera_angled_2020-05-13-23-39-35-818188/net_best_dy_model.pth"


    dataset_name = "dps_box_on_side_600"
    dataset_paths = get_dataset_paths(dataset_name)
    dataset_root = dataset_paths['dataset_root']
    dataset_name = dataset_paths['dataset_name']
    env_config = dataset_paths['config']

    print("dataset_root", dataset_root)

    multi_episode_dict = DCDrakeSimEpisodeReader.load_dataset(dataset_root)


    camera_list = ["camera_angled"]

    # should really use validation data, but will use train for now . . .
    # will be cross-scene so that shouldn't matter . . . .
    model_config_file = os.path.join(os.path.dirname(model_file), 'config.yaml')
    model_config = getDictFromYamlFilename(model_config_file)
    dataset = DynamicDrakeSimDataset(model_config, multi_episode_dict, phase="train") # could also use train data

    model = torch.load(model_file)
    model = model.cuda()
    model = model.eval()


    heatmap_vis = HeatmapVisualization(model_config,
                                       dataset,
                                       model,
                                       visualize_3D=False,
                                       camera_list=camera_list,
                                       verbose=True,
                                       target_camera_names=['camera_angled', 'camera_angled_rotated'])
    heatmap_vis.run()


if __name__ == "__main__":
    main()

