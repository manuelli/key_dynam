"""
Does all the pre-processing for the vision network

- sample reference descriptors
- compute confidence scores
- select spatially separated descriptors
-
"""
import os
import pydrake
import matplotlib.pyplot as plt
import shutil
import torch
import cv2

from key_dynam.dense_correspondence.precompute_descriptors import compute_descriptor_confidences
from key_dynam.utils.utils import get_project_root, load_yaml, get_current_YYYY_MM_DD_hh_mm_ss_ms, get_data_root, load_pickle
from key_dynam.utils.torch_utils import get_freer_gpu

from key_dynam.dynamics.utils import set_seed

from dense_correspondence_manipulation.utils.utils import set_cuda_visible_devices
from key_dynam.dense_correspondence.dc_drake_sim_episode_reader import DCDrakeSimEpisodeReader
from key_dynam.dense_correspondence.precompute_descriptors import precompute_descriptor_keypoints
from key_dynam.dense_correspondence.keypoint_selection import score_and_select_spatially_separated_keypoints
from key_dynam.experiments.exp_18_box_on_side import utils as exp_18_utils
from key_dynam.experiments.exp_20_mugs import utils as exp_20_utils

# CUDA_VISIBLE_DEVICES = [0]
# set_cuda_visible_devices(CUDA_VISIBLE_DEVICES)

set_cuda_visible_devices([get_freer_gpu()])

from dense_correspondence_manipulation.utils.visualization import draw_reticles


def get_DD_model_file(dataset_name):

    model_name = None
    model_file = None
    if dataset_name == "dps_box_on_side_600":
        model_name = ""
        model_file = "/home/manuelli/data/key_dynam/dev/experiments/drake_pusher_slider_box_on_side/dataset_dps_box_on_side_600/trained_models/perception/dense_descriptor/3D_loss_camera_angled_2020-05-13-23-39-35-818188/net_best_dy_model.pth"
    elif dataset_name == "correlle_mug-small_single_color_600":
        model_name = ""
        model_file = "/home/manuelli/data/key_dynam/dev/experiments/20/dataset_correlle_mug-small_single_color_600/trained_models/perception/dense_descriptors/data_aug_2020-06-03-00-27-50-738970/net_best_model.pth"
    elif dataset_name == "correlle_mug-small_many_colors_600":
        model_name = ""
        model_file = "/home/manuelli/data/key_dynam/dev/experiments/20/dataset_correlle_mug-small_many_colors_600/trained_models/perception/dense_descriptors/data_aug_2020-06-03-16-41-29-740641/net_best_model.pth"
    else:
        raise ValueError("unknown dataset")

    train_dir = os.path.dirname(model_file)
    model_name = os.path.split(train_dir)[-1]

    return model_name, model_file

def main(dataset_name):


    # sample from specific image
    dataset_paths = None
    episode_name = None
    camera_name = None
    episode_idx = None
    if dataset_name == "dps_box_on_side_600":
        camera_name = "camera_angled"
        episode_name = "2020-05-13-21-55-01-487901_idx_33"
        episode_idx = 22
        dataset_paths = exp_18_utils.get_dataset_paths(dataset_name)
    elif dataset_name == "correlle_mug-small_single_color_600":
        camera_name = "camera_1_top_down"
        episode_name = "2020-06-02-14-15-27-898104_idx_56"
        episode_idx = 18
        dataset_paths = exp_20_utils.get_dataset_paths(dataset_name)
    elif dataset_name == "correlle_mug-small_many_colors_600":
        camera_name = "camera_1_top_down"
        episode_name = "2020-06-03-15-48-50-165064_idx_56"
        episode_idx = 20
        dataset_paths = exp_20_utils.get_dataset_paths(dataset_name)
    else:
        raise ValueError("unknown dataset")


    dataset_root = dataset_paths['dataset_root']
    dataset_name = dataset_paths['dataset_name']


    ## Load Model
    model_name, model_file = get_DD_model_file(dataset_name)
    model_train_dir = os.path.dirname(model_file)

    print("model_train_dir", model_train_dir)
    print("model_file", model_file)
    model = torch.load(model_file)
    model = model.cuda()
    model = model.eval()



    multi_episode_dict = DCDrakeSimEpisodeReader.load_dataset(dataset_root, max_num_episodes=None)


    output_dir = os.path.join(model_train_dir,
                              'precomputed_vision_data/descriptor_keypoints/dataset_%s/' % (dataset_name))




    # compute descriptor confidence scores
    if False:
        print("\n\n---------Computing Descriptor Confidence Scores-----------")
        metadata_file = os.path.join(output_dir, 'metadata.p')
        if os.path.isfile(metadata_file):
            answer = input("metadata.p file already exists, do you want to overwrite it? y/n\n")

            if answer == "y":
                shutil.rmtree(output_dir)
                print("removing existing file and continuing")

            else:
                print("aborting")
                quit()

        set_seed(0)




        compute_descriptor_confidences(multi_episode_dict,
                                       model,
                                       output_dir,
                                       batch_size=10,
                                       num_workers=20,
                                       model_file=model_file,
                                       camera_name=camera_name,
                                       num_ref_descriptors=50,
                                       num_batches=10,
                                       episode_name_arg=episode_name,
                                       episode_idx=episode_idx,
                                       )



    if False:
        confidence_score_data_file = os.path.join(output_dir, 'data.p')
        confidence_score_data = load_pickle(confidence_score_data_file)

        metadata_file = os.path.join(output_dir, 'metadata.p')
        metadata = load_pickle(metadata_file)

        print("\n\n---------Selecting Spatially Separated Keypoints-----------")
        score_and_select_spatially_separated_keypoints(metadata,
                                                       confidence_score_data=confidence_score_data,
                                                       K=4,
                                                       position_diff_threshold=15,
                                                       output_dir=output_dir,
                                                       )

    # visualize descriptors
    if False:
        metadata_file = os.path.join(output_dir, 'metadata.p')
        metadata = load_pickle(metadata_file)

        episode_name = metadata['episode_name']
        episode_idx = metadata['episode_idx']
        camera_name = metadata['camera_name']

        episode = multi_episode_dict[episode_name]
        data = episode.get_image_data(camera_name, episode_idx)
        rgb = data['rgb']

        uv = metadata['indices']

        print("uv.shape", uv.shape)

        color = [0,255,0]
        draw_reticles(rgb, uv[:, 0], uv[:, 1], label_color=color)

        save_file = os.path.join(output_dir, 'sampled_descriptors.png')

        plt.figure()
        plt.imshow(rgb)
        plt.savefig(save_file)
        plt.show()

    # visualize spatially separated descriptors
    if False:
        metadata_file = os.path.join(output_dir, 'metadata.p')
        metadata = load_pickle(metadata_file)

        spatial_descriptor_file = os.path.join(output_dir, 'spatial_descriptors.p')
        spatial_descriptors_data = load_pickle(spatial_descriptor_file)
        des_idx = spatial_descriptors_data['spatial_descriptors_idx']

        episode_name = metadata['episode_name']
        episode_idx = metadata['episode_idx']
        camera_name = metadata['camera_name']

        episode = multi_episode_dict[episode_name]
        data = episode.get_image_data(camera_name, episode_idx)
        rgb = data['rgb']

        uv = metadata['indices']

        print("uv.shape", uv.shape)

        color = [0, 255, 0]
        draw_reticles(rgb, uv[des_idx, 0], uv[des_idx, 1], label_color=color)

        save_file = os.path.join(output_dir, 'spatially_separated_descriptors.png')

        plt.figure()
        plt.imshow(rgb)
        plt.savefig(save_file)
        plt.show()



    if True:
        metadata_file = os.path.join(output_dir, 'metadata.p')
        metadata = load_pickle(metadata_file)

        # metadata_file = "/media/hdd/data/key_dynam/dev/experiments/09/precomputed_vision_data/dataset_2020-03-25-19-57-26-556093_constant_velocity_500/model_name_2020-04-07-14-31-35-804270_T_aug_dataset/2020-04-09-20-51-50-624799/metadata.p"
        # metadata = load_pickle(metadata_file)

        print("\n\n---------Precomputing Descriptor Keypoints-----------")
        descriptor_keypoints_output_dir = os.path.join(output_dir, "descriptor_keypoints")
        precompute_descriptor_keypoints(multi_episode_dict,
                                        model,
                                        descriptor_keypoints_output_dir,
                                        ref_descriptors_metadata=metadata,
                                        batch_size=8,
                                        num_workers=20,
                                        camera_names=[camera_name]
                                        )



    print("Data saved at: ", output_dir)
    print("Finished Normally")

if __name__ == "__main__":
    dataset_name = "dps_box_on_side_600"
    # dataset_name = "correlle_mug-small_many_colors_600"
    main(dataset_name)