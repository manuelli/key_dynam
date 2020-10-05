"""
Does all the pre-processing for the vision network

- sample reference descriptors
- compute confidence scores
- select spatially separated descriptors
-
"""
import os
import pydrake

import torch

from key_dynam.dense_correspondence.precompute_descriptors import compute_descriptor_confidences
from key_dynam.utils.utils import get_project_root, load_yaml, get_current_YYYY_MM_DD_hh_mm_ss_ms, get_data_root, load_pickle

from dense_correspondence_manipulation.utils.utils import set_cuda_visible_devices
from key_dynam.dense_correspondence.dc_drake_sim_episode_reader import DCDrakeSimEpisodeReader
from key_dynam.dense_correspondence.precompute_descriptors import precompute_descriptor_keypoints
from key_dynam.dense_correspondence.keypoint_selection import score_and_select_spatially_separated_keypoints

CUDA_VISIBLE_DEVICES = [0]
set_cuda_visible_devices(CUDA_VISIBLE_DEVICES)


def load_episodes():
    # DATASET_NAME = "2020-03-25-19-57-26-556093_constant_velocity_500"
    # DATASET_NAME = "2020-04-15-21-15-56-602712_T_aug_random_velocity_500"
    DATASET_NAME = "2020-04-20-14-58-21-418302_T_aug_random_velocity_1000"
    dataset_root = os.path.join(get_data_root(), "dev/experiments/09/data", DATASET_NAME)


    max_num_episodes = None
    multi_episode_dict = DCDrakeSimEpisodeReader.load_dataset(dataset_root,
                                                              max_num_episodes=max_num_episodes)

    return DATASET_NAME, multi_episode_dict


def get_DD_model_file():
    model_name = "2020-04-07-14-31-35-804270_T_aug_dataset"
    model_file = os.path.join(get_data_root(),
                              "dev/experiments/09/trained_models/dense_descriptors/%s/net_best_dy_model.pth" %(model_name))


    return model_name, model_file


def main():
    dataset_name, multi_episode_dict = load_episodes()

    ## Load Model
    model_name, model_file = get_DD_model_file()
    model = torch.load(model_file)
    model = model.cuda()
    model = model.eval()

    # make this unique
    output_dir = os.path.join(get_data_root(), "dev/experiments/09/precomputed_vision_data", "dataset_%s" %(dataset_name),
                              "model_name_%s" %(model_name), get_current_YYYY_MM_DD_hh_mm_ss_ms())


    camera_name = "camera_1_top_down"
    episode_name = "2020-05-13-21-55-01-487901_idx_33"
    episode_idx = 22

    # compute descriptor confidence scores
    if True:
        print("\n\n---------Computing Descriptor Confidence Scores-----------")
        metadata_file = os.path.join(output_dir, 'metadata.p')
        if os.path.isfile(metadata_file):
            answer = input("metadata.p file already exists, do you want to overwrite it? y/n")

            if answer == "y":
                os.rmdir(output_dir)
                print("removing existing file and continuing")

            else:
                print("aborting")
                quit()

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
                                        batch_size=10,
                                        num_workers=20,
                                        )



    if True:
        confidence_score_data_file = os.path.join(output_dir, 'data.p')
        confidence_score_data = load_pickle(confidence_score_data_file)
        print("\n\n---------Selecting Spatially Separated Keypoints-----------")
        score_and_select_spatially_separated_keypoints(metadata,
                                                       confidence_score_data=confidence_score_data,
                                                       K=5,
                                                       position_diff_threshold=25,
                                                       output_dir=output_dir,
                                                       )




    print("Data saved at: ", output_dir)
    print("Finished Normally")

if __name__ == "__main__":
    main()