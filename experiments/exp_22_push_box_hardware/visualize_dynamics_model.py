import os

import pydrake

import torch

# dense_correspondence
from dense_correspondence_manipulation.utils import meshcat_utils as pdc_meshcat_utils
from dense_correspondence_manipulation.utils.constants import DEPTH_IM_SCALE

# key_dynam
from key_dynam.utils.utils import get_project_root, load_yaml, get_current_YYYY_MM_DD_hh_mm_ss_ms, get_data_root, \
    load_pickle
from key_dynam.models.model_builder import build_dynamics_model
from key_dynam.utils.torch_utils import get_freer_gpu, set_cuda_visible_devices
from key_dynam.utils import torch_utils
from key_dynam.dataset.dynamic_spartan_episode_reader import DynamicSpartanEpisodeReader
from key_dynam.dataset.function_factory import ObservationFunctionFactory, ActionFunctionFactory
from key_dynam.dataset.vision_function_factory import PrecomputedVisualObservationFunctionFactory
from key_dynam.experiments.exp_22_push_box_hardware.utils import get_dataset_paths
from key_dynam.experiments.drake_pusher_slider import DD_utils
from key_dynam.dataset.episode_dataset import MultiEpisodeDataset
from key_dynam.utils import meshcat_utils
from key_dynam.dynamics.models_dy import rollout_model, get_object_and_robot_state_indices



def load_model_and_data():

    dataset_name = "push_box_hardware"

    # model_name = "DD_2D/2020-06-24-22-22-58-234812_DD_3D_n_his_2" # this model is actually 3D
    # model_name = "DD_3D/2020-06-25-00-49-29-679042_DD_3D_n_his_2_T_aug"
    # model_name = "DD_3D/2020-06-25-00-39-29-020621_DD_3D_n_his_2"

    model_name = "DD_3D/2020-07-02-17-59-21-362337_DD_3D_n_his_2_T_aug"
    train_dir = "/home/manuelli/data/key_dynam/dev/experiments/22/dataset_push_box_hardware/trained_models/dynamics"

    train_dir = os.path.join(train_dir, model_name)
    ckpt_file = os.path.join(train_dir, "net_best_dy_state_dict.pth")

    config = load_yaml(os.path.join(train_dir, 'config.yaml'))
    state_dict = torch.load(ckpt_file)

    # build dynamics model
    model_dy = build_dynamics_model(config)
    # print("state_dict.keys()", state_dict.keys())
    model_dy.load_state_dict(state_dict)
    model_dy = model_dy.eval()
    model_dy = model_dy.cuda()

    spatial_descriptor_data = load_pickle(os.path.join(train_dir, 'spatial_descriptors.p'))
    metadata = load_pickle(os.path.join(train_dir, 'metadata.p'))

    # load the dataset
    dataset_paths = get_dataset_paths(dataset_name)
    dataset_root = dataset_paths['dataset_root']
    episodes_config = dataset_paths['episodes_config']


    precomputed_vision_data_root = DD_utils.get_precomputed_data_root(dataset_name)['precomputed_data_root']

    # descriptor_keypoints_root = os.path.join(precomputed_vision_data_root, 'descriptor_keypoints')
    descriptor_keypoints_root = os.path.join(precomputed_vision_data_root, 'descriptor_keypoints')


    multi_episode_dict = DynamicSpartanEpisodeReader.load_dataset(config=config,
                                                                  episodes_config=episodes_config,
                                                                  episodes_root=dataset_paths['dataset_root'],
                                                                  load_image_episode=True,
                                                                  precomputed_data_root=descriptor_keypoints_root,
                                                                  max_num_episodes=None)



    visual_observation_function = PrecomputedVisualObservationFunctionFactory.function_from_config(config,
                                                                                                   keypoint_idx=
                                                                                                   spatial_descriptor_data[
                                                                                                       'spatial_descriptors_idx'])


    action_function = ActionFunctionFactory.function_from_config(config)
    observation_function = ObservationFunctionFactory.function_from_config(config)
    dataset = MultiEpisodeDataset(config,
                                  action_function=action_function,
                                  observation_function=observation_function,
                                  episodes=multi_episode_dict,
                                  visual_observation_function=visual_observation_function,
                                  phase="valid", # this means no data augmentation
                                  )


    return {"model_dy": model_dy,
            'dataset': dataset,
            'config': config,
            "multi_episode_dict": multi_episode_dict,
            'spatial_descriptor_data': spatial_descriptor_data,
            }

def visualize_episode_data_single_timestep(vis,
                                           dataset,
                                           camera_name,
                                           episode,
                                           episode_idx,
                                           display_idx,
                                           ):

    image_episode_idx = episode.image_episode_idx_from_query_idx(episode_idx)
    image_data = episode.image_episode.get_image_data(camera_name=camera_name, idx=image_episode_idx)

    depth = image_data['depth_int16']/DEPTH_IM_SCALE

    # pointcloud
    name = "pointclouds/%d" %(display_idx)
    pdc_meshcat_utils.visualize_pointcloud(vis,
                                           name,
                                           depth=depth,
                                           K=image_data['K'],
                                           rgb=image_data['rgb'],
                                           T_world_camera=image_data['T_world_camera'])


    data = dataset._getitem(episode,
                            episode_idx,
                            n_history=1,
                            rollout_length=0)
    # action position
    # name = "actions/%d" %(display_idx)
    # actions = torch_utils.cast_to_numpy(data['actions'])
    #
    # meshcat_utils.visualize_points(vis,
    #                                name,
    #                                actions,
    #                                color=[255,0,0],
    #                                size=0.01,
    #                                )

    # observation position
    name = "observations/%d" %(display_idx)
    observations = torch_utils.cast_to_numpy(data['observations'])
    # print("observations.shape", observations.shape)
    # print("observations", observations)
    meshcat_utils.visualize_points(vis,
                                   name,
                                   observations,
                                   color=[0, 255, 0],
                                   size=0.01,
                                   )


    # keypoints
    if True:
        name = "keypoints/%d" %(display_idx)
        keypoints = torch_utils.cast_to_numpy(data['visual_observation_func_collated']['keypoints_3d'][0])
        # print("keypoints.shape", keypoints.shape)
        meshcat_utils.visualize_points(vis,
                                       name,
                                       keypoints,
                                       color=[0, 255, 0],
                                       size=0.01,
                                       )
        # print("keypoints.shape", keypoints.shape)

def visualize_model_prediction_single_timestep(vis,
                                               config,
                                               z_pred, # [z_dim]
                                               display_idx,
                                               name_prefix=None,
                                               color=None,
                                               display_robot_state=True,
                                               ):

    if color is None:
        color = [0, 0, 255]


    idx_dict = get_object_and_robot_state_indices(config)


    z_object = z_pred[idx_dict['object_indices']].reshape(config['dataset']['object_state_shape'])
    z_robot = z_pred[idx_dict['robot_indices']].reshape(config['dataset']['robot_state_shape'])


    name = "z_object/%d" %(display_idx)
    if name_prefix is not None:
        name = name_prefix + "/" + name
    meshcat_utils.visualize_points(vis,
                                   name,
                                   torch_utils.cast_to_numpy(z_object),
                                   color=color,
                                   size=0.01,
                                   )


    if display_robot_state:
        name = "z_robot/%d" % (display_idx)
        if name_prefix is not None:
            name = name_prefix + "/" + name
        meshcat_utils.visualize_points(vis,
                                       name,
                                       torch_utils.cast_to_numpy(z_robot),
                                       color=color,
                                       size=0.01,
                                       )

    # print("z_object.shape", z_object.shape)
    # print("z_robot.shape", z_robot.shape)


def main():
    d = load_model_and_data()
    model_dy = d['model_dy']
    dataset = d['dataset']
    config = d['config']
    multi_episode_dict = d['multi_episode_dict']


    n_his = config['train']['n_history']




    # rotate
    # episode_names = dataset.get_episode_names()
    # print("len(episode_names)", len(episode_names))
    # episode_name = episode_names[0]
    # start_idx = 1
    # n_roll = 15
    # n_his = 2


    # # straight + rotate
    # episode_name = "2020-06-29-21-04-16"
    # print('episode_name', episode_name)
    # start_idx = 1
    # n_roll = 15
    # n_his = 2

    # this is a nice straight push . . .
    # push with box in horizontal position
    episode_name = "2020-06-29-22-03-45"
    start_idx = 2
    n_roll = 15
    n_his = 2


    camera_name = "d415_01"
    episode = multi_episode_dict[episode_name]
    print("len(episode)", len(episode))

    vis = meshcat_utils.make_default_visualizer_object()
    vis.delete()


    idx_list = range(start_idx, start_idx + n_roll + n_his)


    # visualize ground truth rollout
    if True:
        for display_idx, episode_idx in enumerate(idx_list):
            visualize_episode_data_single_timestep(vis=vis,
                                                   dataset=dataset,
                                                   episode=episode,
                                                   camera_name=camera_name,
                                                   episode_idx=episode_idx,
                                                   display_idx=display_idx,
                                                   )



    ##### VISUALIZE PREDICTED ROLLOUT ##########
    data = dataset._getitem(episode,
                            start_idx,
                            rollout_length=n_roll)

    states = data['observations_combined'].unsqueeze(0)
    z = model_dy.compute_z_state(states)['z']
    actions = data['actions'].unsqueeze(0)




    # z_init
    z_init = z[:, :n_his]

    # actions_init
    action_start_idx = 0
    action_end_idx = n_his + n_roll - 1
    action_seq = actions[:, action_start_idx:action_end_idx]


    with torch.no_grad():
        rollout_data = rollout_model(state_init=z_init.cuda(),
                                     action_seq=action_seq.cuda(),
                                     dynamics_net=model_dy,
                                     compute_debug_data=False)

    # [B, n_roll, state_dim]
    # state_rollout_pred = rollout_data['state_pred']
    z_rollout_pred = rollout_data['state_pred'].squeeze()

    if True:
        for idx in range(len(z_rollout_pred)):
            visualize_model_prediction_single_timestep(vis,
                                                       config,
                                                       z_pred=z_rollout_pred[idx],
                                                       display_idx=idx+1)

        print("z_rollout_pred.shape", z_rollout_pred.shape)





if __name__ == "__main__":
    set_cuda_visible_devices([get_freer_gpu()])
    main()

