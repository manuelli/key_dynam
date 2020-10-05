import os
import copy
import numpy as np
import time

import pydrake

import torch

# dense_correspondence
from dense_correspondence_manipulation.utils import meshcat_utils as pdc_meshcat_utils
from dense_correspondence_manipulation.utils.constants import DEPTH_IM_SCALE

# key_dynam
from key_dynam.utils.utils import get_project_root, load_yaml, get_current_YYYY_MM_DD_hh_mm_ss_ms, get_data_root, \
    load_pickle, save_pickle
from key_dynam.models.model_builder import build_dynamics_model
from key_dynam.utils.torch_utils import get_freer_gpu, set_cuda_visible_devices
from key_dynam.utils import torch_utils
from key_dynam.dataset.dynamic_spartan_episode_reader import DynamicSpartanEpisodeReader
from key_dynam.dataset.function_factory import ObservationFunctionFactory, ActionFunctionFactory
from key_dynam.dataset.vision_function_factory import PrecomputedVisualObservationFunctionFactory, VisualObservationFunctionFactory
from key_dynam.experiments.exp_22_push_box_hardware.utils import get_dataset_paths
from key_dynam.experiments.drake_pusher_slider import DD_utils
from key_dynam.dataset.episode_dataset import MultiEpisodeDataset
from key_dynam.utils import meshcat_utils
from key_dynam.dynamics.models_dy import rollout_model, get_object_and_robot_state_indices
from key_dynam.planner.planners import RandomShootingPlanner, PlannerMPPI
import key_dynam.planner.utils as planner_utils
from key_dynam.dataset.mpc_dataset import DynamicsModelInputBuilder
from key_dynam.dataset.online_episode_reader import OnlineEpisodeReader
from key_dynam.dynamics.utils import set_seed

PLANNER_TYPE = "mppi"

SEED = 0
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

    # build dense-descriptor model
    model_dd_file = metadata['model_file']
    model_dd = torch.load(model_dd_file)
    model_dd = model_dd.eval()
    model_dd = model_dd.cuda()


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
                                  phase="valid",  # this means no data augmentation
                                  )

    #### PLANNER #######
    planner = None
    # make a planner config
    planner_config = copy.copy(model_dy.config)
    config_tmp = load_yaml(os.path.join(get_project_root(), 'experiments/exp_22_push_box_hardware/config_DD_3D.yaml'))
    planner_config['mpc'] = config_tmp['mpc']
    if PLANNER_TYPE == "random_shooting":
        planner = RandomShootingPlanner(planner_config)
    elif PLANNER_TYPE == "mppi":
        planner = PlannerMPPI(planner_config)
    else:
        raise ValueError("unknown planner type: %s" % (PLANNER_TYPE))



    return {"model_dy": model_dy,
            'model_dd': model_dd,
            'dataset': dataset,
            'config': config,
            "multi_episode_dict": multi_episode_dict,
            'spatial_descriptor_data': spatial_descriptor_data,
            'planner': planner,
            'observation_function': observation_function,
            'action_function': action_function,
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

    depth = image_data['depth_int16'] / DEPTH_IM_SCALE

    # pointcloud
    name = "pointclouds/%d" % (display_idx)
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
    name = "observations/%d" % (display_idx)
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
        name = "keypoints/%d" % (display_idx)
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
                                               z_pred,  # [z_dim]
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

    name = "z_object/%d" % (display_idx)
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


def add_images_to_episode_data(episode,
                               episode_idx,
                               camera_name):

    data = copy.deepcopy(episode.get_data(episode_idx))
    # add image info
    image_episode_idx = episode.image_episode_idx_from_query_idx(episode_idx)
    image_data = episode.image_episode.get_image_data(camera_name, image_episode_idx)

    depth_int16 = image_data['depth_int16'].astype(np.uint16)
    data['observations']['images'][camera_name]['rgb'] = image_data['rgb']
    data['observations']['images'][camera_name]['depth_int16'] = depth_int16
    data['observations']['images'][camera_name]['depth_16U'] = depth_int16

    # for compatibility with sim datasets
    data['observation'] = data['observations']
    data['action'] = data['actions']

    return data

def main():
    d = load_model_and_data()
    model_dy = d['model_dy']
    dataset = d['dataset']
    config = d['config']
    multi_episode_dict = d['multi_episode_dict']
    planner = d['planner']
    planner_config = planner.config

    idx_dict = get_object_and_robot_state_indices(config)
    object_indices = idx_dict['object_indices']
    robot_indices = idx_dict['robot_indices']

    n_his = config['train']['n_history']

    # save_dir = os.path.join(get_project_root(),  'sandbox/mpc/', get_current_YYYY_MM_DD_hh_mm_ss_ms())
    save_dir = os.path.join(get_project_root(), 'sandbox/mpc/push_right_box_horizontal')
    print("save_dir", save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # rotate
    # episode_names = dataset.get_episode_names()
    # print("len(episode_names)", len(episode_names))
    # episode_name = episode_names[0]
    # start_idx = 1
    # n_roll = 15

    # # straight + rotate
    # episode_name = "2020-06-29-21-04-16"
    # print('episode_name', episode_name)
    # start_idx = 1
    # n_roll = 15

    # this is a nice straight push . . .
    # push with box in horizontal position
    episode_name = "2020-06-29-22-03-45"
    start_idx = 2
    n_roll = 10

    # # validation set episodes
    # episode_names = dataset.get_episode_names()
    # print("len(episode_names)", len(episode_names))
    # episode_name = episode_names[1]
    # start_idx = 2
    # n_roll = 15

    camera_name = "d415_01"
    episode = multi_episode_dict[episode_name]
    print("episode_name", episode_name)

    vis = meshcat_utils.make_default_visualizer_object()
    vis.delete()

    idx_list = list(range(start_idx, start_idx + n_roll + 1))
    idx_list_GT = idx_list
    goal_idx = idx_list[-1]
    print("idx_list", idx_list)

    # visualize ground truth rollout
    if True:
        for display_idx, episode_idx in enumerate(idx_list):
            visualize_episode_data_single_timestep(vis=vis,
                                                   dataset=dataset,
                                                   episode=episode,
                                                   camera_name=camera_name,
                                                   episode_idx=episode_idx,
                                                   display_idx=episode_idx,
                                                   )

    data_goal = dataset._getitem(episode,
                                 goal_idx,
                                 rollout_length=1,
                                 n_history=1
                                 )
    states_goal = data_goal['observations_combined'][0]
    z_states_goal = model_dy.compute_z_state(states_goal)['z']

    print("states_goal.shape", states_goal.shape)
    print("z_states_goal.shape", z_states_goal.shape)

    ##### VISUALIZE PREDICTED ROLLOUT ##########
    data = dataset._getitem(episode,
                            start_idx,
                            rollout_length=n_roll)



    states = data['observations_combined'].unsqueeze(0)
    z = model_dy.compute_z_state(states)['z']
    actions = data['actions'].unsqueeze(0)
    idx_range_model_dy_input = data['idx_range']

    print("data.keys()", data.keys())
    print("data['idx_range']", data['idx_range'])

    # z_init
    z_init = z[:, :n_his]

    # actions_init
    action_start_idx = 0
    action_end_idx = n_his + n_roll - 1
    action_seq = actions[:, action_start_idx:action_end_idx]

    print("action_seq GT\n", action_seq)

    with torch.no_grad():
        rollout_data = rollout_model(state_init=z_init.cuda(),
                                     action_seq=action_seq.cuda(),
                                     dynamics_net=model_dy,
                                     compute_debug_data=False)

    # [B, n_roll, state_dim]
    # state_rollout_pred = rollout_data['state_pred']
    z_rollout_pred = rollout_data['state_pred'].squeeze()
    print("z_rollout_pred.shape", z_rollout_pred.shape)

    if True:
        for idx in range(len(z_rollout_pred)):
            display_idx = data['idx_range'][idx + n_his]
            visualize_model_prediction_single_timestep(vis,
                                                       config,
                                                       z_pred=z_rollout_pred[idx],
                                                       display_idx=display_idx)

        print("z_rollout_pred.shape", z_rollout_pred.shape)

    # compute loss when rolled out using GT action sequence
    eval_indices = object_indices
    obs_goal = z_states_goal[object_indices].cuda()
    reward_data = planner_utils.evaluate_model_rollout(state_pred=rollout_data['state_pred'],
                                                       obs_goal=obs_goal,
                                                       eval_indices=eval_indices,
                                                       terminal_cost_only=planner_config['mpc']['mppi'][
                                                           'terminal_cost_only'],
                                                       p=planner_config['mpc']['mppi']['cost_norm']
                                                       )

    print("reward_data using action_seq_GT\n", reward_data['reward'])

    ##### MPC ##########
    data = dataset._getitem(episode,
                            start_idx,
                            rollout_length=0,
                            n_history=config['train']['n_history'])

    state_cur = data['observations_combined'].cuda()
    z_state_cur = model_dy.compute_z_state(state_cur)['z']
    action_his = data['actions'][:(n_his - 1)].cuda()

    print("z_state_cur.shape", state_cur.shape)
    print("action_his.shape", action_his.shape)

    # don't seed with nominal actions just yet
    action_seq_rollout_init = None

    set_seed(SEED)
    mpc_out = planner.trajectory_optimization(state_cur=z_state_cur,
                                              action_his=action_his,
                                              obs_goal=obs_goal,
                                              model_dy=model_dy,
                                              action_seq_rollout_init=action_seq_rollout_init,
                                              n_look_ahead=n_roll,
                                              eval_indices=object_indices,
                                              rollout_best_action_sequence=True,
                                              verbose=True,
                                              add_grid_action_samples=True,
                                              )

    print("\n\n------MPC output-------\n\n")
    print("action_seq:\n", mpc_out['action_seq'])
    mpc_state_pred = mpc_out['state_pred']

    # current shape is [n_roll + 1, state_dim] but really should be
    # [n_roll, state_dim] . . . something is  up
    print("mpc_state_pred.shape", mpc_state_pred.shape)
    print("mpc_out['action_seq'].shape", mpc_out['action_seq'].shape)
    print("n_roll", n_roll)

    # visualize
    for idx in range(n_roll):
        episode_idx = start_idx + idx + 1
        visualize_model_prediction_single_timestep(vis,
                                                   config,
                                                   z_pred=mpc_state_pred[idx],
                                                   display_idx=episode_idx,
                                                   name_prefix="mpc",
                                                   color=[255, 0, 0])

    ######## MPC w/ dynamics model input builder #############
    print("\n\n-----DynamicsModelInputBuilder-----")

    # dynamics model input builder
    online_episode = OnlineEpisodeReader(no_copy=True)

    ref_descriptors = d['spatial_descriptor_data']['spatial_descriptors']
    ref_descriptors = torch_utils.cast_to_torch(ref_descriptors).cuda()
    K_matrix = episode.image_episode.camera_K_matrix(camera_name)
    T_world_camera = episode.image_episode.camera_pose(camera_name, 0)
    visual_observation_function = \
        VisualObservationFunctionFactory.descriptor_keypoints_3D(config=config,
                                                                 camera_name=camera_name,
                                                                 model_dd=d['model_dd'],
                                                                 ref_descriptors=ref_descriptors,
                                                                 K_matrix=K_matrix,
                                                                 T_world_camera=T_world_camera,
                                                                 )

    input_builder = DynamicsModelInputBuilder(observation_function=d['observation_function'],
                                              visual_observation_function=visual_observation_function,
                                              action_function=d['action_function'],
                                              episode=online_episode)


    compute_control_action_msg = dict()
    compute_control_action_msg['type'] = "COMPUTE_CONTROL_ACTION"

    for i in range(n_his):
        episode_idx = idx_range_model_dy_input[i]
        print("episode_idx", episode_idx)


        # add image information to
        data = add_images_to_episode_data(episode,
                                          episode_idx,
                                          camera_name)


        online_episode.add_data(copy.deepcopy(data))
        compute_control_action_msg['data'] = data

        # hack for seeing how much the history matters .. .
        # online_episode.add_data(copy.deepcopy(data))


    # save informatin for running zmq controller
    save_pickle(compute_control_action_msg, os.path.join(save_dir, 'compute_control_action_msg.p'))
    goal_idx = idx_list_GT[-1]
    goal_data = add_images_to_episode_data(episode,
                                           goal_idx,
                                           camera_name)
    goal_data['observations']['timestamp_system'] = time.time()
    plan_msg = {'type': "PLAN",
                'data': [goal_data],
                'n_roll': n_roll,
                'K_matrix': K_matrix,
                'T_world_camera': T_world_camera,
                }
    save_pickle(plan_msg, os.path.join(save_dir, "plan_msg.p"))


    print("len(online_episode)", len(online_episode))

    # use this to construct input
    # verify it's the same as what we got from using the dataset directly
    idx = online_episode.get_latest_idx()
    mpc_input_data = input_builder.get_dynamics_model_input(idx,
                                                            n_history=n_his)

    # print("mpc_input_data\n", mpc_input_data)
    state_cur_ib = mpc_input_data['states'].cuda()
    action_his_ib = mpc_input_data['actions'].cuda()

    z_state_cur_ib = model_dy.compute_z_state(state_cur_ib)['z']

    set_seed(SEED)
    mpc_out = planner.trajectory_optimization(state_cur=z_state_cur_ib,
                                              action_his=action_his_ib,
                                              obs_goal=obs_goal,
                                              model_dy=model_dy,
                                              action_seq_rollout_init=None,
                                              n_look_ahead=n_roll,
                                              eval_indices=object_indices,
                                              rollout_best_action_sequence=True,
                                              verbose=True,
                                              add_grid_action_samples=True,
                                              )

    # visualize
    for idx in range(n_roll):
        episode_idx = start_idx + idx + 1
        visualize_model_prediction_single_timestep(vis,
                                                   config,
                                                   z_pred=mpc_out['state_pred'][idx],
                                                   display_idx=episode_idx,
                                                   name_prefix="mpc_input_builder",
                                                   color=[255, 255, 0])


    # now try doing the MPC


    # print("state_cur_ib.shape", state_cur_ib.shape)
    # print("action_his_ib.shape", action_his_ib.shape)
    #
    # state_cur_delta = state_cur_ib - state_cur
    # print("state_cur_delta", state_cur_delta)
    # print("state_cur_delta[0][robot_indices]", state_cur_delta[0][robot_indices])
    #
    #
    # print("state_cur_ib[0][robot_indices]", state_cur_ib[0][robot_indices])
    # print("state_cur[0][robot_indices]", state_cur[0][robot_indices])
    # print("torch.norm(state_cur_delta)", torch.norm(state_cur_delta))
    #
    # action_his_delta = action_his_ib - action_his
    # print("action_his_ib", action_his_ib)
    # print("action_his", action_his)
    # print("action_his_delta", action_his_ib - action_his)




if __name__ == "__main__":
    set_cuda_visible_devices([get_freer_gpu()])
    main()
