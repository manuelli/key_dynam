import os
import numpy as np
import pydrake
import torch
import copy as copy

from key_dynam.dataset.function_factory import ObservationFunctionFactory, ActionFunctionFactory, \
    slider_pose_from_observation
from key_dynam.dataset.vision_function_factory import VisualObservationFunctionFactory

from key_dynam.utils.utils import get_project_root, load_yaml, get_current_YYYY_MM_DD_hh_mm_ss_ms, get_data_root, \
    load_pickle
from key_dynam.training.train_dynamics_pusher_slider_precomputed_keypoints_old import train_dynamics
from key_dynam.envs.drake_pusher_slider import DrakePusherSliderEnv
from key_dynam.envs.drake_mugs import DrakeMugsEnv
from key_dynam.dataset.online_episode_reader import OnlineEpisodeReader
from key_dynam.dataset.mpc_dataset import DynamicsModelInputBuilder
from key_dynam.planner.planners import RandomShootingPlanner, PlannerMPPI
from key_dynam.eval import mpc_eval_drake_pusher_slider
from key_dynam.dynamics.models_dy import get_object_and_robot_state_indices
from key_dynam.models import model_builder

"""
Helper code to evaluate dense-descriptor models on closed-loop MPC
performance
"""


def get_precomputed_data_root(dataset_name):
    precomputed_vision_data_root = None

    raise ValueError("unknown dataset:", dataset_name)

    return {'precomputed_data_root': precomputed_vision_data_root}


def load_model(model_train_dir,
               strict=True):
    """
    Helper function to load dynamics model and vision model
    """

    model_dy_dict = model_builder.load_dynamics_model_from_folder(model_train_dir, strict=strict)
    model_dy = model_dy_dict['model_dy']
    model_dy = model_dy.eval()
    model_dy = model_dy.cuda()
    model_name = model_dy_dict['model_name']

    # load dense descriptor model
    metadata = load_pickle(os.path.join(model_train_dir, 'metadata.p'))
    model_dd_file = metadata['model_file']
    model_dd = torch.load(model_dd_file)
    model_dd = model_dd.eval()
    model_dd = model_dd.cuda()

    spatial_descriptor_data = load_pickle(os.path.join(model_train_dir, 'spatial_descriptors.p'))

    return {'model_dy': model_dy,
            'model_dd': model_dd,
            'spatial_descriptor_data': spatial_descriptor_data,
            'metadata': metadata,
            'model_name': model_name,
            }


def evaluate_mpc(model_dir,
                 config_planner_mpc=None,
                 save_dir=None,
                 planner_type=None,
                 env_config=None,
                 strict=True,
                 generate_initial_condition_func=None,
                 ):

    assert save_dir is not None
    assert planner_type is not None
    assert env_config is not None
    assert generate_initial_condition_func is not None

    model_data = load_model(model_dir, strict=strict)
    model_dy = model_data['model_dy']
    model_config = model_dy.config
    config = model_config

    model_dd = model_data['model_dd']

    # create the environment
    env = DrakePusherSliderEnv(env_config, visualize=False)
    env.reset()

    camera_name = model_data['metadata']['camera_name']

    # sanity check
    camera_name_in_training = model_config['dataset']['visual_observation_function']['camera_name']
    assert camera_name == camera_name_in_training, "camera_names don't match: camera_name = %s, camera_name_in_trainig = %s" % (
        camera_name, camera_name_in_training)

    T_world_camera = env.camera_pose(camera_name)
    camera_K_matrix = env.camera_K_matrix(camera_name)
    spatial_descriptor_data = model_data['spatial_descriptor_data']
    ref_descriptors = torch.Tensor(spatial_descriptor_data['spatial_descriptors']).cuda()
    K = ref_descriptors.shape[0]

    action_function = ActionFunctionFactory.function_from_config(config)
    observation_function = ObservationFunctionFactory.drake_pusher_position_3D(config)
    visual_observation_function = \
        VisualObservationFunctionFactory.function_from_config(config,
                                                              camera_name=camera_name,
                                                              model_dd=model_dd,
                                                              ref_descriptors=ref_descriptors,
                                                              K_matrix=camera_K_matrix,
                                                              T_world_camera=T_world_camera,
                                                              )

    episode = OnlineEpisodeReader()

    mpc_input_builder = DynamicsModelInputBuilder(observation_function=observation_function,
                                                  visual_observation_function=visual_observation_function,
                                                  action_function=action_function,
                                                  episode=episode)

    def goal_func(obs_local):
        keypoints_dict = visual_observation_function(obs_local)
        return keypoints_dict['tensor']

    eval_indices = np.arange(3 * K)  # extract the keypoints

    # make a planner config, same as model config but with mpc and eval sections
    # replaced
    planner_config = copy.copy(model_config)
    if config_planner_mpc is not None:
        planner_config['mpc'] = config_planner_mpc['mpc']
        planner_config['eval'] = config_planner_mpc['eval']

    planner = None
    if planner_type == "random_shooting":
        planner = RandomShootingPlanner(planner_config)
    elif planner_type == "mppi":
        planner = PlannerMPPI(planner_config)
    else:
        raise ValueError("unknown planner type: %s" % (planner_type))

    # run a single iteration
    mpc_eval_drake_pusher_slider.evaluate_mpc(model_dy=model_dy,
                                              env=env,
                                              episode=episode,
                                              mpc_input_builder=mpc_input_builder,
                                              planner=planner,
                                              eval_indices=eval_indices,
                                              goal_func=goal_func,
                                              config=planner_config,
                                              wait_for_user_input=False,
                                              save_dir=save_dir,
                                              model_name="test",
                                              experiment_name="test",
                                              generate_initial_condition_func=generate_initial_condition_func
                                              )

    return {'save_dir': save_dir}


def evaluate_mpc_z_state(model_dir,
                         config_planner_mpc=None,
                         save_dir=None,
                         planner_type=None,
                         env_config=None,
                         strict=True,
                         generate_initial_condition_func=None,
                         env_type="DrakePusherSliderEnv"
                         ):
    assert save_dir is not None
    assert planner_type is not None
    assert env_config is not None
    assert generate_initial_condition_func is not None

    model_data = load_model(model_dir, strict=strict)
    model_dy = model_data['model_dy']
    model_config = model_dy.config
    config = model_config

    model_dd = model_data['model_dd']

    # create the environment
    env = None
    if env_type == "DrakePusherSliderEnv":
        env = DrakePusherSliderEnv(env_config, visualize=False)
    elif env_type == "DrakeMugsEnv":
        env = DrakeMugsEnv(env_config, visualize=False)
    else:
        raise ValueError("unknown env type: %s" % (env_type))
    env.reset()

    camera_name = model_data['metadata']['camera_name']

    # sanity check
    camera_name_in_training = model_config['dataset']['visual_observation_function']['camera_name']
    assert camera_name == camera_name_in_training, "camera_names don't match: camera_name = %s, camera_name_in_trainig = %s" % (
        camera_name, camera_name_in_training)

    T_world_camera = env.camera_pose(camera_name)
    camera_K_matrix = env.camera_K_matrix(camera_name)
    spatial_descriptor_data = model_data['spatial_descriptor_data']
    ref_descriptors = torch.Tensor(spatial_descriptor_data['spatial_descriptors']).cuda()
    K = ref_descriptors.shape[0]

    action_function = ActionFunctionFactory.function_from_config(config)
    observation_function = ObservationFunctionFactory.drake_pusher_position_3D(config)
    visual_observation_function = \
        VisualObservationFunctionFactory.function_from_config(config,
                                                              camera_name=camera_name,
                                                              model_dd=model_dd,
                                                              ref_descriptors=ref_descriptors,
                                                              K_matrix=camera_K_matrix,
                                                              T_world_camera=T_world_camera,
                                                              )

    episode = OnlineEpisodeReader()

    mpc_input_builder = DynamicsModelInputBuilder(observation_function=observation_function,
                                                  visual_observation_function=visual_observation_function,
                                                  action_function=action_function,
                                                  episode=episode)

    def goal_func(obs_tmp):
        state_tmp = mpc_input_builder.get_state_input_single_timestep({'observation': obs_tmp})['state']
        # z_dict= model_dy.compute_z_state(state_tmp.unsqueeze(0))
        # print("z_dict['z_object'].shape", z_dict['z_object'].shape)
        return model_dy.compute_z_state(state_tmp.unsqueeze(0))['z_object_flat']

    index_dict = get_object_and_robot_state_indices(model_config)
    object_indices = index_dict['object_indices']

    # make a planner config, same as model config but with mpc and eval sections
    # replaced
    planner_config = copy.copy(model_config)
    if config_planner_mpc is not None:
        planner_config['mpc'] = config_planner_mpc['mpc']
        planner_config['eval'] = config_planner_mpc['eval']

    planner = None
    if planner_type == "random_shooting":
        planner = RandomShootingPlanner(planner_config)
    elif planner_type == "mppi":
        planner = PlannerMPPI(planner_config)
    else:
        raise ValueError("unknown planner type: %s" % (planner_type))

    # run a single iteration
    mpc_eval_drake_pusher_slider.evaluate_mpc(model_dy=model_dy,
                                              env=env,
                                              episode=episode,
                                              mpc_input_builder=mpc_input_builder,
                                              planner=planner,
                                              eval_indices=object_indices,
                                              goal_func=goal_func,
                                              config=planner_config,
                                              wait_for_user_input=False,
                                              save_dir=save_dir,
                                              model_name="test",
                                              experiment_name="test",
                                              generate_initial_condition_func=generate_initial_condition_func,
                                              )

    return {'save_dir': save_dir}
