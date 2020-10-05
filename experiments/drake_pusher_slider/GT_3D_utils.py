import os
import numpy as np
import pydrake
import torch
import copy as copy
import functools

from key_dynam.dataset.function_factory import ObservationFunctionFactory, ActionFunctionFactory, \
    slider_pose_from_observation

from key_dynam.utils.utils import get_project_root, load_yaml, get_current_YYYY_MM_DD_hh_mm_ss_ms, get_data_root, \
    load_pickle
from key_dynam.training.train_dynamics_pusher_slider_precomputed_keypoints_old import train_dynamics
from key_dynam.envs.drake_pusher_slider import DrakePusherSliderEnv
from key_dynam.dataset.online_episode_reader import OnlineEpisodeReader
from key_dynam.dataset.mpc_dataset import DynamicsModelInputBuilder
from key_dynam.planner.planners import RandomShootingPlanner, PlannerMPPI
from key_dynam.eval import mpc_eval_drake_pusher_slider
from key_dynam.dataset.drake_sim_episode_reader import DrakeSimEpisodeReader
from key_dynam.models import model_builder


"""
Helper code to evaluate GT_3D models on closed-loop MPC performance
"""


def goal_state_from_observation(obs,
                                observation_function,
                                eval_indices):
    """
    Helper function for getting the goal state from an observation
    """
    return observation_function(obs)[eval_indices]



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

    model_data = model_builder.load_dynamics_model_from_folder(model_dir, strict=strict)
    model_dy = model_data['model_dy']
    model_dy = model_dy.eval()
    model_dy = model_dy.cuda()
    model_config = model_dy.config




    # create the environment
    env = DrakePusherSliderEnv(env_config, visualize=False)
    env.reset()

    observation_function = ObservationFunctionFactory.function_from_config(model_config)
    action_function = ActionFunctionFactory.function_from_config(model_config)
    episode = OnlineEpisodeReader()

    mpc_input_builder = DynamicsModelInputBuilder(observation_function=observation_function,
                                                  action_function=action_function,
                                                  visual_observation_function=None,
                                                  episode=episode)

    pts_GT = np.array(model_config['dataset']['observation_function']['GT_3D_object_points'])
    K = pts_GT.shape[0]  # num keypoints
    eval_indices = np.arange(3 * K)  # extract the keypoints

    def goal_func(obs_local):
        """
        Helper function for getting the goal state from an observation
        """
        return observation_function(obs_local)[eval_indices]


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
        raise ValueError("unknown planner type: %s" %(planner_type))

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
                                              generate_initial_condition_func=generate_initial_condition_func,
                                              )

    return {'save_dir': save_dir}


def evaluate_mpc_z_state(model_dir,
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

    model_data = model_builder.load_dynamics_model_from_folder(model_dir, strict=strict)
    model_dy = model_data['model_dy']
    model_dy = model_dy.eval()
    model_dy = model_dy.cuda()
    model_config = model_dy.config




    # create the environment
    env = DrakePusherSliderEnv(env_config, visualize=False)
    env.reset()

    observation_function = ObservationFunctionFactory.function_from_config(model_config)
    action_function = ActionFunctionFactory.function_from_config(model_config)
    episode = OnlineEpisodeReader()

    mpc_input_builder = DynamicsModelInputBuilder(observation_function=observation_function,
                                                  action_function=action_function,
                                                  visual_observation_function=None,
                                                  episode=episode)

    pts_GT = np.array(model_config['dataset']['observation_function']['GT_3D_object_points'])
    K = pts_GT.shape[0]  # num keypoints
    eval_indices = np.arange(3 * K)  # extract the keypoints

    def goal_func(obs_tmp):
        state_tmp = mpc_input_builder.get_state_input_single_timestep({'observation': obs_tmp})['state']
        # z_dict= model_dy.compute_z_state(state_tmp.unsqueeze(0))
        # print("z_dict['z_object'].shape", z_dict['z_object'].shape)
        return model_dy.compute_z_state(state_tmp.unsqueeze(0))['z_object_flat']


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
        raise ValueError("unknown planner type: %s" %(planner_type))

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
                                              generate_initial_condition_func=generate_initial_condition_func,
                                              )

    return {'save_dir': save_dir}