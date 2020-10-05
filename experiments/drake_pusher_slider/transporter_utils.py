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
from key_dynam.envs.drake_pusher_slider import DrakePusherSliderEnv
from key_dynam.envs.drake_mugs import DrakeMugsEnv
from key_dynam.dataset.online_episode_reader import OnlineEpisodeReader
from key_dynam.dataset.mpc_dataset import DynamicsModelInputBuilder
from key_dynam.planner.planners import RandomShootingPlanner, PlannerMPPI
from key_dynam.eval import mpc_eval_drake_pusher_slider
from key_dynam.dataset.vision_function_factory import VisualObservationFunctionFactory
from key_dynam.dynamics.models_dy import get_object_and_robot_state_indices
from key_dynam.models import model_builder
from key_dynam.transporter.models_kp import Transporter


def get_precomputed_data_root(dataset_name):
    transporter_model_name = None
    precomputed_data_root = None

    if dataset_name == "2020-04-20-14-58-21-418302_T_aug_random_velocity_1000":
        transporter_model_name = "transporter_2020-05-06-19-11-54-206998"
        precomputed_data_root = os.path.join(get_data_root(),
                                             "dev/experiments/14/trained_models/perception",
                                             transporter_model_name,
                                             "precomputed_vision_data/transporter_keypoints",
                                             "dataset_%s" % (dataset_name))
    elif dataset_name == "2020-04-23-20-45-12-697915_T_aug_random_velocity_1000_angled_cam":
        transporter_model_name = "transporter_2020-05-07-22-26-56-654913"
        precomputed_data_root = os.path.join(get_data_root(),
                                             "dev/experiments/15/trained_models/perception",
                                             transporter_model_name,
                                             "precomputed_vision_data/transporter_keypoints",
                                             "dataset_%s" % (dataset_name))
    elif dataset_name == "dps_box_on_side_600":
        transporter_model_name = "transporter_camera_angled_2020-05-13-23-38-18-580817"
        precomputed_data_root = "/home/manuelli/data/key_dynam/dev/experiments/drake_pusher_slider_box_on_side/dataset_dps_box_on_side_600/trained_models/perception/transporter_camera_angled_2020-05-13-23-38-18-580817/precomputed_vision_data/transporter_keypoints/dataset_dps_box_on_side_600"
    elif dataset_name == "correlle_mug-small_many_colors_600":
        transporter_model_name = "transporter_2020-06-10-22-59-54-896478"
        precomputed_data_root = "/home/manuelli/data/key_dynam/dev/experiments/20/dataset_correlle_mug-small_many_colors_600/trained_models/perception/transporter/transporter_2020-06-10-22-59-54-896478/precomputed_vision_data/transporter_keypoints/dataset_correlle_mug-small_many_colors_600"
    elif dataset_name == "box_push_1000_top_down":
        transporter_model_name = "transporter_standard_2020-06-14-22-29-31-256422"
        precomputed_data_root = "/home/manuelli/data/key_dynam/dev/experiments/drake_pusher_slider_v2/dataset_box_push_1000_top_down/trained_models/perception/transporter/transporter_standard_2020-06-14-22-29-31-256422/precomputed_vision_data/transporter_keypoints/dataset_box_push_1000_top_down"
    elif dataset_name == "box_push_1000_angled":
        transporter_model_name = "transporter_standard_2020-06-15-18-35-52-478769"
        precomputed_data_root = "/home/manuelli/data/key_dynam/dev/experiments/drake_pusher_slider_v2/dataset_box_push_1000_angled/trained_models/perception/transporter/transporter_standard_2020-06-15-18-35-52-478769/precomputed_vision_data/transporter_keypoints/dataset_box_push_1000_angled"
    else:
        raise ValueError("unknown dataset:", dataset_name)

    return {'transporter_model_name': transporter_model_name,
            'precomputed_data_root': precomputed_data_root}


def load_transporter_model(model_file=None):
    train_dir = os.path.dirname(os.path.dirname(model_file))

    print("train_dir", train_dir)

    config = load_yaml(os.path.join(train_dir, 'config.yaml'))

    model_kp = Transporter(config, use_gpu=True)
    model_kp.load_state_dict(torch.load(model_file))
    model_kp = model_kp.cuda()
    model_kp = model_kp.eval()

    return {'model': model_kp,
            'model_file': model_file,
            'train_dir': train_dir,
            'config': config,
            }


def load_model(model_folder, strict=True):
    model_dy_dict = model_builder.load_dynamics_model_from_folder(model_folder, strict=strict)
    _, model_name = os.path.split(model_folder)

    config = model_dy_dict['config']

    # correct way
    precomputed_data_root = config['dataset']['precomputed_data_root']
    metadata = load_yaml(os.path.join(precomputed_data_root, 'metadata.yaml'))
    model_kp_file = metadata['model_file']

    print("model_kp_file", model_kp_file)

    model_kp_dict = load_transporter_model(model_file=model_kp_file)

    return {"model_dy": model_dy_dict,
            'model_kp': model_kp_dict,
            'model_name': model_name}


def evaluate_mpc(model_dir,
                 config_planner_mpc=None,
                 save_dir=None,
                 planner_type=None,
                 env_config=None,
                 strict=True,
                 generate_initial_condition_func=None
                 ):
    assert save_dir is not None
    assert planner_type is not None
    assert env_config is not None
    assert generate_initial_condition_func is not None

    model_dict = load_model(model_dir, strict=strict)
    model_dy = model_dict['model_dy']['model_dy']
    config = model_dict['model_dy']['config']
    model_config = config

    model_kp = model_dict['model_kp']['model']
    config_kp = model_kp.config
    camera_name = config_kp['perception']['camera_name']

    # create the environment
    env = DrakePusherSliderEnv(env_config, visualize=False)
    env.reset()

    T_world_camera = env.camera_pose(camera_name)
    camera_K_matrix = env.camera_K_matrix(camera_name)
    mask_labels = env.get_labels_to_mask_list()

    action_function = ActionFunctionFactory.function_from_config(config)
    observation_function = ObservationFunctionFactory.drake_pusher_position_3D(config)
    visual_observation_function = \
        VisualObservationFunctionFactory.function_from_config(config,
                                                              camera_name=camera_name,
                                                              model_kp=model_kp,
                                                              K_matrix=camera_K_matrix,
                                                              T_world_camera=T_world_camera,
                                                              mask_labels=mask_labels)

    episode = OnlineEpisodeReader()

    mpc_input_builder = DynamicsModelInputBuilder(observation_function=observation_function,
                                                  visual_observation_function=visual_observation_function,
                                                  action_function=action_function,
                                                  episode=episode)

    def goal_func(obs_local):
        keypoints_dict = visual_observation_function(obs_local)
        return torch.Tensor(keypoints_dict['tensor'])

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
                         env_type="DrakePusherSliderEnv",
                         ):
    assert save_dir is not None
    assert planner_type is not None
    assert env_config is not None
    assert generate_initial_condition_func is not None

    model_dict = load_model(model_dir, strict=strict)
    model_dy = model_dict['model_dy']['model_dy']
    config = model_dict['model_dy']['config']
    model_config = config

    model_kp = model_dict['model_kp']['model']
    config_kp = model_kp.config
    camera_name = config_kp['perception']['camera_name']

    # create the environment
    # create the environment
    env = None
    if env_type == "DrakePusherSliderEnv":
        env = DrakePusherSliderEnv(env_config, visualize=False)
    elif env_type == "DrakeMugsEnv":
        env = DrakeMugsEnv(env_config, visualize=False)
    else:
        raise ValueError("unknown env type: %s" % (env_type))
    env.reset()

    T_world_camera = env.camera_pose(camera_name)
    camera_K_matrix = env.camera_K_matrix(camera_name)
    mask_labels = env.get_labels_to_mask_list()

    action_function = ActionFunctionFactory.function_from_config(config)
    observation_function = ObservationFunctionFactory.drake_pusher_position_3D(config)
    visual_observation_function = \
        VisualObservationFunctionFactory.function_from_config(config,
                                                              camera_name=camera_name,
                                                              model_kp=model_kp,
                                                              K_matrix=camera_K_matrix,
                                                              T_world_camera=T_world_camera,
                                                              mask_labels=mask_labels)

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
                                              generate_initial_condition_func=generate_initial_condition_func
                                              )

    return {'save_dir': save_dir}
