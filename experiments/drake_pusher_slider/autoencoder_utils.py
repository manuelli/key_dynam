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
from key_dynam.models.model_builder import build_dynamics_model
from key_dynam.autoencoder.autoencoder_models import ConvolutionalAutoencoder


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
    else:
        raise ValueError("unknown dataset:", dataset_name)

    return {'transporter_model_name': transporter_model_name,
            'precomputed_data_root': precomputed_data_root}


def load_autoencoder_model(train_dir):
    chkpt_file = "net_best_state_dict.pth"
    ckpt_file = os.path.join(train_dir, chkpt_file)

    config = load_yaml(os.path.join(train_dir, 'config.yaml'))
    state_dict = torch.load(ckpt_file)

    # build dynamics model
    model_dy = build_dynamics_model(config)
    model_dy.load_state_dict(state_dict['dynamics'])
    model_dy = model_dy.eval()
    model_dy = model_dy.cuda()

    # build autoencoder file
    model_ae = ConvolutionalAutoencoder.from_global_config(config)
    model_ae.load_state_dict(state_dict['autoencoder'])
    model_ae = model_ae.eval()
    model_ae = model_ae.cuda()


    return {'model_ae': model_ae,
            'model_dy': model_dy,
            'config': config,
            }


def evaluate_mpc_z_state(train_dir,
                         config_planner_mpc=None,
                         save_dir=None,
                         planner_type=None,
                         env_config=None,
                         generate_initial_condition_func=None,
                         env_type="DrakePusherSliderEnv",
                         ):

    assert save_dir is not None
    assert planner_type is not None
    assert env_config is not None
    assert generate_initial_condition_func is not None
    assert env_type is not None

    model_dict = load_autoencoder_model(train_dir)
    model_dy = model_dict['model_dy']
    model_ae = model_dict['model_ae']
    config = model_dict['config']
    model_config = config

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

    action_function = ActionFunctionFactory.function_from_config(config)
    observation_function = ObservationFunctionFactory.drake_pusher_position_3D(config)
    # visual observation function
    visual_observation_function = VisualObservationFunctionFactory.autoencoder_latent_state(config,
                                                                                        model_ae=model_ae)

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
