import os
import functools
import shutil
import numpy as np
import pydrake
import torch
import copy as copy

# dense correspondence
from dense_correspondence.training.train_integral_heatmap_3d import train_dense_descriptors

# key_dynam
from key_dynam.experiments.exp_09 import utils as exp_utils
from key_dynam.utils.utils import get_project_root, load_yaml, get_current_YYYY_MM_DD_hh_mm_ss_ms, get_data_root, \
    load_pickle
from key_dynam.experiments.drake_pusher_slider.utils import get_dataset_paths
from key_dynam.experiments.drake_pusher_slider import utils as dps_utils
from key_dynam.dataset.drake_sim_episode_reader import DrakeSimEpisodeReader
from key_dynam.dataset.vision_function_factory import PrecomputedVisualObservationFunctionFactory
from key_dynam.utils.torch_utils import get_freer_gpu, set_cuda_visible_devices
from key_dynam.dense_correspondence.dc_drake_sim_episode_reader import DCDrakeSimEpisodeReader
from key_dynam.autoencoder.train_autoencoder import train_autoencoder
from key_dynam.autoencoder.dataset import AutoencoderImagePreprocessFunctionFactory
from key_dynam.transporter.models_kp import Transporter
from key_dynam.training.train_dynamics_pusher_slider_weight_matrix import \
    train_dynamics as train_dynamics_weight_matrix
from key_dynam.dense_correspondence.precompute_descriptors import run_precompute_descriptors_pipeline

from multiprocessing import Process

TRAIN_VALID_RATIO = 0.5


def get_experiment_save_root(dataset_name):
    return os.path.join(get_data_root(), 'dev/experiments/drake_pusher_slider_v2/dataset_%s' % (dataset_name))


def get_eval_config_mpc():
    return load_yaml(os.path.join(get_project_root(), 'experiments/drake_pusher_slider/eval_config.yaml'))


def create_initial_condition_func():
    eval_config = get_eval_config_mpc()

    generate_initial_condition_func = functools.partial(dps_utils.generate_initial_condition,
                                                        config=eval_config,
                                                        T_aug_enabled=True)

    return generate_initial_condition_func


def GT_3D(dataset_name):
    from key_dynam.training.train_dynamics_pusher_slider_precomputed_keypoints import train_dynamics
    from key_dynam.experiments.drake_pusher_slider import GT_3D_utils

    def load_config():
        return load_yaml(os.path.join(get_project_root(), 'experiments/drake_pusher_slider/config_GT_3D.yaml'))

    dataset_paths = get_dataset_paths(dataset_name)
    dataset_root = dataset_paths['dataset_root']
    env_config = dataset_paths['config']

    multi_episode_dict = DrakeSimEpisodeReader.load_dataset(dataset_root=dataset_root,
                                                            load_image_data=False,
                                                            )

    experiment_save_root = get_experiment_save_root(dataset_name)
    planner_type = "mppi"

    # standard
    if False:
        TRAIN = True
        EVAL_MPC = True

        train_dir = None

        config = load_config()
        config['train']['n_history'] = 2
        config['train']['random_seed'] = 1
        suffix = "_GT_3D_n_his_2"
        model_name = get_current_YYYY_MM_DD_hh_mm_ss_ms() + suffix

        if TRAIN:
            train_dir = os.path.join(experiment_save_root, 'trained_models/dynamics/GT_3D/',
                                     model_name)

            os.makedirs(train_dir)

            train_dynamics(config=config,
                           train_dir=train_dir,
                           multi_episode_dict=multi_episode_dict,
                           visual_observation_function=None,
                           )

        if EVAL_MPC:
            # for debugging purposes
            config_planner_mpc = get_eval_config_mpc()
            # save_dir = "/tmp/mpc"
            save_dir = os.path.join(experiment_save_root, 'eval/mpc/GT_3D', model_name,
                                    get_current_YYYY_MM_DD_hh_mm_ss_ms())

            eval_mpc_data = GT_3D_utils.evaluate_mpc_z_state(model_dir=train_dir,
                                                             config_planner_mpc=config_planner_mpc,
                                                             save_dir=save_dir,
                                                             planner_type=planner_type,
                                                             env_config=env_config,
                                                             strict=False,
                                                             generate_initial_condition_func=create_initial_condition_func(),
                                                             )

            # eval_mpc_data = GT_3D_utils.evaluate_mpc(model_dir=train_dir,
            #                                          config_planner_mpc=config_planner_mpc,
            #                                          save_dir=save_dir,
            #                                          planner_type=planner_type,
            #                                          env_config=env_config,
            #                                          strict=False,
            #                                          generate_initial_condition_func=create_initial_condition_func(),
            #                                          )

    # no_T_aug
    if True:
        TRAIN = True
        EVAL_MPC = True

        train_dir = None

        config = load_config()
        config['train']['n_history'] = 2
        config['train']['random_seed'] = 1
        config['dataset']['data_augmentation']['enabled'] = False
        suffix = "_GT_3D_n_his_2_no_T_aug"
        model_name = get_current_YYYY_MM_DD_hh_mm_ss_ms() + suffix

        if TRAIN:
            train_dir = os.path.join(experiment_save_root, 'trained_models/dynamics/GT_3D/',
                                     model_name)

            os.makedirs(train_dir)

            train_dynamics(config=config,
                           train_dir=train_dir,
                           multi_episode_dict=multi_episode_dict,
                           visual_observation_function=None,
                           )

        if EVAL_MPC:

            config_planner_mpc = get_eval_config_mpc()
            save_dir = os.path.join(experiment_save_root, 'eval/mpc/GT_3D', model_name,
                                    get_current_YYYY_MM_DD_hh_mm_ss_ms())

            eval_mpc_data = GT_3D_utils.evaluate_mpc_z_state(model_dir=train_dir,
                                                             config_planner_mpc=config_planner_mpc,
                                                             save_dir=save_dir,
                                                             planner_type=planner_type,
                                                             env_config=env_config,
                                                             strict=False,
                                                             generate_initial_condition_func=create_initial_condition_func(),
                                                             )


def DD_3D(dataset_name):
    from key_dynam.training.train_dynamics_pusher_slider_precomputed_keypoints import train_dynamics
    from key_dynam.experiments.drake_pusher_slider import DD_utils

    def load_config():
        return load_yaml(os.path.join(get_project_root(), 'experiments/drake_pusher_slider/config_DD_3D.yaml'))

    dataset_paths = get_dataset_paths(dataset_name)
    dataset_root = dataset_paths['dataset_root']
    env_config = dataset_paths['config']

    precomputed_vision_data_root = DD_utils.get_precomputed_data_root(dataset_name)['precomputed_data_root']

    descriptor_keypoints_root = os.path.join(precomputed_vision_data_root, 'descriptor_keypoints')

    multi_episode_dict = DrakeSimEpisodeReader.load_dataset(dataset_root=dataset_root,
                                                            load_image_data=True,
                                                            precomputed_data_root=descriptor_keypoints_root)

    experiment_save_root = get_experiment_save_root(dataset_name)
    planner_type = "mppi"

    # experiment_save_root = os.path.join(get_data_root(), 'sandbox')

    # standard
    if True:
        TRAIN = True
        EVAL_MPC = True

        spatial_descriptor_data = load_pickle(os.path.join(precomputed_vision_data_root, 'spatial_descriptors.p'))
        metadata = load_pickle(os.path.join(precomputed_vision_data_root, 'metadata.p'))

        train_dir = None

        config = load_config()
        config['train']['n_history'] = 2
        config['train']['random_seed'] = 1
        config['dataset']['visual_observation_function']['camera_name'] = dataset_paths['main_camera_name']

        config['dataset']['precomputed_data_root'] = precomputed_vision_data_root
        suffix = "_DD_3D_n_his_2"
        model_name = get_current_YYYY_MM_DD_hh_mm_ss_ms() + suffix

        if TRAIN:
            train_dir = os.path.join(experiment_save_root, 'trained_models/dynamics/DD_2D/',
                                     model_name)

            os.makedirs(train_dir)

            visual_observation_function = PrecomputedVisualObservationFunctionFactory.function_from_config(config,
                                                                                                           keypoint_idx=
                                                                                                           spatial_descriptor_data[
                                                                                                               'spatial_descriptors_idx'])

            train_dynamics(config=config,
                           train_dir=train_dir,
                           multi_episode_dict=multi_episode_dict,
                           visual_observation_function=visual_observation_function,
                           metadata=metadata,
                           spatial_descriptors_data=spatial_descriptor_data
                           )

        if EVAL_MPC:


            config_planner_mpc = get_eval_config_mpc()
            save_dir = os.path.join(experiment_save_root, 'eval/mpc/DD_3D', model_name,
                                    get_current_YYYY_MM_DD_hh_mm_ss_ms())

            save_dir = os.path.join(experiment_save_root, 'eval/mpc/DD_3D', model_name,
                                    get_current_YYYY_MM_DD_hh_mm_ss_ms() + "_new_planner")

            eval_mpc_data = DD_utils.evaluate_mpc_z_state(model_dir=train_dir,
                                                          config_planner_mpc=config_planner_mpc,
                                                          save_dir=save_dir,
                                                          planner_type=planner_type,
                                                          env_config=env_config,
                                                          generate_initial_condition_func=create_initial_condition_func(),
                                                          )

    # spatial_z
    # WSDS
    if True:
        TRAIN = True
        EVAL_MPC = True

        train_dir = None

        config = load_config()
        config['train']['n_history'] = 2
        config['train']['random_seed'] = 1
        config['dataset']['visual_observation_function']['camera_name'] = dataset_paths['main_camera_name']

        config['dataset']['precomputed_data_root'] = precomputed_vision_data_root

        config['dynamics_net']['model_type'] = 'mlp_weight_matrix'
        config['freeze_keypoint_weights'] = False

        suffix = "_DD_3D_spatial_z_n_his_2"
        model_name = get_current_YYYY_MM_DD_hh_mm_ss_ms() + suffix

        spatial_descriptor_data = load_pickle(os.path.join(precomputed_vision_data_root, 'spatial_descriptors.p'))
        metadata = load_pickle(os.path.join(precomputed_vision_data_root, 'metadata.p'))

        if TRAIN:
            train_dir = os.path.join(experiment_save_root, 'trained_models/dynamics/DD_3D/',
                                     model_name)

            os.makedirs(train_dir)

            visual_observation_function = PrecomputedVisualObservationFunctionFactory.function_from_config(config,
                                                                                                           keypoint_idx=
                                                                                                           spatial_descriptor_data[
                                                                                                               'spatial_descriptors_idx'])

            train_dynamics_weight_matrix(config=config,
                                         train_dir=train_dir,
                                         multi_episode_dict=multi_episode_dict,
                                         visual_observation_function=visual_observation_function,
                                         metadata=metadata,
                                         spatial_descriptors_data=spatial_descriptor_data
                                         )

        if EVAL_MPC:
            config_planner_mpc = get_eval_config_mpc()
            save_dir = os.path.join(experiment_save_root, 'eval/mpc/DD_3D', model_name,
                                    get_current_YYYY_MM_DD_hh_mm_ss_ms())

            eval_mpc_data = DD_utils.evaluate_mpc_z_state(model_dir=train_dir,
                                                          config_planner_mpc=config_planner_mpc,
                                                          save_dir=save_dir,
                                                          planner_type=planner_type,
                                                          env_config=env_config,
                                                          generate_initial_condition_func=create_initial_condition_func(),
                                                          )

    # all
    if True:
        TRAIN = True
        EVAL_MPC = True

        train_dir = None

        spatial_descriptor_data = load_pickle(os.path.join(precomputed_vision_data_root, 'spatial_descriptors.p'))
        metadata = load_pickle(os.path.join(precomputed_vision_data_root, 'metadata.p'))

        config = load_config()

        # uses all descriptors, rather than just first 5
        ref_descriptors = metadata['ref_descriptors']
        spatial_descriptor_data['spatial_descriptors'] = ref_descriptors
        K = spatial_descriptor_data['spatial_descriptors'].shape[0]
        spatial_descriptor_data['spatial_descriptors_idx'] = list(range(0, K))
        config['dataset']['state_dim'] = 3 * K + 3
        config['dataset']['object_state_shape'] = [K, 3]

        config['train']['n_history'] = 2
        config['train']['random_seed'] = 1
        config['dataset']['visual_observation_function']['camera_name'] = dataset_paths['main_camera_name']

        config['dataset']['precomputed_data_root'] = precomputed_vision_data_root

        config['dynamics_net']['type'] = 'mlp'
        suffix = "_DD_3D_all_n_his_2"
        model_name = get_current_YYYY_MM_DD_hh_mm_ss_ms() + suffix

        if TRAIN:
            train_dir = os.path.join(experiment_save_root, 'trained_models/dynamics/DD_3D/',
                                     model_name)

            os.makedirs(train_dir)

            visual_observation_function = PrecomputedVisualObservationFunctionFactory.function_from_config(config,
                                                                                                           keypoint_idx=
                                                                                                           spatial_descriptor_data[
                                                                                                               'spatial_descriptors_idx'])

            train_dynamics(config=config,
                           train_dir=train_dir,
                           multi_episode_dict=multi_episode_dict,
                           visual_observation_function=visual_observation_function,
                           metadata=metadata,
                           spatial_descriptors_data=spatial_descriptor_data
                           )

        if EVAL_MPC:


            config_planner_mpc = get_eval_config_mpc()
            save_dir = os.path.join(experiment_save_root, 'eval/mpc/DD_3D', model_name,
                                    get_current_YYYY_MM_DD_hh_mm_ss_ms())

            eval_mpc_data = DD_utils.evaluate_mpc_z_state(model_dir=train_dir,
                                                          config_planner_mpc=config_planner_mpc,
                                                          save_dir=save_dir,
                                                          planner_type=planner_type,
                                                          env_config=env_config,
                                                          generate_initial_condition_func=create_initial_condition_func(),
                                                          )

    # all_z
    # WDS
    if True:
        TRAIN = True
        EVAL_MPC = True

        train_dir = None

        spatial_descriptor_data = load_pickle(os.path.join(precomputed_vision_data_root, 'spatial_descriptors.p'))
        metadata = load_pickle(os.path.join(precomputed_vision_data_root, 'metadata.p'))

        config = load_config()

        ref_descriptors = metadata['ref_descriptors']
        spatial_descriptor_data['spatial_descriptors'] = ref_descriptors
        K = spatial_descriptor_data['spatial_descriptors'].shape[0]
        spatial_descriptor_data['spatial_descriptors_idx'] = list(range(0, K))
        config['dataset']['state_dim'] = 3 * K + 3
        config['dataset']['object_state_shape'] = [K, 3]

        config['train']['n_history'] = 2
        config['train']['random_seed'] = 1
        config['dataset']['visual_observation_function']['camera_name'] = dataset_paths['main_camera_name']

        config['dataset']['precomputed_data_root'] = precomputed_vision_data_root

        config['dynamics_net']['model_type'] = 'mlp_weight_matrix'
        config['freeze_keypoint_weights'] = False

        suffix = "_DD_3D_all_z_n_his_2"
        model_name = get_current_YYYY_MM_DD_hh_mm_ss_ms() + suffix

        if TRAIN:
            train_dir = os.path.join(experiment_save_root, 'trained_models/dynamics/DD_3D/',
                                     model_name)

            os.makedirs(train_dir)

            visual_observation_function = PrecomputedVisualObservationFunctionFactory.function_from_config(config,
                                                                                                           keypoint_idx=
                                                                                                           spatial_descriptor_data[
                                                                                                               'spatial_descriptors_idx'])

            train_dynamics_weight_matrix(config=config,
                                         train_dir=train_dir,
                                         multi_episode_dict=multi_episode_dict,
                                         visual_observation_function=visual_observation_function,
                                         metadata=metadata,
                                         spatial_descriptors_data=spatial_descriptor_data
                                         )

        if EVAL_MPC:
            config_planner_mpc = get_eval_config_mpc()
            save_dir = os.path.join(experiment_save_root, 'eval/mpc/DD_3D', model_name,
                                    get_current_YYYY_MM_DD_hh_mm_ss_ms())

            eval_mpc_data = DD_utils.evaluate_mpc_z_state(model_dir=train_dir,
                                                          config_planner_mpc=config_planner_mpc,
                                                          save_dir=save_dir,
                                                          planner_type=planner_type,
                                                          env_config=env_config,
                                                          generate_initial_condition_func=create_initial_condition_func(),
                                                          )

    # standard (no T_aug)
    if True:
        TRAIN = True
        EVAL_MPC = True

        spatial_descriptor_data = load_pickle(os.path.join(precomputed_vision_data_root, 'spatial_descriptors.p'))
        metadata = load_pickle(os.path.join(precomputed_vision_data_root, 'metadata.p'))

        train_dir = None

        config = load_config()
        config['train']['n_history'] = 2
        config['train']['random_seed'] = 1
        config['dataset']['visual_observation_function']['camera_name'] = dataset_paths['main_camera_name']

        config['dataset']['precomputed_data_root'] = precomputed_vision_data_root
        config['dataset']['data_augmentation']['enabled'] = False
        suffix = "_DD_3D_n_his_2_no_T_aug"
        model_name = get_current_YYYY_MM_DD_hh_mm_ss_ms() + suffix

        if TRAIN:
            train_dir = os.path.join(experiment_save_root, 'trained_models/dynamics/DD_3D/',
                                     model_name)

            os.makedirs(train_dir)

            visual_observation_function = PrecomputedVisualObservationFunctionFactory.function_from_config(config,
                                                                                                           keypoint_idx=
                                                                                                           spatial_descriptor_data[
                                                                                                               'spatial_descriptors_idx'])

            train_dynamics(config=config,
                           train_dir=train_dir,
                           multi_episode_dict=multi_episode_dict,
                           visual_observation_function=visual_observation_function,
                           metadata=metadata,
                           spatial_descriptors_data=spatial_descriptor_data
                           )

        if EVAL_MPC:
            config_planner_mpc = get_eval_config_mpc()
            save_dir = os.path.join(experiment_save_root, 'eval/mpc/DD_3D', model_name,
                                    get_current_YYYY_MM_DD_hh_mm_ss_ms())

            save_dir = os.path.join(experiment_save_root, 'eval/mpc/DD_3D', model_name,
                                    get_current_YYYY_MM_DD_hh_mm_ss_ms() + "_new_planner")

            eval_mpc_data = DD_utils.evaluate_mpc_z_state(model_dir=train_dir,
                                                          config_planner_mpc=config_planner_mpc,
                                                          save_dir=save_dir,
                                                          planner_type=planner_type,
                                                          env_config=env_config,
                                                          generate_initial_condition_func=create_initial_condition_func(),
                                                          )

    # all_z no T_aug
    # WDS
    if True:
        TRAIN = True
        EVAL_MPC = True

        train_dir = None

        spatial_descriptor_data = load_pickle(os.path.join(precomputed_vision_data_root, 'spatial_descriptors.p'))
        metadata = load_pickle(os.path.join(precomputed_vision_data_root, 'metadata.p'))

        config = load_config()

        ref_descriptors = metadata['ref_descriptors']
        spatial_descriptor_data['spatial_descriptors'] = ref_descriptors
        K = spatial_descriptor_data['spatial_descriptors'].shape[0]
        spatial_descriptor_data['spatial_descriptors_idx'] = list(range(0, K))
        config['dataset']['state_dim'] = 3 * K + 3
        config['dataset']['object_state_shape'] = [K, 3]
        config['dataset']['data_augmentation']['enabled'] = False

        config['train']['n_history'] = 2
        config['train']['random_seed'] = 1
        config['dataset']['visual_observation_function']['camera_name'] = dataset_paths['main_camera_name']

        config['dataset']['precomputed_data_root'] = precomputed_vision_data_root

        config['dynamics_net']['model_type'] = 'mlp_weight_matrix'
        config['freeze_keypoint_weights'] = False

        suffix = "_DD_3D_all_z_n_his_2_no_T_aug"
        model_name = get_current_YYYY_MM_DD_hh_mm_ss_ms() + suffix

        if TRAIN:
            train_dir = os.path.join(experiment_save_root, 'trained_models/dynamics/DD_3D/',
                                     model_name)

            os.makedirs(train_dir)

            visual_observation_function = PrecomputedVisualObservationFunctionFactory.function_from_config(config,
                                                                                                           keypoint_idx=
                                                                                                           spatial_descriptor_data[
                                                                                                               'spatial_descriptors_idx'])

            train_dynamics_weight_matrix(config=config,
                                         train_dir=train_dir,
                                         multi_episode_dict=multi_episode_dict,
                                         visual_observation_function=visual_observation_function,
                                         metadata=metadata,
                                         spatial_descriptors_data=spatial_descriptor_data
                                         )

        if EVAL_MPC:
            config_planner_mpc = get_eval_config_mpc()
            save_dir = os.path.join(experiment_save_root, 'eval/mpc/DD_3D', model_name,
                                    get_current_YYYY_MM_DD_hh_mm_ss_ms())

            eval_mpc_data = DD_utils.evaluate_mpc_z_state(model_dir=train_dir,
                                                          config_planner_mpc=config_planner_mpc,
                                                          save_dir=save_dir,
                                                          planner_type=planner_type,
                                                          env_config=env_config,
                                                          generate_initial_condition_func=create_initial_condition_func(),
                                                          )


def DD_2D(dataset_name):
    from key_dynam.training.train_dynamics_pusher_slider_precomputed_keypoints import train_dynamics
    from key_dynam.experiments.drake_pusher_slider import DD_utils

    dataset_paths = get_dataset_paths(dataset_name)
    dataset_root = dataset_paths['dataset_root']
    env_config = dataset_paths['config']

    def load_config():
        config = load_yaml(os.path.join(get_project_root(), 'experiments/drake_pusher_slider/config_DD_3D.yaml'))
        config['dataset']['visual_observation_function']['type'] = "precomputed_descriptor_keypoints_2D"
        config['dataset']['visual_observation_function']['camera_name'] = dataset_paths['main_camera_name']
        config['dataset']['data_augmentation']['enabled'] = False
        return config


    precomputed_vision_data_root = DD_utils.get_precomputed_data_root(dataset_name)['precomputed_data_root']

    descriptor_keypoints_root = os.path.join(precomputed_vision_data_root, 'descriptor_keypoints')

    multi_episode_dict = DrakeSimEpisodeReader.load_dataset(dataset_root=dataset_root,
                                                            load_image_data=True,
                                                            precomputed_data_root=descriptor_keypoints_root)

    experiment_save_root = get_experiment_save_root(dataset_name)
    planner_type = "mppi"

    # experiment_save_root = os.path.join(get_data_root(), 'sandbox')

    # standard
    if True:
        TRAIN = True
        EVAL_MPC = True

        spatial_descriptor_data = load_pickle(os.path.join(precomputed_vision_data_root, 'spatial_descriptors.p'))
        metadata = load_pickle(os.path.join(precomputed_vision_data_root, 'metadata.p'))

        train_dir = None

        config = load_config()
        config['train']['n_history'] = 2
        config['train']['random_seed'] = 1
        config['dataset']['visual_observation_function']['camera_name'] = dataset_paths['main_camera_name']

        K = spatial_descriptor_data['spatial_descriptors'].shape[0]
        spatial_descriptor_data['spatial_descriptors_idx'] = list(range(0, K))
        config['dataset']['state_dim'] = 2 * K + 3
        config['dataset']['object_state_shape'] = [K, 2]

        config['dataset']['precomputed_data_root'] = precomputed_vision_data_root
        suffix = "_DD_2D_spatial_n_his_2"
        model_name = get_current_YYYY_MM_DD_hh_mm_ss_ms() + suffix

        if TRAIN:
            train_dir = os.path.join(experiment_save_root, 'trained_models/dynamics/DD_2D/',
                                     model_name)

            os.makedirs(train_dir)

            visual_observation_function = PrecomputedVisualObservationFunctionFactory.function_from_config(config,
                                                                                                           keypoint_idx=
                                                                                                           spatial_descriptor_data[
                                                                                                               'spatial_descriptors_idx'])

            train_dynamics(config=config,
                           train_dir=train_dir,
                           multi_episode_dict=multi_episode_dict,
                           visual_observation_function=visual_observation_function,
                           metadata=metadata,
                           spatial_descriptors_data=spatial_descriptor_data
                           )

        if EVAL_MPC:


            config_planner_mpc = get_eval_config_mpc()
            save_dir = os.path.join(experiment_save_root, 'eval/mpc/DD_2D', model_name,
                                    get_current_YYYY_MM_DD_hh_mm_ss_ms())

            eval_mpc_data = DD_utils.evaluate_mpc_z_state(model_dir=train_dir,
                                                          config_planner_mpc=config_planner_mpc,
                                                          save_dir=save_dir,
                                                          planner_type=planner_type,
                                                          env_config=env_config,
                                                          generate_initial_condition_func=create_initial_condition_func(),
                                                          )

    # all_z
    # WDS
    if True:
        TRAIN = True
        EVAL_MPC = True

        train_dir = None

        spatial_descriptor_data = load_pickle(os.path.join(precomputed_vision_data_root, 'spatial_descriptors.p'))
        metadata = load_pickle(os.path.join(precomputed_vision_data_root, 'metadata.p'))

        config = load_config()

        ref_descriptors = metadata['ref_descriptors']
        spatial_descriptor_data['spatial_descriptors'] = ref_descriptors
        K = spatial_descriptor_data['spatial_descriptors'].shape[0]
        spatial_descriptor_data['spatial_descriptors_idx'] = list(range(0, K))
        config['dataset']['state_dim'] = 2 * K + 3
        config['dataset']['object_state_shape'] = [K, 2]

        config['train']['n_history'] = 2
        config['train']['random_seed'] = 1
        config['dataset']['visual_observation_function']['camera_name'] = dataset_paths['main_camera_name']

        config['dataset']['precomputed_data_root'] = precomputed_vision_data_root

        config['dynamics_net']['model_type'] = 'mlp_weight_matrix'
        config['freeze_keypoint_weights'] = False

        suffix = "_DD_2D_all_z_n_his_2"
        model_name = get_current_YYYY_MM_DD_hh_mm_ss_ms() + suffix

        if TRAIN:
            train_dir = os.path.join(experiment_save_root, 'trained_models/dynamics/DD_2D/',
                                     model_name)

            os.makedirs(train_dir)

            visual_observation_function = PrecomputedVisualObservationFunctionFactory.function_from_config(config,
                                                                                                           keypoint_idx=
                                                                                                           spatial_descriptor_data[
                                                                                                               'spatial_descriptors_idx'])

            train_dynamics_weight_matrix(config=config,
                                         train_dir=train_dir,
                                         multi_episode_dict=multi_episode_dict,
                                         visual_observation_function=visual_observation_function,
                                         metadata=metadata,
                                         spatial_descriptors_data=spatial_descriptor_data
                                         )

        if EVAL_MPC:
            config_planner_mpc = get_eval_config_mpc()
            save_dir = os.path.join(experiment_save_root, 'eval/mpc/DD_2D', model_name,
                                    get_current_YYYY_MM_DD_hh_mm_ss_ms())

            eval_mpc_data = DD_utils.evaluate_mpc_z_state(model_dir=train_dir,
                                                          config_planner_mpc=config_planner_mpc,
                                                          save_dir=save_dir,
                                                          planner_type=planner_type,
                                                          env_config=env_config,
                                                          generate_initial_condition_func=create_initial_condition_func(),
                                                          )

    # spatial_z
    # WSDS
    if False:
        TRAIN = True
        EVAL_MPC = True

        train_dir = None

        config = load_config()
        config['train']['n_history'] = 2
        config['train']['random_seed'] = 1
        config['dataset']['visual_observation_function']['camera_name'] = dataset_paths['main_camera_name']

        config['dataset']['precomputed_data_root'] = precomputed_vision_data_root

        config['dynamics_net']['model_type'] = 'mlp_weight_matrix'
        config['freeze_keypoint_weights'] = False

        K = spatial_descriptor_data['spatial_descriptors'].shape[0]
        spatial_descriptor_data['spatial_descriptors_idx'] = list(range(0, K))
        config['dataset']['state_dim'] = 2 * K + 3
        config['dataset']['object_state_shape'] = [K, 2]

        suffix = "_DD_2D_spatial_z_n_his_2"
        model_name = get_current_YYYY_MM_DD_hh_mm_ss_ms() + suffix

        spatial_descriptor_data = load_pickle(os.path.join(precomputed_vision_data_root, 'spatial_descriptors.p'))
        metadata = load_pickle(os.path.join(precomputed_vision_data_root, 'metadata.p'))

        if TRAIN:
            train_dir = os.path.join(experiment_save_root, 'trained_models/dynamics/DD_2D/',
                                     model_name)

            os.makedirs(train_dir)

            visual_observation_function = PrecomputedVisualObservationFunctionFactory.function_from_config(config,
                                                                                                           keypoint_idx=
                                                                                                           spatial_descriptor_data[
                                                                                                               'spatial_descriptors_idx'])

            train_dynamics_weight_matrix(config=config,
                                         train_dir=train_dir,
                                         multi_episode_dict=multi_episode_dict,
                                         visual_observation_function=visual_observation_function,
                                         metadata=metadata,
                                         spatial_descriptors_data=spatial_descriptor_data
                                         )

        if EVAL_MPC:
            config_planner_mpc = get_eval_config_mpc()
            save_dir = os.path.join(experiment_save_root, 'eval/mpc/DD_2D', model_name,
                                    get_current_YYYY_MM_DD_hh_mm_ss_ms())

            eval_mpc_data = DD_utils.evaluate_mpc_z_state(model_dir=train_dir,
                                                          config_planner_mpc=config_planner_mpc,
                                                          save_dir=save_dir,
                                                          planner_type=planner_type,
                                                          env_config=env_config,
                                                          generate_initial_condition_func=create_initial_condition_func(),
                                                          )

    # all
    if False:
        TRAIN = True
        EVAL_MPC = True

        train_dir = None

        spatial_descriptor_data = load_pickle(os.path.join(precomputed_vision_data_root, 'spatial_descriptors.p'))
        metadata = load_pickle(os.path.join(precomputed_vision_data_root, 'metadata.p'))

        config = load_config()

        # uses all descriptors, rather than just first 5
        ref_descriptors = metadata['ref_descriptors']
        spatial_descriptor_data['spatial_descriptors'] = ref_descriptors
        K = spatial_descriptor_data['spatial_descriptors'].shape[0]
        spatial_descriptor_data['spatial_descriptors_idx'] = list(range(0, K))
        config['dataset']['state_dim'] = 2 * K + 3
        config['dataset']['object_state_shape'] = [K, 2]

        config['train']['n_history'] = 2
        config['train']['random_seed'] = 1
        config['dataset']['visual_observation_function']['camera_name'] = dataset_paths['main_camera_name']

        config['dataset']['precomputed_data_root'] = precomputed_vision_data_root

        config['dynamics_net']['type'] = 'mlp'
        suffix = "_DD_2D_all_n_his_2"
        model_name = get_current_YYYY_MM_DD_hh_mm_ss_ms() + suffix

        if TRAIN:
            train_dir = os.path.join(experiment_save_root, 'trained_models/dynamics/DD_2D/',
                                     model_name)

            os.makedirs(train_dir)

            visual_observation_function = PrecomputedVisualObservationFunctionFactory.function_from_config(config,
                                                                                                           keypoint_idx=
                                                                                                           spatial_descriptor_data[
                                                                                                               'spatial_descriptors_idx'])

            train_dynamics(config=config,
                           train_dir=train_dir,
                           multi_episode_dict=multi_episode_dict,
                           visual_observation_function=visual_observation_function,
                           metadata=metadata,
                           spatial_descriptors_data=spatial_descriptor_data
                           )

        if EVAL_MPC:
            # train_dir = "/media/hdd/data/key_dynam/dev/experiments/drake_pusher_slider_v2/dataset_2020-04-20-14-58-21-418302_T_aug_random_velocity_1000/trained_models/dynamics/DD_3D/2020-05-12-00-43-07-392914_DD_3D_all_n_his_2"
            # model_name = "2020-05-12-00-43-07-392914_DD_3D_all_n_his_2"

            config_planner_mpc = get_eval_config_mpc()
            save_dir = os.path.join(experiment_save_root, 'eval/mpc/DD_2D', model_name,
                                    get_current_YYYY_MM_DD_hh_mm_ss_ms())

            eval_mpc_data = DD_utils.evaluate_mpc_z_state(model_dir=train_dir,
                                                          config_planner_mpc=config_planner_mpc,
                                                          save_dir=save_dir,
                                                          planner_type=planner_type,
                                                          env_config=env_config,
                                                          generate_initial_condition_func=create_initial_condition_func(),
                                                          )





def transporter_3D(dataset_name=None, debug=False):
    from key_dynam.training.train_dynamics_pusher_slider_precomputed_keypoints import train_dynamics
    from key_dynam.experiments.drake_pusher_slider import transporter_utils

    def load_config():
        return load_yaml(os.path.join(get_project_root(), 'experiments/drake_pusher_slider/config_transporter_3D.yaml'))

    dataset_paths = get_dataset_paths(dataset_name)
    dataset_root = dataset_paths['dataset_root']
    dataset_name = dataset_paths['dataset_name']
    env_config = dataset_paths['config']

    precomputed_data_dict = transporter_utils.get_precomputed_data_root(dataset_name)
    precomputed_data_root = precomputed_data_dict['precomputed_data_root']

    multi_episode_dict = \
        DrakeSimEpisodeReader.load_dataset(dataset_root,
                                           precomputed_data_root=precomputed_data_root)

    experiment_save_root = get_experiment_save_root(dataset_name)
    planner_type = "mppi"

    # 3D
    if True:
        TRAIN = True
        EVAL_MPC = True

        train_dir = None

        config = load_config()
        config['train'] = config['train_dynamics']
        config['dataset']['precomputed_data_root'] = precomputed_data_root
        config['dataset']['dataset_root'] = dataset_root
        config['train']['n_history'] = 2
        config['train']['random_seed'] = 1

        config['dataset']['visual_observation_function']['camera_name'] = dataset_paths['main_camera_name']

        suffix = "_transporter_3D_n_his_2"
        model_name = get_current_YYYY_MM_DD_hh_mm_ss_ms() + suffix

        if TRAIN:
            train_dir = os.path.join(experiment_save_root, 'trained_models/dynamics/transporter_3D/',
                                     model_name)

            os.makedirs(train_dir)

            visual_observation_function = PrecomputedVisualObservationFunctionFactory.function_from_config(config)

            train_dynamics(config=config,
                           train_dir=train_dir,
                           multi_episode_dict=multi_episode_dict,
                           visual_observation_function=visual_observation_function,
                           )

        if EVAL_MPC:
            config_planner_mpc = get_eval_config_mpc()
            save_dir = os.path.join(experiment_save_root, 'eval/mpc/transporter_3D', model_name,
                                    get_current_YYYY_MM_DD_hh_mm_ss_ms())

            eval_mpc_data = transporter_utils.evaluate_mpc_z_state(model_dir=train_dir,
                                                                   config_planner_mpc=config_planner_mpc,
                                                                   save_dir=save_dir,
                                                                   planner_type=planner_type,
                                                                   env_config=env_config,
                                                                   strict=False,
                                                                   generate_initial_condition_func=create_initial_condition_func(),
                                                                   )

    # 3D + T_aug
    if True:
        TRAIN = True
        EVAL_MPC = True

        train_dir = None

        config = load_config()
        config['train'] = config['train_dynamics']
        config['dataset']['precomputed_data_root'] = precomputed_data_root
        config['dataset']['dataset_root'] = dataset_root
        config['dataset']['data_augmentation']['enabled'] = True
        config['train']['n_history'] = 2
        config['train']['random_seed'] = 1

        config['dataset']['visual_observation_function']['camera_name'] = dataset_paths['main_camera_name']

        suffix = "_transporter_3D_T_aug_n_his_2"
        model_name = get_current_YYYY_MM_DD_hh_mm_ss_ms() + suffix

        if TRAIN:
            train_dir = os.path.join(experiment_save_root, 'trained_models/dynamics/transporter_3D/',
                                     model_name)

            os.makedirs(train_dir)

            visual_observation_function = PrecomputedVisualObservationFunctionFactory.function_from_config(config)

            train_dynamics(config=config,
                           train_dir=train_dir,
                           multi_episode_dict=multi_episode_dict,
                           visual_observation_function=visual_observation_function,
                           )

        if EVAL_MPC:

            config_planner_mpc = get_eval_config_mpc()
            save_dir = os.path.join(experiment_save_root, 'eval/mpc/transporter_3D', model_name,
                                    get_current_YYYY_MM_DD_hh_mm_ss_ms())

            eval_mpc_data = transporter_utils.evaluate_mpc_z_state(model_dir=train_dir,
                                                                   config_planner_mpc=config_planner_mpc,
                                                                   save_dir=save_dir,
                                                                   planner_type=planner_type,
                                                                   env_config=env_config,
                                                                   strict=False,
                                                                   generate_initial_condition_func=create_initial_condition_func(),
                                                                       )

    # transporter_2D
    if True:
        TRAIN = True
        EVAL_MPC = True

        train_dir = None

        config = load_config()
        config['train'] = config['train_dynamics']
        config['dataset']['precomputed_data_root'] = precomputed_data_root
        config['dataset']['dataset_root'] = dataset_root
        config['train']['n_history'] = 2
        config['train']['random_seed'] = 1

        # make it 2D keypoints
        config['dataset']['visual_observation_function']['type'] = 'transporter_keypoints_2D'
        config['dataset']['visual_observation_function']['camera_name'] = dataset_paths['main_camera_name']
        config['dataset']['object_state_shape'] = [6, 2]
        config['dataset']['state_dim'] = 6 * 2 + 3

        suffix = "_transporter_2D_n_his_2"
        model_name = get_current_YYYY_MM_DD_hh_mm_ss_ms() + suffix

        if TRAIN:
            train_dir = os.path.join(experiment_save_root, 'trained_models/dynamics/transporter_3D/',
                                     model_name)

            os.makedirs(train_dir)

            visual_observation_function = PrecomputedVisualObservationFunctionFactory.function_from_config(config)

            train_dynamics(config=config,
                           train_dir=train_dir,
                           multi_episode_dict=multi_episode_dict,
                           visual_observation_function=visual_observation_function,
                           )

        if EVAL_MPC:

            config_planner_mpc = get_eval_config_mpc()
            save_dir = os.path.join(experiment_save_root, 'eval/mpc/transporter_3D', model_name,
                                    get_current_YYYY_MM_DD_hh_mm_ss_ms())

            eval_mpc_data = transporter_utils.evaluate_mpc_z_state(model_dir=train_dir,
                                                                                     config_planner_mpc=config_planner_mpc,
                                                                                     save_dir=save_dir,
                                                                                     planner_type=planner_type,
                                                                                     env_config=env_config,
                                                                                     generate_initial_condition_func=create_initial_condition_func(),
                                                                                     )

    # transporter_3D_z
    if True:
        TRAIN = True
        EVAL_MPC = True

        train_dir = None

        config = load_config()
        config['train'] = config['train_dynamics']
        config['dataset']['precomputed_data_root'] = precomputed_data_root
        config['dataset']['dataset_root'] = dataset_root
        config['train']['n_history'] = 2
        config['train']['random_seed'] = 1

        config['dynamics_net']['model_type'] = 'mlp_weight_matrix'
        config['freeze_keypoint_weights'] = False

        if debug:
            config['train']['n_epoch'] = 10

        config['dataset']['visual_observation_function']['camera_name'] = dataset_paths['main_camera_name']

        suffix = "_transporter_3D_z_n_his_2"
        model_name = get_current_YYYY_MM_DD_hh_mm_ss_ms() + suffix

        if TRAIN:
            train_dir = os.path.join(experiment_save_root, 'trained_models/dynamics/transporter_3D/',
                                     model_name)

            os.makedirs(train_dir)

            visual_observation_function = PrecomputedVisualObservationFunctionFactory.function_from_config(config)

            train_dynamics_weight_matrix(config=config,
                                         train_dir=train_dir,
                                         multi_episode_dict=multi_episode_dict,
                                         visual_observation_function=visual_observation_function,
                                         )

        if EVAL_MPC:
            config_planner_mpc = get_eval_config_mpc()
            save_dir = os.path.join(experiment_save_root, 'eval/mpc/transporter_3D', model_name,
                                    get_current_YYYY_MM_DD_hh_mm_ss_ms())

            eval_mpc_data = transporter_utils.evaluate_mpc_z_state(model_dir=train_dir,
                                                                   config_planner_mpc=config_planner_mpc,
                                                                   save_dir=save_dir,
                                                                   planner_type=planner_type,
                                                                   env_config=env_config,
                                                                   strict=False,
                                                                   generate_initial_condition_func=create_initial_condition_func(),
                                                                   )

    # transporter_2D_z
    if False:
        TRAIN = True
        EVAL_MPC = True

        train_dir = None

        config = load_config()
        config['train'] = config['train_dynamics']
        config['dataset']['precomputed_data_root'] = precomputed_data_root
        config['dataset']['dataset_root'] = dataset_root
        config['train']['n_history'] = 2
        config['train']['random_seed'] = 1

        config['dynamics_net']['model_type'] = 'mlp_weight_matrix'
        config['freeze_keypoint_weights'] = False

        # make it 2D keypoints
        config['dataset']['visual_observation_function']['type'] = 'transporter_keypoints_2D'
        config['dataset']['visual_observation_function']['camera_name'] = dataset_paths['main_camera_name']
        config['dataset']['object_state_shape'] = [6, 2]
        config['dataset']['state_dim'] = 6 * 2 + 3

        suffix = "_transporter_2D_z_n_his_2"
        model_name = get_current_YYYY_MM_DD_hh_mm_ss_ms() + suffix

        if TRAIN:
            train_dir = os.path.join(experiment_save_root, 'trained_models/dynamics/transporter_3D/',
                                     model_name)

            os.makedirs(train_dir)

            visual_observation_function = PrecomputedVisualObservationFunctionFactory.function_from_config(config)

            train_dynamics_weight_matrix(config=config,
                                         train_dir=train_dir,
                                         multi_episode_dict=multi_episode_dict,
                                         visual_observation_function=visual_observation_function,
                                         )

        if EVAL_MPC:

            config_planner_mpc = get_eval_config_mpc()
            save_dir = os.path.join(experiment_save_root, 'eval/mpc/transporter_3D', model_name,
                                    get_current_YYYY_MM_DD_hh_mm_ss_ms())

            eval_mpc_data = transporter_utils.evaluate_mpc_z_state(model_dir=train_dir,
                                                                   config_planner_mpc=config_planner_mpc,
                                                                   save_dir=save_dir,
                                                                   planner_type=planner_type,
                                                                   env_config=env_config,
                                                                   generate_initial_condition_func=create_initial_condition_func(),
                                                                   )


def eval_DD_3D(dataset_name,
               planner_type="mppi",
               ):
    """
    Runs mpc evaluation for each trained model in the directory
    :param dataset_name:
    :type dataset_name:
    :return:
    :rtype:
    """

    dataset_paths = get_dataset_paths(dataset_name)
    env_config = dataset_paths['config']

    from key_dynam.experiments.drake_pusher_slider import DD_utils

    experiment_save_root = get_experiment_save_root(dataset_name=dataset_name)

    train_root_dir = os.path.join(experiment_save_root, 'trained_models/dynamics/DD_3D')

    model_names = ["2020-06-17-00-00-53-515616_DD_3D_all_z_n_his_2"]
    # model_names = sorted(os.listdir(train_root_dir))
    for model_name in model_names:

        print("model_name", model_name)
        train_dir = os.path.join(train_root_dir, model_name)

        if not os.path.isdir(train_dir):
            continue

        print("------ EVAL: %s ----------------" % (model_name))

        config_planner_mpc = get_eval_config_mpc()
        save_dir = os.path.join(experiment_save_root, 'eval/mpc/DD_3D', model_name,
                                get_current_YYYY_MM_DD_hh_mm_ss_ms())

        eval_mpc_data = DD_utils.evaluate_mpc_z_state(model_dir=train_dir,
                                                      config_planner_mpc=config_planner_mpc,
                                                      save_dir=save_dir,
                                                      planner_type=planner_type,
                                                      env_config=env_config,
                                                      strict=False,
                                                      generate_initial_condition_func=create_initial_condition_func(),

                                                      )


def eval_GT_3D(dataset_name,
               planner_type="mppi",
               ):
    """
    Runs mpc evaluation for each trained model in the directory
    :param dataset_name:
    :type dataset_name:
    :return:
    :rtype:
    """

    from key_dynam.experiments.drake_pusher_slider import GT_3D_utils

    dataset_paths = get_dataset_paths(dataset_name)
    env_config = dataset_paths['config']

    from key_dynam.experiments.drake_pusher_slider import DD_utils

    experiment_save_root = get_experiment_save_root(dataset_name=dataset_name)

    train_root_dir = os.path.join(experiment_save_root, 'trained_models/dynamics/GT_3D')

    for model_name in sorted(os.listdir(train_root_dir)):

        print("model_name", model_name)
        train_dir = os.path.join(train_root_dir, model_name)

        if not os.path.isdir(train_dir):
            continue

        print("------ EVAL: %s ----------------" % (model_name))

        config_planner_mpc = get_eval_config_mpc()
        save_dir = os.path.join(experiment_save_root, 'eval/mpc/GT_3D', model_name,
                                get_current_YYYY_MM_DD_hh_mm_ss_ms())

        eval_mpc_data = GT_3D_utils.evaluate_mpc_z_state(model_dir=train_dir,
                                                         config_planner_mpc=config_planner_mpc,
                                                         save_dir=save_dir,
                                                         planner_type=planner_type,
                                                         env_config=env_config,
                                                         strict=False,
                                                         generate_initial_condition_func=create_initial_condition_func(),

                                                         )


def train_transporter_vision(dataset_name=None,
                             ):
    from key_dynam.transporter.train_transporter import train_transporter
    def load_config():
        config_tmp = load_yaml(
            os.path.join(get_project_root(), 'experiments/drake_pusher_slider/config_transporter_3D.yaml'))

        config_tmp['train'] = config_tmp['train_transporter']
        config_tmp['dataset'] = config_tmp['dataset_transporter']
        return config_tmp

    dataset_paths = get_dataset_paths(dataset_name)
    dataset_root = dataset_paths['dataset_root']
    dataset_name = dataset_paths['dataset_name']

    multi_episode_dict = DCDrakeSimEpisodeReader.load_dataset(dataset_root=dataset_root)

    experiment_save_root = get_experiment_save_root(dataset_name)

    # standard
    if True:
        TRAIN = True
        EVALUATE = True
        config = load_config()
        config['perception']['camera_name'] = dataset_paths['main_camera_name']
        config['perception']['dataset_name'] = dataset_name

        model_name = "transporter_standard_%s" % (get_current_YYYY_MM_DD_hh_mm_ss_ms())
        train_dir = os.path.join(experiment_save_root,
                                 'trained_models/perception/transporter',
                                 model_name)

        ckp_dir = os.path.join(train_dir, 'train_nKp%d_invStd%.1f' % (
            config['perception']['n_kp'], config['perception']['inv_std']))

        if TRAIN:
            train_transporter(config=config,
                              train_dir=train_dir,
                              ckp_dir=ckp_dir,
                              multi_episode_dict=multi_episode_dict)

        if EVALUATE:
            pass


def run_precompute_transporter_keypoints(dataset_name):
    from key_dynam.transporter.precompute_transporter_keypoints import precompute_transporter_keypoints

    dataset_paths = get_dataset_paths(dataset_name)
    dataset_root = dataset_paths['dataset_root']
    dataset_name = dataset_paths['dataset_name']
    env_config = dataset_paths['config']

    transporter_model_chkpt = dataset_paths['transporter_model_chkpt']

    multi_episode_dict = DCDrakeSimEpisodeReader.load_dataset(dataset_root=dataset_root)

    if True:

        model_data = Transporter.load_model_from_checkpoint(model_chkpt_file=transporter_model_chkpt)
        model_train_dir = model_data['train_dir']
        model_kp = model_data['model']
        camera_names = [model_data['config']['perception']['camera_name']]

        output_dir = os.path.join(model_train_dir,
                                  'precomputed_vision_data/transporter_keypoints/dataset_%s/' % (dataset_name))

        print("output_dir:", output_dir)

        if os.path.exists(output_dir):
            print("output dir already exists, do you want to continue and overwrite (yes/no)")
            choice = input().lower()

            if choice != "yes":
                quit()

            shutil.rmtree(output_dir)

        os.makedirs(output_dir)

        print("\n\n---------Precomputing Transporter Keypoints-----------")
        precompute_transporter_keypoints(multi_episode_dict,
                                         model_kp=model_kp,
                                         output_dir=output_dir,
                                         batch_size=10,
                                         num_workers=20,
                                         camera_names=camera_names,
                                         model_file=transporter_model_chkpt,
                                         )


def eval_transporter(dataset_name,
                     planner_type="mppi",
                     ):
    """
    Runs mpc evaluation for each trained model in the directory
    :param dataset_name:
    :type dataset_name:
    :return:
    :rtype:
    """

    from key_dynam.experiments.drake_pusher_slider import transporter_utils

    dataset_paths = get_dataset_paths(dataset_name)
    env_config = dataset_paths['config']

    from key_dynam.experiments.drake_pusher_slider import DD_utils

    experiment_save_root = get_experiment_save_root(dataset_name=dataset_name)

    train_root_dir = os.path.join(experiment_save_root, 'trained_models/dynamics/transporter_3D')

    models_to_eval = ["2020-06-16-23-58-03-947526_transporter_2D_n_his_2"]
    # models_to_eval = sorted(os.listdir(train_root_dir))

    for model_name in models_to_eval:

        print("model_name", model_name)
        train_dir = os.path.join(train_root_dir, model_name)

        if not os.path.isdir(train_dir):
            continue

        print("------ EVAL: %s ----------------" % (model_name))

        config_planner_mpc = get_eval_config_mpc()
        save_dir = os.path.join(experiment_save_root, 'eval/mpc/transporter_3D', model_name,
                                get_current_YYYY_MM_DD_hh_mm_ss_ms())

        eval_mpc_data = transporter_utils.evaluate_mpc_z_state(model_dir=train_dir,
                                                               config_planner_mpc=config_planner_mpc,
                                                               save_dir=save_dir,
                                                               planner_type=planner_type,
                                                               env_config=env_config,
                                                               strict=False,
                                                               generate_initial_condition_func=create_initial_condition_func(),

                                                               )


def train_spatial_autoencoder_vision(dataset_name):
    dataset_paths = get_dataset_paths(dataset_name)
    dataset_root = dataset_paths['dataset_root']
    dataset_name = dataset_paths['dataset_name']

    multi_episode_dict = DCDrakeSimEpisodeReader.load_dataset(dataset_root=dataset_root)

    config_file = os.path.join(get_project_root(), 'experiments/drake_pusher_slider/config_spatial_autoencoder.yaml')
    config = load_yaml(config_file)

    config['train'] = config['train_autoencoder']

    model_name = "%s_size_60_80" % (get_current_YYYY_MM_DD_hh_mm_ss_ms())
    train_dir = os.path.join(get_experiment_save_root(dataset_name), 'trained_models/perception/spatial_autoencoder',
                             model_name)

    train_autoencoder(config=config,
                      train_dir=train_dir,
                      multi_episode_dict=multi_episode_dict,
                      type="SpatialAutoencoder")


def train_conv_autoencoder_vision(dataset_name):
    dataset_paths = get_dataset_paths(dataset_name)
    dataset_root = dataset_paths['dataset_root']
    dataset_name = dataset_paths['dataset_name']

    multi_episode_dict = DCDrakeSimEpisodeReader.load_dataset(dataset_root=dataset_root)

    def load_config():
        config_file = os.path.join(get_project_root(), 'experiments/drake_pusher_slider/config_autoencoder.yaml')
        config = load_yaml(config_file)
        config['train'] = config['train_autoencoder']

        return config

    # z_dim 8, standard
    if False:
        model_name = "%s_z_dim_8" % (get_current_YYYY_MM_DD_hh_mm_ss_ms())
        config = load_config()
        train_dir = os.path.join(get_experiment_save_root(dataset_name), 'trained_models/perception/autoencoder',
                                 model_name)

        train_autoencoder(config=config,
                          train_dir=train_dir,
                          multi_episode_dict=multi_episode_dict,
                          type="ConvolutionalAutoencoder")

    # z_dim 16
    if True:
        z_dim = 16
        config = load_config()
        config['perception']['z_dim'] = z_dim
        model_name = "%s_z_dim_%d" % (get_current_YYYY_MM_DD_hh_mm_ss_ms(), z_dim)
        train_dir = os.path.join(get_experiment_save_root(dataset_name), 'trained_models/perception/autoencoder',
                                 model_name)

        train_autoencoder(config=config,
                          train_dir=train_dir,
                          multi_episode_dict=multi_episode_dict,
                          type="ConvolutionalAutoencoder")

    # z_dim 32
    if False:
        z_dim = 32
        config = load_config()
        config['perception']['z_dim'] = z_dim
        model_name = "%s_z_dim_%d" % (get_current_YYYY_MM_DD_hh_mm_ss_ms(), z_dim)

        train_dir = os.path.join(get_experiment_save_root(dataset_name), 'trained_models/perception/autoencoder',
                                 model_name)

        train_autoencoder(config=config,
                          train_dir=train_dir,
                          multi_episode_dict=multi_episode_dict,
                          type="ConvolutionalAutoencoder")


def train_conv_autoencoder_dynamics(dataset_name):
    from key_dynam.training.train_autoencoder_dynamics import train_dynamics
    from key_dynam.experiments.drake_pusher_slider import autoencoder_utils

    dataset_paths = get_dataset_paths(dataset_name)
    dataset_root = dataset_paths['dataset_root']
    dataset_name = dataset_paths['dataset_name']
    env_config = dataset_paths['config']

    multi_episode_dict = DrakeSimEpisodeReader.load_dataset(dataset_root=dataset_root)
    experiment_save_root = get_experiment_save_root(dataset_name)

    def load_config():
        config_file = os.path.join(get_project_root(), 'experiments/drake_pusher_slider/config_autoencoder.yaml')
        config = load_yaml(config_file)
        config['train'] = config['train_dynamics']

        return config

    # z_dim 64
    if True:

        TRAIN = True
        EVAL_MPC = True

        z_dim = 64
        config = load_config()
        config['perception']['z_dim'] = z_dim
        config['perception']['camera_name'] = dataset_paths['main_camera_name']

        config['dataset']['state_dim'] = z_dim + 3
        config['dataset']['object_state_shape'] = [1, z_dim]

        visual_observation_function = AutoencoderImagePreprocessFunctionFactory.convolutional_autoencoder_from_episode(
            config)

        model_name = "%s_z_dim_%d_n_his_1" % (get_current_YYYY_MM_DD_hh_mm_ss_ms(), z_dim)
        train_dir = os.path.join(get_experiment_save_root(dataset_name), 'trained_models/dynamics/autoencoder',
                                 model_name)

        if TRAIN:
            os.makedirs(train_dir)

            train_dynamics(config=config,
                           train_dir=train_dir,
                           multi_episode_dict=multi_episode_dict,
                           visual_observation_function=visual_observation_function,
                           )

        if EVAL_MPC:


            initial_condition_func = create_initial_condition_func()

            planner_type = "mppi"
            save_dir = os.path.join(experiment_save_root,
                                    'eval/mpc/autoencoder', model_name, planner_type,
                                    get_current_YYYY_MM_DD_hh_mm_ss_ms())

            autoencoder_utils.evaluate_mpc_z_state(train_dir=train_dir,
                                                   config_planner_mpc=get_eval_config_mpc(),
                                                   save_dir=save_dir,
                                                   planner_type=planner_type,
                                                   env_config=env_config,
                                                   generate_initial_condition_func=initial_condition_func)

            planner_type = "random_shooting"
            save_dir = os.path.join(experiment_save_root,
                                    'eval/mpc/autoencoder', model_name, planner_type,
                                    get_current_YYYY_MM_DD_hh_mm_ss_ms())

            autoencoder_utils.evaluate_mpc_z_state(train_dir=train_dir,
                                                   config_planner_mpc=get_eval_config_mpc(),
                                                   save_dir=save_dir,
                                                   planner_type=planner_type,
                                                   env_config=env_config,
                                                   generate_initial_condition_func=initial_condition_func)

    # z_dim 16
    if False:

        TRAIN = False
        EVAL_MPC = True

        z_dim = 16
        config = load_config()
        config['train']['n_history'] = 2
        config['perception']['z_dim'] = z_dim
        config['perception']['camera_name'] = dataset_paths['main_camera_name']

        visual_observation_function = AutoencoderImagePreprocessFunctionFactory.convolutional_autoencoder_from_episode(
            config)

        model_name = "%s_z_dim_%d_n_his_2" % (get_current_YYYY_MM_DD_hh_mm_ss_ms(), z_dim)
        train_dir = os.path.join(get_experiment_save_root(dataset_name), 'trained_models/dynamics/autoencoder',
                                 model_name)

        if TRAIN:
            os.makedirs(train_dir)

            train_dynamics(config=config,
                           train_dir=train_dir,
                           multi_episode_dict=multi_episode_dict,
                           visual_observation_function=visual_observation_function,
                           )

        if EVAL_MPC:
            initial_condition_func = create_initial_condition_func()

            planner_type = "mppi"
            save_dir = os.path.join(experiment_save_root,
                                    'eval/mpc/autoencoder', model_name, planner_type,
                                    get_current_YYYY_MM_DD_hh_mm_ss_ms())

            autoencoder_utils.evaluate_mpc_z_state(train_dir=train_dir,
                                                   config_planner_mpc=get_eval_config_mpc(),
                                                   save_dir=save_dir,
                                                   planner_type=planner_type,
                                                   env_config=env_config,
                                                   generate_initial_condition_func=initial_condition_func,
                                                   env_type="DrakePusherSliderEnv")

            planner_type = "random_shooting"
            save_dir = os.path.join(experiment_save_root,
                                    'eval/mpc/autoencoder', model_name, planner_type,
                                    get_current_YYYY_MM_DD_hh_mm_ss_ms())

            autoencoder_utils.evaluate_mpc_z_state(train_dir=train_dir,
                                                   config_planner_mpc=get_eval_config_mpc(),
                                                   save_dir=save_dir,
                                                   planner_type=planner_type,
                                                   env_config=env_config,
                                                   generate_initial_condition_func=initial_condition_func,
                                                   env_type="DrakePusherSliderEnv")

    # test, with **only** the reconstruction loss
    if False:
        TRAIN = True
        EVAL_MPC = True

        z_dim = 16
        config = load_config()
        config['perception']['z_dim'] = z_dim
        config['perception']['camera_name'] = dataset_paths['main_camera_name']

        config['loss_function'] = dict()
        config['loss_function']['l2_recon'] = {'enabled': True, 'weight': 1.0}

        visual_observation_function = AutoencoderImagePreprocessFunctionFactory.convolutional_autoencoder_from_episode(
            config)

        model_name = "%s_z_dim_%d_l2_recon_only" % (get_current_YYYY_MM_DD_hh_mm_ss_ms(), z_dim)
        train_dir = os.path.join(get_experiment_save_root(dataset_name), 'trained_models/dynamics/autoencoder',
                                 model_name)

        if TRAIN:
            os.makedirs(train_dir)

            train_dynamics(config=config,
                           train_dir=train_dir,
                           multi_episode_dict=multi_episode_dict,
                           visual_observation_function=visual_observation_function,
                           )


def train_dense_descriptor_vision(dataset_name):
    def load_config():
        config = load_yaml(os.path.join(get_project_root(), "experiments/drake_pusher_slider/integral_heatmap_3d.yaml"))
        config['train']['train_valid_ratio'] = TRAIN_VALID_RATIO
        return config

    # standard
    if False:
        model_name = "standard_%s" % (get_current_YYYY_MM_DD_hh_mm_ss_ms())
        output_dir = os.path.join(get_experiment_save_root(dataset_name), 'trained_models/perception/dense_descriptors',
                                  model_name)

        dataset_paths = get_dataset_paths(dataset_name)
        dataset_root = dataset_paths['dataset_root']
        multi_episode_dict = DCDrakeSimEpisodeReader.load_dataset(dataset_root)

        config = load_config()
        config['dataset']['data_augmentation'] = True
        config['dataset']['camera_names'] = dataset_paths['dense_descriptor_camera_list']

        train_dense_descriptors(config,
                                train_dir=output_dir,
                                multi_episode_dict=multi_episode_dict,
                                verbose=False)

    # data augmentation
    if True:
        model_name = "data_aug_%s" % (get_current_YYYY_MM_DD_hh_mm_ss_ms())
        output_dir = os.path.join(get_experiment_save_root(dataset_name),
                                  'trained_models/perception/dense_descriptors', model_name)

        dataset_paths = get_dataset_paths(dataset_name)
        dataset_root = dataset_paths['dataset_root']
        multi_episode_dict = DCDrakeSimEpisodeReader.load_dataset(dataset_root)

        config = load_config()
        config['dataset']['data_augmentation'] = True
        config['dataset']['camera_names'] = dataset_paths['dense_descriptor_camera_list']

        train_dense_descriptors(config,
                                train_dir=output_dir,
                                multi_episode_dict=multi_episode_dict,
                                verbose=False)


def precompute_dense_descriptor_keypoints(dataset_name):
    dataset_paths = get_dataset_paths(dataset_name=dataset_name)
    dataset_root = dataset_paths['dataset_root']
    model_file = dataset_paths['dense_descriptor_model_chkpt']
    model_train_dir = os.path.dirname(model_file)

    episode_name = None
    episode_idx = None

    if dataset_name == "box_push_1000_top_down":
        episode_name = "2020-06-14-21-33-54-637244_idx_817"
        episode_idx = 17
    elif dataset_name == "box_push_1000_angled":
        # episode_name = "2020-06-14-21-29-56-641339_idx_704"
        # episode_idx = 4

        episode_name = "2020-06-14-21-35-27-457852_idx_859"
        episode_idx = 17

    print("model_train_dir", model_train_dir)
    print("model_file", model_file)
    model = torch.load(model_file)
    model = model.cuda()
    model = model.eval()

    multi_episode_dict = DCDrakeSimEpisodeReader.load_dataset(dataset_root, max_num_episodes=None)
    output_dir = os.path.join(model_train_dir,
                              'precomputed_vision_data/descriptor_keypoints/dataset_%s/' % (dataset_name))

    run_precompute_descriptors_pipeline(multi_episode_dict,
                                        model=model,
                                        model_file=model_file,
                                        output_dir=output_dir,
                                        episode_name=episode_name,
                                        episode_idx=episode_idx,
                                        camera_name=dataset_paths['main_camera_name'],
                                        visualize=True,
                                        K=5,
                                        position_diff_threshold=22,
                                        )


def multiprocess_main(dataset_name):
    # dataset_name = "2020-04-20-14-58-21-418302_T_aug_random_velocity_1000"
    # dataset_name = "2020-04-23-20-45-12-697915_T_aug_random_velocity_1000_angled_cam"

    types_to_train = [GT_3D, DD_3D, transporter_3D]
    process_list = []

    # launch all the processes
    for f in types_to_train:
        p = Process(target=f, args=(dataset_name,))
        p.start()
        process_list.append(p)

    # join all the processes
    for p in process_list:
        p.join()


def multiprocess_eval(dataset_name):
    types_to_train = [eval_GT_3D, eval_DD_3D, eval_transporter]
    process_list = []

    # launch all the processes
    for f in types_to_train:
        p = Process(target=f, args=(dataset_name,))
        p.start()
        process_list.append(p)

    # join all the processes
    for p in process_list:
        p.join()


def single_process_main():
    # dataset_name = "2020-04-20-14-58-21-418302_T_aug_random_velocity_1000"
    # dataset_name = "2020-04-23-20-45-12-697915_T_aug_random_velocity_1000_angled_cam"

    # transporter_3D(dataset_name=dataset_name)
    # DD_3D(dataset_name=dataset_name)

    # DD_3D(dataset_name="2020-04-20-14-58-21-418302_T_aug_random_velocity_1000")
    DD_3D(dataset_name="2020-04-23-20-45-12-697915_T_aug_random_velocity_1000_angled_cam")


if __name__ == "__main__":
    # set_cuda_visible_devices([get_freer_gpu()])
    set_cuda_visible_devices([1])

    # top_down = "2020-04-20-14-58-21-418302_T_aug_random_velocity_1000"
    # angled = "2020-04-23-20-45-12-697915_T_aug_random_velocity_1000_angled_cam"

    top_down = "box_push_1000_top_down"
    angled = "box_push_1000_angled"

    # train_conv_autoencoder_dynamics(dataset_name=angled)

    # run_precompute_transporter_keypoints(dataset_name=angled)


    # same for both basically
    # GT_3D(dataset_name=top_down)
    DD_2D(dataset_name=top_down)

    # DD_3D(dataset_name=angled)
    # transporter_3D(dataset_name=angled)
    # transporter_3D(dataset_name=top_down)
    # eval_DD_3D(dataset_name=top_down)
    # eval_transporter(dataset_name=top_down)
    print("Finished normally")
