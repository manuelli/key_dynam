import os
import functools
import shutil
import numpy as np
import pydrake
import torch
import copy as copy
from multiprocessing import Process

from key_dynam.utils.utils import get_project_root, load_yaml, get_current_YYYY_MM_DD_hh_mm_ss_ms, get_data_root, \
    load_pickle
from key_dynam.experiments.exp_18_box_on_side.utils import get_dataset_paths, TRAIN_VALID_RATIO
from key_dynam.experiments.exp_18_box_on_side import collect_episodes
from key_dynam.dataset.drake_sim_episode_reader import DrakeSimEpisodeReader
from key_dynam.dataset.vision_function_factory import PrecomputedVisualObservationFunctionFactory
from key_dynam.utils.torch_utils import get_freer_gpu, set_cuda_visible_devices
from key_dynam.dense_correspondence.dc_drake_sim_episode_reader import DCDrakeSimEpisodeReader
from key_dynam.transporter.models_kp import Transporter
from key_dynam.transporter.eval_transporter import eval_transporter as eval_transporter_vision
from key_dynam.transporter.precompute_transporter_keypoints import precompute_transporter_keypoints
from key_dynam.models.model_builder import build_dynamics_model
from key_dynam.autoencoder.autoencoder_models import ConvolutionalAutoencoder
from key_dynam.training.train_dynamics_pusher_slider_weight_matrix import \
    train_dynamics as train_dynamics_weight_matrix


def get_experiment_save_root(dataset_name):
    return os.path.join(get_data_root(), 'dev/experiments/drake_pusher_slider_box_on_side/dataset_%s' % (dataset_name))


def get_eval_config_mpc():
    return load_yaml(os.path.join(get_project_root(), 'experiments/drake_pusher_slider/eval_config.yaml'))


def create_initial_condition_func():
    eval_config = get_eval_config_mpc()

    generate_initial_condition_func = functools.partial(collect_episodes.generate_initial_condition,
                                                        config=eval_config,
                                                        T_aug_enabled=True)

    return generate_initial_condition_func


def GT_3D(dataset_name):
    from key_dynam.training.train_dynamics_pusher_slider_precomputed_keypoints import train_dynamics
    from key_dynam.experiments.drake_pusher_slider import GT_3D_utils

    def load_config():
        config_tmp = load_yaml(os.path.join(get_project_root(), 'experiments/drake_pusher_slider/config_GT_3D.yaml'))
        config_tmp['train']['train_valid_ratio'] = TRAIN_VALID_RATIO
        return config_tmp

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
        TRAIN = False
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
            config_planner_mpc = get_eval_config_mpc()

            save_dir = os.path.join(experiment_save_root, 'eval/mpc/GT_3D', model_name,
                                    get_current_YYYY_MM_DD_hh_mm_ss_ms())

            eval_mpc_data = GT_3D_utils.evaluate_mpc(model_dir=train_dir,
                                                     config_planner_mpc=config_planner_mpc,
                                                     save_dir=save_dir,
                                                     planner_type=planner_type,
                                                     env_config=env_config,
                                                     strict=False,
                                                     generate_initial_condition_func=create_initial_condition_func(),
                                                     )

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

            eval_mpc_data = GT_3D_utils.evaluate_mpc(model_dir=train_dir,
                                                     config_planner_mpc=config_planner_mpc,
                                                     save_dir=save_dir,
                                                     planner_type=planner_type,
                                                     env_config=env_config,
                                                     strict=False,
                                                     generate_initial_condition_func=create_initial_condition_func(),
                                                     )

def DD_3D(dataset_name):
    from key_dynam.training.train_dynamics_pusher_slider_precomputed_keypoints import train_dynamics
    from key_dynam.training.train_dynamics_pusher_slider_weight_matrix import \
        train_dynamics as train_dynamics_weight_matrix
    from key_dynam.experiments.drake_pusher_slider import DD_utils

    def load_config():
        config_tmp = load_yaml(os.path.join(get_project_root(), 'experiments/drake_pusher_slider/config_DD_3D.yaml'))
        config_tmp['train']['train_valid_ratio'] = TRAIN_VALID_RATIO

        K = 4
        config_tmp['dataset']['state_dim'] = 3 * K + 3
        config_tmp['dataset']['object_state_shape'] = [K, 3]
        return config_tmp

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

    # standard (SDS)
    if False:
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

    # standard (SDS) no T_aug
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
        config['dataset']['data_augmentation']['enabled'] = False

        config['dataset']['precomputed_data_root'] = precomputed_vision_data_root
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

            eval_mpc_data = DD_utils.evaluate_mpc_z_state(model_dir=train_dir,
                                                          config_planner_mpc=config_planner_mpc,
                                                          save_dir=save_dir,
                                                          planner_type=planner_type,
                                                          env_config=env_config,
                                                          generate_initial_condition_func=create_initial_condition_func(),
                                                          )

    # spatial_z
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

    # spatial_z_init_uniform
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
        config['dynamics_net']['freeze_keypoint_weights'] = False
        config['dynamics_net']['weight_matrix_init'] = 'uniform'

        suffix = "_DD_3D_spatial_z_init_uniform_n_his_2"
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

    # spatial_z_sparse
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
        config['dynamics_net']['freeze_keypoint_weights'] = False

        config['loss_function']['weight_matrix_sparsity']['enabled'] = True

        suffix = "_DD_3D_spatial_z_sparse_n_his_2"
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

    # spatial_z_sparse_larger_weight
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
        config['dynamics_net']['freeze_keypoint_weights'] = False

        config['loss_function']['weight_matrix_sparsity']['enabled'] = True
        config['loss_function']['weight_matrix_sparsity']['weight'] = 0.01

        suffix = "_DD_3D_spatial_z_sparse_big_weight_n_his_2"
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
    if False:
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

    # all_z_no_T_aug
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
        config['dataset']['data_augmentation']['enabled'] = False

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

    # all_spatial_z_init_uniform
    if False:
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
        config['dynamics_net']['freeze_keypoint_weights'] = False
        config['dynamics_net']['weight_matrix_init'] = 'uniform'

        suffix = "_DD_3D_all_z_init_uniform_n_his_2"
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

    # all_spatial_z_sparse
    if False:
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

        config['loss_function']['weight_matrix_sparsity']['enabled'] = True

        config['train']['n_history'] = 2
        config['train']['random_seed'] = 1
        config['dataset']['visual_observation_function']['camera_name'] = dataset_paths['main_camera_name']

        config['dataset']['precomputed_data_root'] = precomputed_vision_data_root

        config['dynamics_net']['model_type'] = 'mlp_weight_matrix'
        config['freeze_keypoint_weights'] = False

        suffix = "_DD_3D_all_z_sparse_n_his_2"
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

    # all_spatial_z_sparse bigger weight
    if False:
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

        config['loss_function']['weight_matrix_sparsity']['enabled'] = True
        config['loss_function']['weight_matrix_sparsity']['weight'] = 0.01

        config['train']['n_history'] = 2
        config['train']['random_seed'] = 1
        config['dataset']['visual_observation_function']['camera_name'] = dataset_paths['main_camera_name']

        config['dataset']['precomputed_data_root'] = precomputed_vision_data_root

        config['dynamics_net']['model_type'] = 'mlp_weight_matrix'
        config['freeze_keypoint_weights'] = False

        suffix = "_DD_3D_all_z_big_weight_sparse_n_his_2"
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


def transporter_3D(dataset_name=None, debug=False):
    from key_dynam.training.train_dynamics_pusher_slider_precomputed_keypoints import train_dynamics
    from key_dynam.experiments.drake_pusher_slider import transporter_utils

    def load_config():
        config_tmp = load_yaml(
            os.path.join(get_project_root(), 'experiments/drake_pusher_slider/config_transporter_3D.yaml'))
        config_tmp['train_dynamics']['train_valid_ratio'] = TRAIN_VALID_RATIO
        return config_tmp

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

    # standard
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

        if debug:
            config['train']['n_epoch'] = 10

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

            eval_mpc_data = transporter_utils.evaluate_mpc(model_dir=train_dir,
                                                           config_planner_mpc=config_planner_mpc,
                                                           save_dir=save_dir,
                                                           planner_type=planner_type,
                                                           env_config=env_config,
                                                           strict=False,
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

    # transporter_2D
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

            eval_mpc_data = transporter_utils.evaluate_mpc(model_dir=train_dir,
                                                           config_planner_mpc=config_planner_mpc,
                                                           save_dir=save_dir,
                                                           planner_type=planner_type,
                                                           env_config=env_config,
                                                           generate_initial_condition_func=create_initial_condition_func(),
                                                           )

    # transporter_2D_z
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


def train_conv_autoencoder_dynamics(dataset_name):
    from key_dynam.training.train_autoencoder_dynamics import train_dynamics
    from key_dynam.experiments.drake_pusher_slider import autoencoder_utils
    from key_dynam.autoencoder.dataset import AutoencoderImagePreprocessFunctionFactory

    dataset_paths = get_dataset_paths(dataset_name)
    dataset_root = dataset_paths['dataset_root']
    dataset_name = dataset_paths['dataset_name']
    env_config = dataset_paths['config']

    # print("dataset_root", dataset_root){}
    # quit()

    multi_episode_dict = DrakeSimEpisodeReader.load_dataset(dataset_root=dataset_root)
    experiment_save_root = get_experiment_save_root(dataset_name)

    def load_config():
        config_file = os.path.join(get_project_root(), 'experiments/drake_pusher_slider/config_autoencoder.yaml')
        config = load_yaml(config_file)
        config['train_dynamics']['train_valid_ratio'] = TRAIN_VALID_RATIO
        config['train'] = config['train_dynamics']
        return config

    # z_dim 16
    if False:

        TRAIN = True
        EVAL_MPC = True

        z_dim = 16
        config = load_config()
        config['perception']['z_dim'] = z_dim
        config['perception']['camera_name'] = dataset_paths['main_camera_name']

        config['train']['n_history'] = 1

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

    for model_name in sorted(os.listdir(train_root_dir)):

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

    for model_name in sorted(os.listdir(train_root_dir)):

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


def run_precompute_transporter_keypoints(dataset_name=None,
                                         model_chkpt_file=None):
    dataset_paths = get_dataset_paths(dataset_name)
    dataset_root = dataset_paths['dataset_root']
    dataset_name = dataset_paths['dataset_name']
    env_config = dataset_paths['config']

    multi_episode_dict = DCDrakeSimEpisodeReader.load_dataset(dataset_root)

    model_data = Transporter.load_model_from_checkpoint(model_chkpt_file=model_chkpt_file)
    model_kp = model_data['model']
    model_train_dir = model_data['train_dir']

    camera_names = [model_data['config']['perception']['camera_name']]
    output_dir = os.path.join(model_train_dir,
                              'precomputed_vision_data/transporter_keypoints/dataset_%s/' % (dataset_name))

    if os.path.exists(output_dir):
        print("output dir already exists, do you want to continue and overwrite (yes/no)")
        choice = input().lower()

        if choice != "yes":
            quit()

        shutil.rmtree(output_dir)

    os.makedirs(output_dir)

    if True:
        print("\n\n---------Precomputing Transporter Keypoints-----------")
        precompute_transporter_keypoints(multi_episode_dict,
                                         model_kp=model_kp,
                                         output_dir=output_dir,
                                         batch_size=10,
                                         num_workers=20,
                                         camera_names=camera_names,
                                         model_file=model_chkpt_file,
                                         )


# def run_evaluate_transporter(dataset_name=None,
#                              model_train_dir=None,
#                              ):
#
#
#     config = load_yaml(os.path.join(model_train_dir, 'config.yaml'))
#
#     eval_config = get_eval_config_mpc()
#     config['eval'] = eval_config['eval_transporter']
#
#     dataset_paths = get_dataset_paths(dataset_name)
#     dataset_root = dataset_paths['dataset_root']
#     dataset_name = dataset_paths['dataset_name']
#     env_config = dataset_paths['config']
#
#     multi_episode_dict = DCDrakeSimEpisodeReader.load_dataset(dataset_root)
#
#     eval_transporter(config=config,
#                      train_dir=model_train_dir,
#                      multi_episode_dict=multi_episode_dict,
#                      )


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
    set_cuda_visible_devices([get_freer_gpu()])
    # set_cuda_visible_devices([1])

    dataset_name = "dps_box_on_side_600"
    GT_3D(dataset_name=dataset_name)

    print("Finished normally")
