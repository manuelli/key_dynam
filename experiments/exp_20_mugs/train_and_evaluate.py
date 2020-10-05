import os
import functools
import pydrake
import shutil

# pdc
from dense_correspondence_manipulation.utils import utils as pdc_utils
from dense_correspondence.training.train_integral_heatmap_3d import train_dense_descriptors
from dense_correspondence_manipulation.utils.torch_utils import get_freer_gpu
from dense_correspondence_manipulation.utils.utils import set_cuda_visible_devices

# key_dynam
from key_dynam.utils.utils import get_project_root, load_yaml, get_current_YYYY_MM_DD_hh_mm_ss_ms, get_data_root, \
    load_pickle
from key_dynam.dense_correspondence.dc_drake_sim_episode_reader import DCDrakeSimEpisodeReader
from key_dynam.dataset.drake_sim_episode_reader import DrakeSimEpisodeReader
from key_dynam.dataset.vision_function_factory import PrecomputedVisualObservationFunctionFactory
import key_dynam.experiments.exp_20_mugs.utils as exp_utils
from key_dynam.experiments.exp_20_mugs.utils import get_dataset_paths
from key_dynam.experiments.exp_20_mugs import collect_episodes
from key_dynam.autoencoder.train_autoencoder import train_autoencoder
from key_dynam.transporter.models_kp import Transporter
from key_dynam.training.train_dynamics_pusher_slider_weight_matrix import \
    train_dynamics as train_dynamics_weight_matrix
from key_dynam.autoencoder.dataset import AutoencoderImagePreprocessFunctionFactory

TRAIN_VALID_RATIO = 0.85


def get_experiment_save_root(dataset_name):
    return os.path.join(get_data_root(), 'dev/experiments/20/dataset_%s' % (dataset_name))


def get_eval_config_mpc():
    return load_yaml(os.path.join(get_project_root(), 'experiments/drake_pusher_slider/eval_config.yaml'))


def create_initial_condition_func(dataset_name,
                                  dataset_config,  # config associated with that dataset
                                  ):
    T_aug_enabled = None
    randomize_sdf = None
    randomize_color = None
    upright = None

    if dataset_name == "correlle_mug-small_single_color_600":
        T_aug_enabled = True
        randomize_sdf = True
        randomize_color = False
        upright = False
    elif dataset_name == "correlle_mug-small_many_colors_600":
        T_aug_enabled = True
        randomize_sdf = True
        randomize_color = True
        upright = False
    else:
        raise ValueError("unknown dataset: %d" % (dataset_name))

    def func(N):
        initial_cond = collect_episodes.generate_initial_condition(
            config=dataset_config,
            push_length=None,
            N=N,
            T_aug_enabled=T_aug_enabled,
            randomize_sdf=randomize_sdf,
            randomize_color=randomize_color,
            upright=upright)

        return initial_cond

    return func


def train_dense_descriptor_vision(dataset_name):
    def load_config():
        config = load_yaml(os.path.join(get_project_root(), "experiments/exp_20_mugs/integral_heatmap_3d.yaml"))
        config['train']['train_valid_ratio'] = TRAIN_VALID_RATIO
        return config

    # standard
    if False:
        config = load_config()

        model_name = "standard_%s" % (get_current_YYYY_MM_DD_hh_mm_ss_ms())
        output_dir = os.path.join(get_experiment_save_root(dataset_name), 'trained_models/perception/dense_descriptors',
                                  model_name)

        dataset_paths = exp_utils.get_dataset_paths(dataset_name)
        dataset_root = dataset_paths['dataset_root']
        multi_episode_dict = DCDrakeSimEpisodeReader.load_dataset(dataset_root)

        train_dense_descriptors(config,
                                train_dir=output_dir,
                                multi_episode_dict=multi_episode_dict,
                                verbose=False)

    # data augmentation
    if True:
        config = load_config()

        config['dataset']['data_augmentation'] = True

        model_name = "data_aug_%s" % (get_current_YYYY_MM_DD_hh_mm_ss_ms())
        output_dir = os.path.join(get_experiment_save_root(dataset_name),
                                  'trained_models/perception/dense_descriptors', model_name)

        dataset_paths = exp_utils.get_dataset_paths(dataset_name)
        dataset_root = dataset_paths['dataset_root']
        multi_episode_dict = DCDrakeSimEpisodeReader.load_dataset(dataset_root)

        # print("len(multi_episode_dict)", len(multi_episode_dict))
        # print("config['train']['train_valid_ratio']", config['train']['train_valid_ratio'])
        # quit()

        train_dense_descriptors(config,
                                train_dir=output_dir,
                                multi_episode_dict=multi_episode_dict,
                                verbose=False)

    # single color dataset
    if False:
        config = load_config()

        config['dataset']['data_augmentation'] = False

        model_name = "standard_%s" % (get_current_YYYY_MM_DD_hh_mm_ss_ms())
        output_dir = os.path.join(get_experiment_save_root(dataset_name),
                                  'trained_models/perception/dense_descriptors', model_name)

        dataset_paths = exp_utils.get_dataset_paths(dataset_name)
        dataset_root = dataset_paths['dataset_root']
        multi_episode_dict = DCDrakeSimEpisodeReader.load_dataset(dataset_root)

        print("len(multi_episode_dict)", len(multi_episode_dict))

        train_dense_descriptors(config,
                                train_dir=output_dir,
                                multi_episode_dict=multi_episode_dict,
                                verbose=False)


def DD_3D_dynamics(dataset_name):
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
    dataset_config = dataset_paths['config']

    # folder where spatial descriptors etc. are
    precomputed_vision_data_root = DD_utils.get_precomputed_data_root(dataset_name)['precomputed_data_root']

    multi_episode_dict = DrakeSimEpisodeReader.load_dataset(dataset_root=dataset_root,
                                                            load_image_data=True,
                                                            precomputed_data_root=os.path.join(
                                                                precomputed_vision_data_root, 'descriptor_keypoints'))

    experiment_save_root = get_experiment_save_root(dataset_name)
    planner_type = "mppi"

    # experiment_save_root = os.path.join(get_data_root(), 'sandbox')

    # standard
    if False:
        TRAIN = False
        EVAL_MPC = True

        spatial_descriptor_data = load_pickle(os.path.join(precomputed_vision_data_root, 'spatial_descriptors.p'))
        metadata = load_pickle(os.path.join(precomputed_vision_data_root, 'metadata.p'))

        train_dir = None

        config = load_config()
        config['train']['n_history'] = 2
        config['train']['random_seed'] = 1
        config['dataset']['visual_observation_function']['camera_name'] = dataset_paths['main_camera_name']

        config['dataset']['precomputed_data_root'] = precomputed_vision_data_root
        suffix = "_DD_3D_spatial_n_his_2"
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
                                                          generate_initial_condition_func=create_initial_condition_func(
                                                              dataset_name, dataset_config=dataset_config),
                                                          env_type="DrakeMugsEnv"
                                                          )

    # no data aug
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

        config['dataset']['data_augmentation']['enabled'] = False

        config['dataset']['precomputed_data_root'] = precomputed_vision_data_root
        suffix = "_DD_3D_spatial_n_his_2_no_T_aug"
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
                                                          generate_initial_condition_func=create_initial_condition_func(
                                                              dataset_name,
                                                              dataset_config=dataset_config),
                                                          env_type="DrakeMugsEnv"
                                                          )

    # standard
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

        config['dataset']['data_augmentation']['enabled'] = True

        config['dataset']['precomputed_data_root'] = precomputed_vision_data_root
        suffix = "_DD_3D_sptaial_n_his_2"
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
                                                          generate_initial_condition_func=create_initial_condition_func(
                                                              dataset_name,
                                                              dataset_config=dataset_config),
                                                          env_type="DrakeMugsEnv"
                                                          )

    # weight_matrix + spatial_z + no data aug
    if False:
        TRAIN = False
        EVAL_MPC = True

        spatial_descriptor_data = load_pickle(os.path.join(precomputed_vision_data_root, 'spatial_descriptors.p'))
        metadata = load_pickle(os.path.join(precomputed_vision_data_root, 'metadata.p'))

        train_dir = None

        config = load_config()
        config['train']['n_history'] = 2
        config['train']['random_seed'] = 1
        config['train']['lr_scheduler']['enabled'] = False
        config['dataset']['visual_observation_function']['camera_name'] = dataset_paths['main_camera_name']

        config['dataset']['data_augmentation']['enabled'] = False

        config['dataset']['precomputed_data_root'] = precomputed_vision_data_root

        config['dynamics_net']['model_type'] = 'mlp_weight_matrix'

        config['freeze_keypoint_weights'] = False

        suffix = "_DD_3D_z_state_n_his_2_no_T_aug"
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
                                                          generate_initial_condition_func=create_initial_condition_func(
                                                              dataset_name,
                                                              dataset_config=dataset_config),
                                                          env_type="DrakeMugsEnv"
                                                          )

    # weight_matrix + spatial_z
    if False:
        TRAIN = True
        EVAL_MPC = True

        spatial_descriptor_data = load_pickle(os.path.join(precomputed_vision_data_root, 'spatial_descriptors.p'))
        metadata = load_pickle(os.path.join(precomputed_vision_data_root, 'metadata.p'))

        train_dir = None

        config = load_config()
        config['train']['n_history'] = 2
        config['train']['random_seed'] = 1
        config['train']['lr_scheduler']['enabled'] = False
        config['dataset']['visual_observation_function']['camera_name'] = dataset_paths['main_camera_name']

        config['dataset']['data_augmentation']['enabled'] = True

        config['dataset']['precomputed_data_root'] = precomputed_vision_data_root

        config['dynamics_net']['model_type'] = 'mlp_weight_matrix'

        config['freeze_keypoint_weights'] = False

        suffix = "_DD_3D_spatial_z_state_n_his_2"
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
                                                          generate_initial_condition_func=create_initial_condition_func(
                                                              dataset_name,
                                                              dataset_config=dataset_config),
                                                          env_type="DrakeMugsEnv"
                                                          )

    # all
    if False:
        TRAIN = False
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
                                                          generate_initial_condition_func=create_initial_condition_func(
                                                              dataset_name=dataset_name,
                                                              dataset_config=dataset_config),
                                                          env_type="DrakeMugsEnv"
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
                                                          generate_initial_condition_func=create_initial_condition_func(
                                                              dataset_name=dataset_name,
                                                              dataset_config=dataset_config),
                                                          env_type="DrakeMugsEnv"
                                                          )

    # all_z no T_aug
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
                                                          generate_initial_condition_func=create_initial_condition_func(
                                                              dataset_name=dataset_name,
                                                              dataset_config=dataset_config),
                                                          env_type="DrakeMugsEnv"
                                                          )


def train_transporter_vision(dataset_name):
    from key_dynam.transporter.train_transporter import train_transporter

    dataset_paths = get_dataset_paths(dataset_name)
    dataset_root = dataset_paths['dataset_root']
    dataset_name = dataset_paths['dataset_name']
    env_config = dataset_paths['config']

    multi_episode_dict = DCDrakeSimEpisodeReader.load_dataset(dataset_root=dataset_root)

    def load_config():
        config = load_yaml(
            os.path.join(get_project_root(), 'experiments/drake_pusher_slider/config_transporter_3D.yaml'))

        config['train_transporter']['train_valid_ratio'] = TRAIN_VALID_RATIO
        config['train_dynamics']['train_valid_ratio'] = TRAIN_VALID_RATIO

        config['train'] = config['train_transporter']
        config['dataset'] = config['dataset_transporter']
        return config

    if True:
        config = load_config()
        model_name = "transporter_%s" % (get_current_YYYY_MM_DD_hh_mm_ss_ms())
        train_dir = os.path.join(get_experiment_save_root(dataset_name), 'trained_models/perception/transporter',
                                 model_name)

        ckp_dir = os.path.join(train_dir, 'train_nKp%d_invStd%.1f' % (
            config['perception']['n_kp'], config['perception']['inv_std']))

        train_transporter(config=config,
                          train_dir=train_dir,
                          ckp_dir=ckp_dir,
                          multi_episode_dict=multi_episode_dict)


def run_precompute_transporter_keypoints(dataset_name):
    from key_dynam.transporter.precompute_transporter_keypoints import precompute_transporter_keypoints

    dataset_paths = get_dataset_paths(dataset_name)
    dataset_root = dataset_paths['dataset_root']
    dataset_name = dataset_paths['dataset_name']
    env_config = dataset_paths['config']

    multi_episode_dict = DCDrakeSimEpisodeReader.load_dataset(dataset_root=dataset_root)

    if True:

        model_data = Transporter.load_model_from_checkpoint(model_chkpt_file=model_chkpt_file)
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
                                         model_file=model_chkpt_file,
                                         )


def transporter_dynamics(dataset_name=None, debug=False):
    from key_dynam.training.train_dynamics_pusher_slider_precomputed_keypoints import train_dynamics
    from key_dynam.experiments.drake_pusher_slider import transporter_utils

    def load_config():
        config = load_yaml(
            os.path.join(get_project_root(), 'experiments/drake_pusher_slider/config_transporter_3D.yaml'))

        config['train_transporter']['train_valid_ratio'] = TRAIN_VALID_RATIO
        config['train_dynamics']['train_valid_ratio'] = TRAIN_VALID_RATIO

        return config

    dataset_paths = get_dataset_paths(dataset_name)
    dataset_root = dataset_paths['dataset_root']
    dataset_name = dataset_paths['dataset_name']
    env_config = dataset_paths['config']
    dataset_config = dataset_paths['config']

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
                                                                   generate_initial_condition_func=create_initial_condition_func(
                                                                       dataset_name, dataset_config=dataset_config),
                                                                   env_type="DrakeMugsEnv"
                                                                   )

    # transporter_3D_z
    if False:
        raise NotImplementedError
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

            visual_observation_function = PrecomputedVisualObservationFunctionFactory.function_from_config(
                config)

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
                                                                   generate_initial_condition_func=create_initial_condition_func(
                                                                       dataset_name, dataset_config=dataset_config),
                                                                   env_type="DrakeMugsEnv"
                                                                   )

    # transporter_2D_z
    if False:
        raise NotImplementedError("need to fix eval part")
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


def train_conv_autoencoder_vision(dataset_name):
    dataset_paths = get_dataset_paths(dataset_name)
    dataset_root = dataset_paths['dataset_root']
    dataset_name = dataset_paths['dataset_name']

    multi_episode_dict = DCDrakeSimEpisodeReader.load_dataset(dataset_root=dataset_root)

    def load_config():
        config_file = os.path.join(get_project_root(), 'experiments/drake_pusher_slider/config_autoencoder.yaml')
        config = load_yaml(config_file)

        config['train_autoencoder']['train_valid_ratio'] = TRAIN_VALID_RATIO
        config['train_dynamics']['train_valid_ratio'] = TRAIN_VALID_RATIO
        config['train'] = config['train_autoencoder']

        return config

    # z_dim 64
    if True:
        z_dim = 64
        config = load_config()
        config['perception']['z_dim'] = z_dim
        model_name = "%s_z_dim_%d" % (get_current_YYYY_MM_DD_hh_mm_ss_ms(), z_dim)
        train_dir = os.path.join(get_experiment_save_root(dataset_name), 'trained_models/perception/autoencoder',
                                 model_name)

        train_autoencoder(config=config,
                          train_dir=train_dir,
                          multi_episode_dict=multi_episode_dict,
                          type="ConvolutionalAutoencoder")

    # input image size 128
    if False:
        z_dim = 16
        config = load_config()
        config['perception']['z_dim'] = z_dim
        config['perception']['height'] = 128
        config['perception']['width'] = 128
        model_name = "%s_sz_128_z_dim_%d" % (get_current_YYYY_MM_DD_hh_mm_ss_ms(), z_dim)
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
    dataset_config = dataset_paths['config']

    multi_episode_dict = DrakeSimEpisodeReader.load_dataset(dataset_root=dataset_root)
    experiment_save_root = get_experiment_save_root(dataset_name)

    def load_config():
        config_file = os.path.join(get_project_root(), 'experiments/drake_pusher_slider/config_autoencoder.yaml')
        config = load_yaml(config_file)

        config['train_autoencoder']['train_valid_ratio'] = TRAIN_VALID_RATIO
        config['train_dynamics']['train_valid_ratio'] = TRAIN_VALID_RATIO
        config['train'] = config['train_dynamics']

        return config

    # z_dim 64
    if True:

        TRAIN = False
        EVAL_MPC = True

        z_dim = 64
        config = load_config()
        config['perception']['z_dim'] = z_dim
        config['perception']['camera_name'] = dataset_paths['main_camera_name']

        config['dataset']['state_dim'] = z_dim + 3
        config['dataset']['object_state_shape'] = [1, z_dim]
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

            model_name = "2020-06-12-03-24-33-453446_z_dim_64_n_his_1"

            planner_type = "mppi"
            save_dir = os.path.join(experiment_save_root,
                                    'eval/mpc/autoencoder', model_name, planner_type,
                                    get_current_YYYY_MM_DD_hh_mm_ss_ms())

            autoencoder_utils.evaluate_mpc_z_state(train_dir=train_dir,
                                                   config_planner_mpc=get_eval_config_mpc(),
                                                   save_dir=save_dir,
                                                   planner_type=planner_type,
                                                   env_config=env_config,
                                                   generate_initial_condition_func=create_initial_condition_func(
                                                       dataset_name, dataset_config=dataset_config),
                                                   env_type="DrakeMugsEnv")

            planner_type = "random_shooting"
            save_dir = os.path.join(experiment_save_root,
                                    'eval/mpc/autoencoder', model_name, planner_type,
                                    get_current_YYYY_MM_DD_hh_mm_ss_ms())

            autoencoder_utils.evaluate_mpc_z_state(train_dir=train_dir,
                                                   config_planner_mpc=get_eval_config_mpc(),
                                                   save_dir=save_dir,
                                                   planner_type=planner_type,
                                                   env_config=env_config,
                                                   generate_initial_condition_func=create_initial_condition_func(
                                                       dataset_name, dataset_config=dataset_config),
                                                   env_type="DrakeMugsEnv")


if __name__ == "__main__":
    # dataset_name = "mugs_random_colors_1000"
    # dataset_name = "mugs_correlle_mug-small_1000"
    # dataset_name = "correlle_mug-small_single_color_600"
    # dataset_name = "single_corelle_mug_600"
    # dataset_name = "correlle_mug-small_many_colors_600"
    # train_dense_descriptor_vision(dataset_name=dataset_name)

    set_cuda_visible_devices([get_freer_gpu()])
    # set_cuda_visible_devices([1])
    dataset_name = "correlle_mug-small_many_colors_600"
    DD_3D_dynamics(dataset_name=dataset_name)
    # train_conv_autoencoder_vision(dataset_name)
    # train_transporter_vision(dataset_name)
    # run_precompute_transporter_keypoints(dataset_name)
    # transporter_dynamics(dataset_name)
    # train_conv_autoencoder_dynamics(dataset_name=dataset_name)
    # transporter_dynamics(dataset_name=dataset_name)
