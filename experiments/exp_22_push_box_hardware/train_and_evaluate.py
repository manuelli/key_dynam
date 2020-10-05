import os

import pydrake

import torch

# dense_correspondence
from dense_correspondence.dataset.dynamic_spartan_episode_reader import DynamicSpartanEpisodeReader as DCDynamicSpartanEpisodeReader
from dense_correspondence.training.train_integral_heatmap_3d import train_dense_descriptors


from key_dynam.experiments.exp_22_push_box_hardware.utils import get_dataset_paths
from key_dynam.utils.utils import get_project_root, load_yaml, get_current_YYYY_MM_DD_hh_mm_ss_ms, get_data_root, \
    load_pickle
from key_dynam.utils.torch_utils import get_freer_gpu, set_cuda_visible_devices
from key_dynam.dense_correspondence.precompute_descriptors import run_precompute_descriptors_pipeline
from key_dynam.dataset.vision_function_factory import PrecomputedVisualObservationFunctionFactory
from key_dynam.dataset.dynamic_spartan_episode_reader import DynamicSpartanEpisodeReader

def load_experiment_config():
    return load_yaml(os.path.join(get_project_root(), 'exp_22_push_box_hardware/config_DD_3D.yaml'))

def get_experiment_save_root(dataset_name):
    return os.path.join(get_data_root(), 'dev/experiments/22/dataset_%s' % (dataset_name))

def train_dense_descriptor_vision(dataset_name):
    def load_config():
        config = load_yaml(os.path.join(get_project_root(), "experiments/exp_22_push_box_hardware/integral_heatmap_3d.yaml"))
        return config

    dataset_paths = get_dataset_paths(dataset_name)
    episodes_config = dataset_paths['episodes_config']
    episodes_root = dataset_paths['dataset_root']

    # data augmentation
    if True:
        model_name = "data_aug_%s" % (get_current_YYYY_MM_DD_hh_mm_ss_ms())
        output_dir = os.path.join(get_experiment_save_root(dataset_name),
                                  'trained_models/perception/dense_descriptors', model_name)

        dataset_paths = get_dataset_paths(dataset_name)
        dataset_root = dataset_paths['dataset_root']
        multi_episode_dict = DCDynamicSpartanEpisodeReader.load_dataset(config=episodes_config,
                                                                        episodes_root=episodes_root)

        config = load_config()
        config['dataset']['data_augmentation'] = True
        config['dataset']['camera_names'] = dataset_paths['dense_descriptor_camera_list']

        train_dense_descriptors(config,
                                train_dir=output_dir,
                                multi_episode_dict=multi_episode_dict,
                                verbose=False)


def DD_3D_dynamics(dataset_name):
    from key_dynam.training.train_dynamics_pusher_slider_precomputed_keypoints import train_dynamics
    from key_dynam.experiments.drake_pusher_slider import DD_utils


    def load_config():
        return load_yaml(os.path.join(get_project_root(), 'experiments/exp_22_push_box_hardware/config_DD_3D.yaml'))

    dataset_paths = get_dataset_paths(dataset_name)
    dataset_root = dataset_paths['dataset_root']
    episodes_config = dataset_paths['episodes_config']

    precomputed_vision_data_root = DD_utils.get_precomputed_data_root(dataset_name)['precomputed_data_root']

    # descriptor_keypoints_root = os.path.join(precomputed_vision_data_root, 'descriptor_keypoints')
    descriptor_keypoints_root = os.path.join(precomputed_vision_data_root, 'descriptor_keypoints')

    config = load_config()
    multi_episode_dict = DynamicSpartanEpisodeReader.load_dataset(config=config,
    episodes_config=episodes_config,
                                                                  episodes_root=dataset_paths['dataset_root'],
                                                            load_image_episode=True,
                                                            precomputed_data_root=descriptor_keypoints_root,
                                                                  max_num_episodes=None)

    experiment_save_root = get_experiment_save_root(dataset_name)

    # experiment_save_root = os.path.join(get_data_root(), 'sandbox')

    # standard
    if True:
        TRAIN = True

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

    # data_aug
    if False:
        TRAIN = True

        spatial_descriptor_data = load_pickle(os.path.join(precomputed_vision_data_root, 'spatial_descriptors.p'))
        metadata = load_pickle(os.path.join(precomputed_vision_data_root, 'metadata.p'))

        train_dir = None

        config = load_config()
        config['train']['n_history'] = 2
        config['train']['random_seed'] = 1
        config['dataset']['visual_observation_function']['camera_name'] = dataset_paths['main_camera_name']
        config['dataset']['data_augmentation']['enabled'] = True

        config['dataset']['precomputed_data_root'] = precomputed_vision_data_root
        suffix = "_DD_3D_n_his_2_T_aug"
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


def precompute_dense_descriptor_keypoints(dataset_name):
    dataset_paths = get_dataset_paths(dataset_name=dataset_name)
    dataset_root = dataset_paths['dataset_root']


    multi_episode_dict = DCDynamicSpartanEpisodeReader.load_dataset(dataset_paths['episodes_config'], dataset_paths['dataset_root'], max_num_episodes=None)

    print("len(multi_episode_dict)", len(multi_episode_dict))

    episode_name = None
    episode_idx = None

    if dataset_name == "push_box_hardware":
        episode_name = "2020-06-29-21-42-36_stationary_for_dd_sampling"
        episode_idx = 0

    # load dense correspondence model
    model_file = dataset_paths['dense_descriptor_model_chkpt']
    model_train_dir = os.path.dirname(model_file)
    print("model_train_dir", model_train_dir)
    print("model_file", model_file)
    model = torch.load(model_file)
    model = model.cuda()
    model = model.eval()

    output_dir = os.path.join(model_train_dir,
                              'precomputed_vision_data/descriptor_keypoints/dataset_%s/' % (dataset_name))

    print("output_dir", output_dir)




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

if __name__ == "__main__":
    set_cuda_visible_devices([get_freer_gpu()])
    push_box_hardware = "push_box_hardware"
    # dataset_string_pull = "push_box_string_pull"
    # train_dense_descriptor_vision(dataset_name=dataset_string_pull)
    # precompute_dense_descriptor_keypoints(dataset_name=push_box_hardware)
    DD_3D_dynamics(dataset_name=push_box_hardware)
