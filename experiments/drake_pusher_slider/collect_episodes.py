import os
import time
import numpy as np
from gym.spaces import Box
import transforms3d.derivations.eulerangles

# transforms3d
import transforms3d

from key_dynam.utils.utils import save_yaml, load_yaml, get_data_root, get_project_root, get_data_ssd_root
from key_dynam.scripts.drake_pusher_slider_episode_collector import DrakePusherSliderEpisodeCollector
from key_dynam.utils.utils import get_project_root, get_current_YYYY_MM_DD_hh_mm_ss_ms, load_yaml, random_sample_in_range
from key_dynam.dataset.episode_dataset import MultiEpisodeDataset
from key_dynam.utils import transform_utils
from key_dynam.dynamics.utils import set_seed

"""
Generates a dataset using the DrakePusherSlider environment.
"""


def sample_slider_position(T_aug=None):
    # always at the origin basically
    pos = np.array([0, 0, 0.03])

    yaw_min = np.array([0])
    yaw_max = np.array([2 * np.pi])
    yaw_sampler = Box(yaw_min, yaw_max)
    yaw = yaw_sampler.sample()

    quat = transforms3d.euler.euler2quat(0, 0, yaw)

    T_O_slider = transform_utils.transform_from_pose(pos, quat)

    T_W_slider = None
    if T_aug is not None:
        T_W_slider = T_aug @ T_O_slider
    else:
        T_W_slider = T_O_slider

    pose_dict = transform_utils.matrix_to_dict(T_W_slider)

    # note the quat/pos orderining
    q = np.concatenate((pose_dict['quaternion'], pose_dict['position']))

    return q


def sample_pusher_position_and_velocity(T_aug=None):
    low = np.array([-0.16, -0.08])
    high = np.array([-0.15, 0.08])
    q_pusher = np.array(Box(low, high).sample())

    q_pusher_3d_homog = np.array([0, 0, 0, 1.0])
    q_pusher_3d_homog[:2] = q_pusher

    print("q_pusher", q_pusher)
    print("q_pusher_3d_homog", q_pusher_3d_homog)


    # sample pusher velocity
    vel_min = np.array([0.15])
    vel_max = np.array([0.25])
    magnitude = Box(vel_min, vel_max).sample()
    v_pusher_3d = magnitude * np.array([1,0,0])


    # 1000 timesteps of pusher velocity
    N = 1000
    v_pusher_3d_seq = np.zeros([N, 3])
    for i in range(N):
        v_pusher_3d_tmp = sample_pusher_velocity_3D()

        # always moving to the right
        # v_pusher_3d = magnitude * np.array([1, 0, 0])

        v_pusher_3d_seq[i] = T_aug[:3, :3] @ v_pusher_3d_tmp

    action_seq = v_pusher_3d_seq[:, :2]


    if T_aug is not None:
        v_pusher_3d = T_aug[:3, :3] @ v_pusher_3d
        q_pusher_3d_homog = T_aug @ q_pusher_3d_homog


    v_pusher = v_pusher_3d[:2]
    q_pusher = q_pusher_3d_homog[:2]

    return {'q_pusher': q_pusher,
            'v_pusher': v_pusher,
            'action_seq': action_seq,
            }


def sample_pusher_velocity_3D():
    vel_min = 0.15
    vel_max = 0.25
    magnitude = random_sample_in_range(vel_min, vel_max)

    angle_max = np.deg2rad(30)
    angle_min = -angle_max
    angle = random_sample_in_range(angle_min, angle_max)

    vel_3d = magnitude * np.array([np.cos(angle), np.sin(angle), 0])
    return vel_3d



def generate_initial_condition(config=None,
                               T_aug_enabled=True):
    T_aug = None
    if T_aug_enabled:
        aug_config = config['dataset']['data_augmentation']
        bound = 0.25
        pos_min = np.array([-bound, -bound, 0])
        pos_max = np.array([bound, bound, 0])
        yaw_min = 0
        yaw_max = 2*np.pi
        T_aug = MultiEpisodeDataset.sample_T_aug(pos_min=pos_min,
                                                 pos_max=pos_max,
                                                 yaw_min=yaw_min,
                                                 yaw_max=yaw_max,)


    q_slider = sample_slider_position(T_aug=T_aug)
    d = sample_pusher_position_and_velocity(T_aug=T_aug)

    return {'q_slider': q_slider,
            'q_pusher': d['q_pusher'],
            'v_pusher': d['v_pusher'],
            'action_seq': d['action_seq'],
            }


def collect_episodes(config,
                     output_dir=None,
                     visualize=True,
                     debug=False):
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), 'data')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # save the config
    config_save_file = os.path.join(output_dir, 'config.yaml')
    save_yaml(config, config_save_file)

    # initialize config for DataCollector
    dc = DrakePusherSliderEpisodeCollector(config, visualize=visualize)
    num_episodes = config['dataset']['num_episodes']

    # record some metadata
    metadata = dict()
    metadata['episodes'] = dict()

    while (len(metadata['episodes']) < num_episodes):

        i = len(metadata['episodes'])

        if debug:
            input("Press Enter to continue...")

        print("\n")
        start_time = time.time()
        print("collecting episode %d of %d" % (i + 1, num_episodes))
        name = "%s_idx_%d" % (get_current_YYYY_MM_DD_hh_mm_ss_ms(), i)

        ic = generate_initial_condition(config=config,
                                        T_aug_enabled=True)

        if debug:
            print("initial condition\n", ic)

        episode = dc.collect_single_episode(visualize=visualize,
                                            episode_name=name,
                                            q_pusher=ic['q_pusher'],
                                            q_slider=ic['q_slider'],
                                            # v_pusher=ic['v_pusher'],
                                            action_seq=ic['action_seq'],
                                            )

        # potentially discard it if the object didn't move during the data collection
        if len(episode._data['trajectory']) < 10:
            print("trajectory was too short, skipping")
            continue

        obs_start = episode._data['trajectory'][5]['observation']
        obs_end = episode._data['trajectory'][-1]['observation']

        q_slider_start = obs_start['slider']['position']['translation']
        q_slider_end = obs_end['slider']['position']['translation']

        dq_slider = obs_start['slider']['position']['translation'] - obs_end['slider']['position']['translation']

        if debug:
            print("len(episode._data['trajectory'])", len(episode._data['trajectory']))
            print("q_slider_start", q_slider_start)
            print("q_slider_end", q_slider_end)
            print("dq_slider", dq_slider)
            print("np.linalg.norm(dq_slider)", np.linalg.norm(dq_slider))

        # if slider didn't move by at least 1 mm then discard this episode
        if (np.linalg.norm(dq_slider) < 0.001):  # one mm
            print("discarding episode since slider didn't move")
            continue

        print("saving to disk")
        metadata['episodes'][name] = dict()

        image_data_file = episode.save_images_to_hdf5(output_dir)
        non_image_data_file = episode.save_non_image_data_to_pickle(output_dir)

        print("output_dir:", output_dir)

        print("non_image_data.keys()", episode.non_image_data.keys())

        metadata['episodes'][name]['non_image_data_file'] = non_image_data_file
        metadata['episodes'][name]['image_data_file'] = image_data_file

        print("done saving to disk")
        elapsed = time.time() - start_time
        print("single episode took: %.2f seconds" % (elapsed))

    save_yaml(metadata, os.path.join(output_dir, 'metadata.yaml'))


def main():
    start_time = time.time()
    config = load_yaml(os.path.join(get_project_root(), 'experiments/drake_pusher_slider/env_config.yaml'))
    config['dataset']['num_episodes'] = 1000 # half for train, half for valid

    set_seed(500) # just randomly chosen

    num_episodes = config['dataset']['num_episodes']
    DATASET_NAME = "box_push_%d" %(num_episodes)
    OUTPUT_DIR = os.path.join(get_data_ssd_root(), 'dataset', DATASET_NAME)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    collect_episodes(
        config,
        output_dir=OUTPUT_DIR,
        visualize=False,
        debug=False)

    elapsed = time.time() - start_time
    print("Generating and saving dataset to disk took %d seconds" % (int(elapsed)))


if __name__ == "__main__":
    main()
