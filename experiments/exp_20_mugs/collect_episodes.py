import os
import time
import math
import numpy as np
import random
import copy
from gym.spaces import Box
import transforms3d.derivations.eulerangles

import pydrake

# transforms3d
import transforms3d

import torch

from key_dynam.utils.utils import save_yaml, load_yaml, get_data_root, get_project_root, get_data_ssd_root
from key_dynam.utils.utils import get_project_root, get_current_YYYY_MM_DD_hh_mm_ss_ms, load_yaml, random_sample_in_range
from key_dynam.dataset.episode_dataset import MultiEpisodeDataset
from key_dynam.utils import transform_utils
from key_dynam.experiments.exp_09 import utils as exp_utils
from key_dynam.dynamics.utils import set_seed
from key_dynam.envs import utils as env_utils
from key_dynam.envs.drake_mugs import DrakeMugsEnv, collect_single_episode
from key_dynam.sim_assets.sdf_helper import SDFHelper
from key_dynam.eval.utils import compute_pose_error

from multiprocessing import Process, Queue


"""
Generates a dataset using the DrakePusherSlider environment.
"""


def sample_object_position(T_aug=None,
                           upright=False):

    pos = np.array([0, 0, 0.1])
    quat = None
    if upright:
        quat = np.array([1,0,0,0])
    else:
        quat = transforms3d.euler.euler2quat(np.deg2rad(90), 0, 0)

    T_O_slider = transform_utils.transform_from_pose(pos, quat)

    # apply a random yaw to the object
    yaw_min = 0
    yaw_max = 2*np.pi
    yaw = random_sample_in_range(0, 2*np.pi)
    quat_yaw = transforms3d.euler.euler2quat(0, 0, yaw)
    T_yaw = transform_utils.transform_from_pose([0,0,0], quat_yaw)

    T_O_slider = np.matmul(T_yaw, T_O_slider)

    T_W_slider = None
    if T_aug is not None:
        T_W_slider = T_aug @ T_O_slider
    else:
        T_W_slider = T_O_slider

    pose_dict = transform_utils.matrix_to_dict(T_W_slider)

    # note the quat/pos ordering
    q = np.concatenate((pose_dict['quaternion'], pose_dict['position']))

    return q


def sample_pusher_position_and_velocity(T_aug=None,
                                        N=1000,
                                        n_his=0,
                                        randomize_velocity=None):

    assert randomize_velocity is not None

    if T_aug is None:
        T_aug = np.eye(4)

    low = np.array([-0.1, -0.05])
    high = np.array([-0.09, 0.05])
    q_pusher = np.array(Box(low, high).sample())

    q_pusher_3d_homog = np.array([0, 0, 0, 1.0])
    q_pusher_3d_homog[:2] = q_pusher


    v_pusher_3d = sample_pusher_velocity_3D()

    # 1000 timesteps of pusher velocity
    v_pusher_3d_seq = np.zeros([N, 3])
    for i in range(N):

        v_pusher_3d_tmp = None
        if randomize_velocity:
            v_pusher_3d_tmp = sample_pusher_velocity_3D()
        else:
            v_pusher_3d_tmp = v_pusher_3d

        # always moving to the right
        # v_pusher_3d = magnitude * np.array([1, 0, 0])

        v_pusher_3d_seq[i] = T_aug[:3, :3] @ v_pusher_3d_tmp

    action_seq = v_pusher_3d_seq[:, :2]

    if n_his > 0:
        action_zero_seq = np.zeros([n_his, 2])
        action_seq = np.concatenate((action_zero_seq, action_seq), axis=0)



    v_pusher_3d = T_aug[:3, :3] @ v_pusher_3d
    v_pusher = v_pusher_3d[:2]
    q_pusher_3d_homog = T_aug @ q_pusher_3d_homog
    q_pusher = q_pusher_3d_homog[:2]

    return {'q_pusher': q_pusher,
            'v_pusher': v_pusher,
            'action_sequence': action_seq,
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
                               T_aug_enabled=True,
                               push_length=0.4,
                               N=None,
                               n_his=0,  # whether to add some zero actions at the beginning
                               randomize_velocity=False,
                               randomize_sdf=True,
                               randomize_color=True,
                               upright=False,
                               ):
    T_aug = np.eye(4)
    if T_aug_enabled:
        bound = 0.25
        pos_min = np.array([-bound, -bound, 0])
        pos_max = np.array([bound, bound, 0])
        yaw_min = 0
        yaw_max = 2*np.pi
        T_aug = MultiEpisodeDataset.sample_T_aug(pos_min=pos_min,
                                                 pos_max=pos_max,
                                                 yaw_min=yaw_min,
                                                 yaw_max=yaw_max,)



    q_slider = sample_object_position(T_aug=T_aug, upright=upright)

    vel_avg = 0.2
    dt = config['env']['step_dt']

    if N is None:
        N = math.ceil(push_length/(vel_avg * dt))

    d = sample_pusher_position_and_velocity(T_aug=T_aug,
                                            N=N,
                                            n_his=n_his,
                                            randomize_velocity=randomize_velocity)

    action_sequence = None
    action_sequence = d['action_sequence']

    config_copy = copy.deepcopy(config)
    if randomize_sdf:
        config_copy['env']['model']['sdf'] = sample_random_mug()

    if randomize_color:
        config_copy['env']['model']['color'] = sample_random_color()

    return {'q_slider': q_slider,
            'q_object': q_slider,
            'q_pusher': d['q_pusher'],
            'v_pusher': d['v_pusher'],
            'action_sequence': action_sequence,
            'config': config_copy,
            }

def sample_random_mug():
    sdf_dir = os.path.join(get_data_root(), "stable/sim_assets/anzu_mugs")
    # sdf_file = random.choice(SDFHelper.get_sdf_list(sdf_dir))

    mug_list = load_yaml(os.path.join(get_project_root(), 'experiments/exp_20_mugs/mugs.yaml'))
    sdf_file = random.choice(mug_list['corelle_mug-small'])
    sdf_file = os.path.join(sdf_dir, sdf_file)
    return sdf_file

def sample_random_color():
    return list(np.random.choice(range(256), size=3))


def collect_episodes(config,
                     output_dir=None,
                     visualize=True,
                     debug=False,
                     run_from_thread=False,
                     seed=None):

    # gets a random seed for each thread/process independently
    if seed is None:
        seed = np.random.RandomState().randint(0, 10000)

    set_seed(seed)

    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), 'data')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # save the config
    config_save_file = os.path.join(output_dir, 'config.yaml')
    save_yaml(config, config_save_file)

    # initialize config for DataCollector
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


        n_his = config['train_dynamics']['n_history']
        ic = generate_initial_condition(config=config,
                                        T_aug_enabled=True,
                                        n_his=n_his,
                                        randomize_velocity=True,
                                        randomize_sdf=True,
                                        randomize_color=True,
                                        )

        env = DrakeMugsEnv(ic['config'], visualize=visualize)

        if debug:
            print("initial condition\n", ic)

        # set initial condition on environment
        if visualize:
            print("setting target realtime rate 1.0")
            env.simulator.set_target_realtime_rate(1.0)

        env.reset()
        context = env.get_mutable_context()
        env.set_object_position(context, ic['q_slider'])
        env.set_pusher_position(context, ic['q_pusher'])

        print("ic['action_sequence'].shape", ic['action_sequence'].shape)

        # simulate for 10 seconds to let the mug stabilize
        action_zero = env.get_zero_action()
        env.step(action_zero, dt=10.0)



        episode = collect_single_episode(env,
                                              action_seq=ic['action_sequence'])['episode_container']

        # potentially discard it if the object didn't move during the data collection
        if len(episode._data['trajectory']) < 10:
            print("trajectory was too short, skipping")
            continue

        obs_start = episode._data['trajectory'][0]['observation']
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


        pose_error = compute_pose_error(obs_start, obs_end)

        # if slider didn't move by at least 1 mm then discard this episode
        if (pose_error['position_error'] < 0.01) and (pose_error['angle_error_degrees'] < 10):
            print("discarding episode since slider didn't move sufficiently far")
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


    if not run_from_thread:
        save_yaml(metadata, os.path.join(output_dir, 'metadata.yaml'))

    print("Finished collecting episodes")

    return {'metadata': metadata}



def multiprocess_main(num_episodes=1000, num_threads=4):
    set_seed(500)  # just randomly chosen

    start_time = time.time()
    config = load_yaml(os.path.join(get_project_root(), 'experiments/exp_20_mugs/config.yaml'))

    num_episodes_per_thread = math.ceil(num_episodes / num_threads)
    num_episodes = num_threads * num_episodes_per_thread

    # DATASET_NAME = "mugs_random_colors_%d" % (num_episodes)
    # DATASET_NAME = "single_mug_%d"
    # DATASET_NAME = "correlle_mug-small_single_color_%d" %(num_episodes)
    # DATASET_NAME = "single_corelle_mug_%d" %(num_episodes)
    # DATASET_NAME = "correlle_mug-small_many_colors_%d" %(num_episodes)
    DATASET_NAME = "correlle_mug-small_many_colors_random_%d" % (num_episodes)
    # OUTPUT_DIR = os.path.join(get_data_root(), 'sandbox', DATASET_NAME)
    OUTPUT_DIR = os.path.join(get_data_ssd_root(), 'dataset', DATASET_NAME)
    print("OUTPUT_DIR:", OUTPUT_DIR)


    output_dir = OUTPUT_DIR
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)



    def f(q_tmp):
        config = load_yaml(os.path.join(get_project_root(), 'experiments/exp_20_mugs/config.yaml'))
        config['dataset']['num_episodes'] = num_episodes_per_thread
        out = collect_episodes(
            config,
            output_dir=OUTPUT_DIR,
            visualize=False,
            debug=False,
            run_from_thread=True)

        q_tmp.put(out)

    q = Queue()

    process_list = []
    for i in range(num_threads):
        p = Process(target=f, args=(q,))
        p.start()
        process_list.append(p)

    metadata = {'episodes': {}}
    for p in process_list:
        while p.is_alive():
            p.join(timeout=1)

            # empty out the queue
            while not q.empty():
                out = q.get()
                metadata['episodes'].update(out['metadata']['episodes'])




    # double check
    for p in process_list:
        p.join()

    time.sleep(1.0)
    print("All threads joined")
    elapsed = time.time() - start_time

    # collect the metadata.yaml files

    while not q.empty():
        out = q.get()
        metadata['episodes'].update(out['metadata']['episodes'])

    save_yaml(metadata, os.path.join(OUTPUT_DIR, 'metadata.yaml'))
    print("Generating and saving dataset to disk took %d seconds" % (int(elapsed)))


def main():
    start_time = time.time()
    config = load_yaml(os.path.join(get_project_root(), 'experiments/exp_20_mugs/config.yaml'))


    config['dataset']['num_episodes'] = 10

    set_seed(500)  # just randomly chosen

    DATASET_NAME = "mugs_%d" %(config['dataset']['num_episodes'])
    OUTPUT_DIR = os.path.join(get_data_root(), 'sandbox', DATASET_NAME)
    print("OUTPUT_DIR:", OUTPUT_DIR)

    collect_episodes(
        config,
        output_dir=OUTPUT_DIR,
        visualize=False,
        debug=False)

    elapsed = time.time() - start_time
    print("Generating and saving dataset to disk took %d seconds" % (int(elapsed)))


def print_random_number():
    print(np.random.rand())

def test():

    def f():
        seed = np.random.RandomState().randint(0, 10000)
        # print(np.random.RandomState().randint(0, 10000))
        set_seed(seed)
        print(np.random.rand())

    num_threads = 5

    process_list = []
    for i in range(num_threads):
        p = Process(target=f)
        p.start()
        process_list.append(p)


    for p in process_list:
        p.join()


if __name__ == "__main__":
    # main()
    # multiprocess_main(num_episodes=4, num_threads=2)
    # multiprocess_main(num_episodes=600, num_threads=10)
    # multiprocess_main(num_episodes=1000, num_threads=1)
    test()
