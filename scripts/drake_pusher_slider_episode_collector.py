from __future__ import print_function

from gym.spaces import Box
import numpy as np
import os
import transforms3d.derivations.eulerangles
import time


# key_dynam
from key_dynam.envs.drake_pusher_slider import DrakePusherSliderEnv
from key_dynam.dataset.episode_container import EpisodeContainer, MultiEpisodeContainer
from key_dynam.utils.utils import get_current_YYYY_MM_DD_hh_mm_ss_ms, save_yaml


DEBUG = False

def sample_pusher_velocity_func():
    angle_threshold = np.deg2rad(30)
    magnitude = 0.2

    angle_space = Box(-angle_threshold, angle_threshold)
    angle = angle_space.sample()

    pusher_velocity = magnitude * np.array([np.cos(angle), np.sin(angle)])

    return pusher_velocity

class DrakePusherSliderEpisodeCollector(object):

    def __init__(self, config, visualize=False):
        self._env = DrakePusherSliderEnv(config, visualize=visualize)
        self._config = config


    @staticmethod
    def make_default_config():
        config = dict()
        config['num_timesteps'] = 100
        return config

    def get_pusher_slider_initial_positions(self):

        # yaw angle
        high = np.deg2rad(30)
        low = -high
        yaw_angle = Box(low, high).sample()
        quat = np.array(transforms3d.euler.euler2quat(0., 0., yaw_angle))

        # slider_position
        low = np.array([-0.15, -0.05, 0.05])
        high = np.array([-0.2, 0.05, 0.06])
        slider_position = np.array(Box(low, high).sample())


        q_slider = np.concatenate((quat, slider_position))


        low = np.array([-0.07, -0.01])
        high = np.array([-0.09, 0.01])
        delta_pusher = Box(low, high).sample()
        q_pusher = slider_position[0:2] + delta_pusher

        return q_pusher, q_slider



    def collect_single_episode(self,
                               visualize=True,
                               episode_name=None,
                               q_pusher=None,
                               q_slider=None,
                               v_pusher=None,
                               action_sampler=None,  # functional
                               action_seq=None,  # sequence of actions [N, action_dim]
                               ):
        """
        This collects a single episode by performing the following steps

        - sample initial conditions for environment
        - sample action for environment
        - collect the interaction

        :return: EpisodeContainer
        :rtype:
        """

        if episode_name is None:
            episode_name = get_current_YYYY_MM_DD_hh_mm_ss_ms()

        env = self._env
        env.reset()



        if visualize:
            print("setting target realtime rate 1.0")
            env.simulator.set_target_realtime_rate(1.0)
            # it takes a while for this to take effect, so it might initially
            # look like the sim is running really fast, but this is
            # just a cosmetic issue


        # sample if they weren't passed in
        if q_pusher is None or q_slider is None:
            q_pusher, q_slider = self.get_pusher_slider_initial_positions()

        use_constant_action = (v_pusher is None)
        if action_sampler is None:
            action_sampler = self.sample_pusher_velocity


        def get_action(counter):
            if v_pusher is not None:
                return v_pusher
            elif action_seq is not None:
                return action_seq[counter]
            elif action_sampler is not None:
                return action_sampler()
            else:
                return self.sample_pusher_velocity()


        context = env.get_mutable_context()
        env.set_pusher_position(context, q_pusher)
        env.set_slider_position(context, q_slider)
        action_zero = np.zeros(2)

        # if visualize:
        #     env.step(action_zero)
        #     print("sleeping for several seconds to let meshcat visualize")
        #     time.sleep(5.0)

        # log data into EpisodeContainer object
        episode_container = EpisodeContainer()
        episode_container.set_config(env.config)
        episode_container.set_name(episode_name)
        episode_container.set_metadata(env.get_metadata())

        # step for 2 seconds to let object fall
        env.step(action_zero, dt=2.0)

        obs_prev = env.get_observation()

        start_time = time.time()

        num_timesteps = self._config['dataset']['num_timesteps'] + self._config['train']['n_history']
        counter = 0
        for i in range(num_timesteps):

            start_time_1 = time.time()

            # stand still for n_history timesteps
            if i < self._config['train']['n_history']:
                action = action_zero # just sit still

            # this makes it easier to train our dynamics model
            # since we need to get a history of observations
            else:
                action = get_action(counter)
                counter += 1

                # exploration_type = self._config['dataset']['data_generation']['exploration_type']
                # if exploration_type == 'random':
                #     action = self.sample_pusher_velocity()
                # elif exploration_type == 'constant':
                #     action = action_actual


            # single sim time
            obs, reward, done, info = env.step(action)
            episode_container.add_obs_action(obs_prev, action)
            obs_prev = obs

            if visualize:
                # note that sim wil
                elapsed = time.time() - start_time_1
                hz = 1.0/elapsed
                # print("sim rate (hz)", hz)


            # print("pusher velocity", obs['pusher']['velocity'])


            # terminate if outside boundary
            if not env.slider_within_boundary():
                print('slider outside boundary, terminating episode')
                break

            if not env.pusher_within_boundary():
                print('pusher ouside boundary, terminating episode')
                break

        elapsed = time.time() - start_time
        if DEBUG:
            print("elapsed wall clock", elapsed)
            print("elapsed sim time", obs_prev['sim_time'])

        return episode_container

    def sample_pusher_velocity(self):
        """
        Default action sampler
        :return:
        :rtype:
        """
        angle_threshold = np.deg2rad(30)
        magnitude = 0.2

        angle_space = Box(-angle_threshold, angle_threshold)
        angle = angle_space.sample()

        pusher_velocity = magnitude * np.array([np.cos(angle), np.sin(angle)])

        return pusher_velocity



def collect_episodes(config, output_dir=None, visualize=True, use_threads=False):

    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), 'data')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # save the config
    config_save_file = os.path.join(output_dir, 'config.yaml')
    save_yaml(config, config_save_file)

    # initialize config for DataCollector
    dc = DrakePusherSliderEpisodeCollector(config)
    num_episodes = config['dataset']['num_episodes']

    # record some metadata
    metadata = dict()
    metadata['episodes'] = dict()


    for i in range(num_episodes):

        print("\n")
        start_time = time.time()
        print("collecting episode %d of %d" %(i+1, num_episodes))
        name = "%s_idx_%d" %(get_current_YYYY_MM_DD_hh_mm_ss_ms(), i)

        episode = dc.collect_single_episode(visualize, episode_name=name)

        print("saving to disk")
        metadata['episodes'][name] = dict()

        image_data_file = episode.save_images_to_hdf5(output_dir)
        non_image_data_file = episode.save_non_image_data_to_pickle(output_dir)

        print("non_image_data.keys()", episode.non_image_data.keys())

        metadata['episodes'][name]['non_image_data_file'] = non_image_data_file
        metadata['episodes'][name]['image_data_file'] = image_data_file

        print("done saving to disk")
        elapsed = time.time() - start_time
        print("single episode took: %.2f seconds" %(elapsed))


    save_yaml(metadata, os.path.join(output_dir, 'metadata.yaml'))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default=None, required=False, type=str)
    parser.add_argument('--no_vis', action='store_false', dest='visualize', default=True)
    args = parser.parse_args()

    collect_episodes(output_dir=args.output_dir,
                     visualize=args.visualize)

