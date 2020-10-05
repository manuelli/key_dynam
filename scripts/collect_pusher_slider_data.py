from __future__ import print_function

from gym.spaces import Box
import numpy as np
import os

# key_dynam
from key_dynam.envs.pusher_slider import PusherSlider
from key_dynam.dataset.episode_container import EpisodeContainer, MultiEpisodeContainer
from key_dynam.utils.utils import get_current_YYYY_MM_DD_hh_mm_ss_ms, save_yaml


DEBUG = False

class PusherSliderDataCollector(object):

    def __init__(self, config):
        self._env = PusherSlider(config)
        self._config = config

    @staticmethod
    def make_default_config():
        config = dict()
        config['num_timesteps'] = 100
        return config

    def collect_single_episode(self, visualize=True, episode_name=None):
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

        # set pusher position
        # this is in Body frame
        pusher_position_B = self.pusher_initial_position_space_from_config(env.config).sample()

        # this is in world frame
        pusher_position_W = env.slider_body.local_to_world(pusher_position_B)

        # set pusher position
        env.pusher_body.position = pusher_position_W

        # step so we can get an observation that has correct image
        env.step([0,0])

        # get action
        # don't worry about converting it to world frame for now
        pusher_velocity_B = self.sample_pusher_velocity()
        action_actual = pusher_velocity_B
        action_zero = np.zeros(2)

        episode_container = EpisodeContainer()
        episode_container.set_config(env.config)
        episode_container.set_name(episode_name)
        obs_prev = env.get_observation()

        num_timesteps = self._config['dataset']['num_timesteps'] + self._config['train']['n_history']
        for i in xrange(num_timesteps):

            if visualize:
                env.render(mode='human')

            # stand still for n_history timesteps
            if i < self._config['train']['n_history']:
                action = action_zero # just sit still
            # this makes it easier to train our dynamics model
            # since we need to get a history of observations
            else:
                exploration_type = self._config['dataset']['data_generation']['exploration_type']
                if exploration_type == 'random':
                    action = self.sample_pusher_velocity()
                elif exploration_type == 'constant':
                    action = action_actual

            obs, reward, done, info = env.step(action)
            episode_container.add_obs_action(obs_prev, action)
            obs_prev = obs

            if DEBUG:
                print("\n\nsim_time = ", env._sim_time)
                print("obs sim_time = ", obs['sim_time'])

        return episode_container

    def pusher_initial_position_space_from_config(self, config): # return: spaces.Box2D
        """
        This should be relative to the box pose
        :param config:
        :type config:
        :return:
        :rtype:
        """
        size = np.array(config["env"]["slider"]["size"])


        low_x = -size[0]/2.0 - 30
        high_x = -size[0]/2.0 - 60

        low_y = -size[1]/2.0
        high_y = size[1]/2.0
        low = np.array([low_x, low_y])
        high = np.array([high_x, high_y])

        pusher_position_space = Box(low, high)
        return pusher_position_space

    def sample_pusher_velocity(self):
        angle_threshold = np.deg2rad(30)
        angle_space = Box(-angle_threshold, angle_threshold)
        angle = angle_space.sample()

        magnitude = 100
        pusher_velocity = magnitude * np.array([np.cos(angle), np.sin(angle)])

        return pusher_velocity


def collect_episodes(config, output_dir=None, visualize=True):

    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), 'data')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # save the config
    config_save_file = os.path.join(output_dir, 'config.yaml')
    save_yaml(config, config_save_file)


    save_file = os.path.join(output_dir, "%s.p" % (get_current_YYYY_MM_DD_hh_mm_ss_ms()))

    # initialize config for DataCollector
    dc = PusherSliderDataCollector(config)
    num_episodes = config['dataset']['num_episodes']
    multi_episode_container = MultiEpisodeContainer()
    for i in range(num_episodes):
        print("collecting episode %d of %d" %(i+1, num_episodes))
        name = "%s_idx_%d" %(get_current_YYYY_MM_DD_hh_mm_ss_ms(), i)

        episode = dc.collect_single_episode(visualize, episode_name=name)
        multi_episode_container.add_episode(episode)

    print("saving data to %s" %save_file)
    multi_episode_container.save_to_file(save_file)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default=None, required=False, type=str)
    parser.add_argument('--no_vis', action='store_false', dest='visualize', default=True)
    args = parser.parse_args()

    collect_episodes(output_dir=args.output_dir,
                     visualize=args.visualize)

