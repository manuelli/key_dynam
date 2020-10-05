from __future__ import print_function

import os
import time
import numpy as np
import pygame


from key_dynam.envs.pusher_slider import PusherSlider
from key_dynam.utils.utils import numpy_to_PIL, get_project_root, load_yaml
from key_dynam.dataset.episode_container import EpisodeContainer, MultiEpisodeContainer

DEBUG_PRINTS = True

def test_gym_API():
    """
    Collects two episodes and saves them to disk
    Currently no images are being saved
    :return:
    :rtype:
    """

    config_file = os.path.join(get_project_root(), 'experiments/01/config.yaml')
    config = load_yaml(config_file)

    env = PusherSlider(config=config)
    env.setup_environment()

    episode_container = None # for namespacing purposes
    multi_episode_container = MultiEpisodeContainer()

    print("fps", env._fps)
    num_episodes = 1
    # num_timesteps_per_episode = 40
    for episode_idx in range(num_episodes):
        env.reset()
        obs_prev = env.get_observation()

        episode_container = EpisodeContainer()
        episode_container.set_config(env.config)
        for i in range(env.config['dataset']['num_timesteps']):
            env.render(mode='human')
            if i == 50:

                # save this image out in local directory
                img = env.render(mode='rgb_array')

                if DEBUG_PRINTS:
                    print("saving figure to file")
                    print("type(img)", type(img))
                    print(img.shape)
                    print(img.dtype)

                pil_img = numpy_to_PIL(img)
                pil_img.save('test_PIL.png')

            action = np.array([100, 0])
            obs, reward, done, info = env.step(action)

            episode_container.add_obs_action(obs_prev, action)
            obs_prev = obs

            if DEBUG_PRINTS:
                print("\n\nsim_time = ", env._sim_time)
                print("obs sim_time = ", obs['sim_time'])




        multi_episode_container.add_episode(episode_container)


    if True:
        save_file = os.path.join(os.getcwd(), "single_episode_log.p")
        episode_container.save_to_file(save_file)

        multi_episode_save_file = os.path.join(os.getcwd(), "multi_episode_log.p")
        multi_episode_container.save_to_file(multi_episode_save_file)


def run_interactive():
    """
    Launch interactive environment where you can move the pusher around with
    the arrow keys
    :return:
    :rtype:
    """
    config_file = os.path.join(get_project_root(), 'experiments/01/config.yaml')
    config = load_yaml(config_file)
    env = PusherSlider(config=config)
    env.reset()


    while env._running:
        action = env.process_events()
        env.step(action)
        obs, reward, done, info = env.step(action)
        env.render(mode='human')

        if True:
            print("\n\n\n")
            print("pygame.key.get_focused():", pygame.key.get_focused())
            print("slider position", obs['slider']['position'])
            print("pusher position", obs['pusher']['position'])

def run_interactive_circle_slider():
    """
    Launch interactive environment where you can move the pusher around with
    the arrow keys
    :return:
    :rtype:
    """
    config_file = os.path.join(get_project_root(), 'experiments/03/config.yaml')
    config = load_yaml(config_file)
    env = PusherSlider(config=config)
    env.reset()

    while env._running:
        action = env.process_events()
        env.step(action)
        obs, reward, done, info = env.step(action)
        env.render(mode='human')


        if True:
            print("\n\n\n")
            print("slider position", obs['slider']['position'])
            print("pusher position", obs['pusher']['position'])

def test_render():
    """
    Launch interactive environment where you can move the pusher around with
    the arrow keys
    :return:
    :rtype:
    """
    env = PusherSlider()
    env.config = PusherSlider.make_default_config()
    env.reset()

    action = np.zeros(2)
    env.step(action) # this is necessary, doesn't work otherwise . . .
    env.render(mode='human')
    print(env._slider_body.position)
    time.sleep(3.0)

    print("moving the block")
    env._slider_body.position = [400, 300]
    env._space.step(1e-4)
    # env.step(action)
    env.render(mode='human')
    time.sleep(3.0)


if __name__ == "__main__":
    # test_gym_API()
    run_interactive()
    # run_interactive_circle_slider()
    # test_render()