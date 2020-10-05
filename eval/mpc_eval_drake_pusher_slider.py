import os
import numpy as np
import transforms3d
import torch
import pandas as pd
import random
import functools
import time


# torch
import torch
from torch.utils.tensorboard import SummaryWriter

from key_dynam.envs.drake_pusher_slider import DrakePusherSliderEnv
from key_dynam.dataset.episode_dataset import MultiEpisodeDataset
from key_dynam.utils import transform_utils, torch_utils
from key_dynam.utils.utils import random_sample_in_range, save_yaml
from key_dynam.dynamics.utils import set_seed
from key_dynam.eval.mpc import mpc_single_episode
from key_dynam.eval.utils import compute_pose_error



def set_random_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def evaluate_mpc(model_dy,  # dynamics model
                 env,  # the environment
                 episode,  # OnlineEpisodeReader
                 mpc_input_builder,  # DynamicsModelInputBuilder
                 planner,  # RandomShooting planner
                 eval_indices=None,
                 goal_func=None,  # function that gets goal from observation
                 config=None,
                 wait_for_user_input=False,
                 save_dir=None,
                 model_name="",
                 experiment_name="",
                 generate_initial_condition_func=None,
                 # (optional) function to generate initial condition, takes episode length N as parameter
                 ):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # must specify initial condition distribution
    assert generate_initial_condition_func is not None

    save_yaml(config, os.path.join(save_dir, 'config.yaml'))
    writer = SummaryWriter(log_dir=save_dir)

    pandas_data_list = []
    for episode_length in config['eval']['episode_length']:
        counter = 0
        seed = 0
        while counter < config['eval']['num_episodes']:

            start_time = time.time()
            seed += 1
            set_seed(seed)  # make it repeatable
            # initial_cond = generate_initial_condition(config, N=episode_length)
            initial_cond = generate_initial_condition_func(N=episode_length)

            env.set_initial_condition_from_dict(initial_cond)

            action_sequence_np = torch_utils.cast_to_numpy(initial_cond['action_sequence'])
            episode_data = mpc_single_episode(model_dy=model_dy,
                                              env=env,
                                              action_sequence=action_sequence_np,
                                              action_zero=np.zeros(2),
                                              episode=episode,
                                              mpc_input_builder=mpc_input_builder,
                                              planner=planner,
                                              eval_indices=eval_indices,
                                              goal_func=goal_func,
                                              config=config,
                                              wait_for_user_input=wait_for_user_input,
                                              )

            # continue if invalid
            if not episode_data['valid']:
                print("invalid episode, skipping")
                continue

            pose_error = compute_pose_error(obs=episode_data['obs_mpc_final'],
                                            obs_goal=episode_data['obs_goal'],
                                            )


            object_delta = compute_pose_error(obs=episode_data['obs_init'],
                                              obs_goal=episode_data['obs_goal'])

            print("object_delta\n", object_delta)

            if wait_for_user_input:
                print("pose_error\n", pose_error)

            pandas_data = {'episode_length': episode_length,
                           'seed': counter,
                           'model_name': model_name,
                           'experiment_name': experiment_name,
                           'object_pos_delta': object_delta['position_error'],
                           'object_angle_delta': object_delta['angle_error'],
                           'object_angle_delta_degrees': object_delta['angle_error_degrees'],
                           }


            pandas_data.update(pose_error)
            pandas_data_list.append(pandas_data)

            # log to tensorboard
            for key, val in pose_error.items():
                plot_name = "%s/episode_len_%d" % (key, episode_length)
                writer.add_scalar(plot_name, val, counter)

            writer.flush()

            print("episode [%d/%d], episode_length %d, duration %.2f" % (
            counter, config['eval']['num_episodes'], episode_length, time.time() - start_time))
            counter += 1

        df_tmp = pd.DataFrame(pandas_data_list)
        keys = ["angle_error_degrees", "position_error"]
        for key in keys:
            for i in range(10):
                mean = df_tmp[key][df_tmp.episode_length == episode_length].mean()
                median = df_tmp[key][df_tmp.episode_length == episode_length].median()

                plot_name_mean = "mean/%s/episode_len_%d" % (key, episode_length)
                writer.add_scalar(plot_name_mean, mean, i)

                plot_name_median = "median/%s/episode_len_%d" % (key, episode_length)
                writer.add_scalar(plot_name_median, median, i)

    # save some data
    df = pd.DataFrame(pandas_data_list)
    df.to_csv(os.path.join(save_dir, "data.csv"))
