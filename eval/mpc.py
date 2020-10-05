import numpy as np
from gym.spaces import Box
import transforms3d
import os
import copy
import pandas as pd

# torch
import torch

# key_dynam
from key_dynam.utils import vis_utils
from key_dynam.utils.utils import save_yaml

DEBUG = True

from key_dynam.scripts.drake_pusher_slider_episode_collector import sample_pusher_velocity_func
from key_dynam.utils import torch_utils
from key_dynam.utils.utils import get_image_diagonal_from_config
from key_dynam.utils.vis_utils import ImageVisualizer
from key_dynam.planner.planner_factory import planner_from_config
from key_dynam.utils import transform_utils
from key_dynam.envs import utils as env_utils
from key_dynam.eval.utils import compute_pose_error


# pdc
from dense_correspondence_manipulation.utils.visualization import draw_reticles


# def sample_pusher_velocity():
#     raise ValueError("Hasn't been updated for Drake Environment")
#
#     angle_threshold = np.deg2rad(30)
#     angle_space = Box(-angle_threshold, angle_threshold)
#     angle = angle_space.sample()
#
#     magnitude = 100
#     pusher_velocity = magnitude * np.array([np.cos(angle), np.sin(angle)])
#
#     return pusher_velocity


def mpc_episode_keypoint_observation(
        config,  # the global config
        model_dy,  # the dynamics model
        model_vision,  # the vision model
        planner,
        obs_goal,
        observation_function,
        env,
        save_dir=None,
        use_gpu=True,
        wait_for_user_input=False,
        debug_dict=None,  # dict for storing arbitrary data for debugging
        visualize=True,
        verbose=True,
        video=True,
):
    '''
    Run an MPC episode on DrakePusherSlider system using the full visuomotor
    policy
    '''

    # build visualizer
    image_visualizer = None
    if visualize:
        num_images = 3
        image_visualizer = ImageVisualizer(num_images, 1)

    # NOTE: this is specific to dense descriptor model
    camera_name = config['vision_net']['camera_name']
    rgb_img_to_tensor = torch_utils.make_default_image_to_tensor_transform()

    # basic info
    n_his = config['train']['n_history']
    num_steps = config['mpc']['num_timesteps'] + n_his

    # set up episode container
    action_zero = np.zeros(2)

    # container to hold the states that get passed to MPC planner
    states_container = np.zeros((num_steps, config['dataset']['state_dim']))

    # container to hold the states that get passed to the MPC planner
    action_container = np.zeros((num_steps, config['dataset']['action_dim']))

    debug_data = []

    action_lower_lim = np.array(config['mpc']['action_lower_lim'])
    action_upper_lim = np.array(config['mpc']['action_upper_lim'])

    num_keypoints = config['vision_net']['num_ref_descriptors']
    diag = get_image_diagonal_from_config(config)
    reward_scale_factor = (1.0 / num_keypoints) * (100 / diag) ** 2  # average of '(squared) percent pixel error'

    # only weight the keypoint positions, not the pusher position
    eval_indices = list(range(2 * num_keypoints))

    # populate action_container with a guess of some initial actions to take
    # only update the ones after n_his since we will stand still for the first
    # n_his steps anyways
    for i in range(n_his, num_steps):
        action_container[i] = sample_pusher_velocity_func()

    '''
    optional: think about passing the ground truth action_sequence into this mpc
    algorithm for testing purposes
    '''

    '''
    model predictive control

    note: will stand still for n_his timesteps to allow for computing the history
    that is needed to pass into the dynamics model
    '''

    image_filenames = []
    for step in range(num_steps):

        print('\n\n----roll_step %d/%d-----------' % (step, num_steps))

        env_obs = env.get_observation()

        # non-vision observation
        # numpy array with shape [obs_dim]
        obs_tmp = observation_function(env_obs)

        # vision observation
        # need to run vision model forwards
        # make sure to convert rgb to rgb_tensor
        with torch.no_grad():
            # this is wrong, needs to be adjusted
            visual_obs = env_obs['images']

            for camera_name_loc in visual_obs:
                if 'rgb' in visual_obs[camera_name_loc]:
                    rgb = np.copy(visual_obs[camera_name_loc]['rgb'])
                    visual_obs[camera_name_loc]['rgb_tensor'] = rgb_img_to_tensor(rgb)

            vision_out_dict = model_vision.forward_visual_obs(visual_obs)

            # [K*2] tensor
            vision_out = vision_out_dict['dynamics_net_input'].squeeze()

            if verbose:
                print("vision_out.shape", vision_out.shape)

        # [K*2 + obs_dim]
        state_cur = torch.cat((vision_out.type_as(obs_tmp), obs_tmp))

        if verbose:
            print("state_cur.shape", state_cur.shape)

        states_container[step] = state_cur

        # stand still for n_his timesteps so that we can accumulate
        # enough history to input to the model
        if step < n_his:
            action = action_zero
        else:
            # state_input_mpc
            # includes states_container[step] which is current observation
            # of the system
            state_input_mpc = states_container[(step + 1) - n_his:step + 1]

            # [n_his + n_look_ahead - 1, action_dim]
            action_input_mpc = action_container[(step + 1) - n_his:]

            if verbose:
                print("state_input_mpc.shape", state_input_mpc.shape)
                print("action_input_mpc.shape", action_input_mpc.shape)
                print("action_input_mpc\n", action_input_mpc)

            # sequence of actions into the future
            n_look_ahead = min(num_steps - step, config['mpc']['n_look_ahead'])

            traj_opt_out = planner.trajectory_optimization(
                state_input_mpc,
                obs_goal,
                model_dy,
                action_input_mpc,
                n_sample=config['mpc']['n_sample'],
                n_look_ahead=n_look_ahead,
                n_update_iter=config['mpc']['n_update_iter_init'] if step == n_his else config['mpc']['n_update_iter'],
                action_lower_lim=action_lower_lim,
                action_upper_lim=action_upper_lim,
                use_gpu=use_gpu,
                eval_indices=eval_indices,
                reward_scale_factor=reward_scale_factor,
                rollout_best_action_sequence=True, )

            # tensor shape [action_dim]
            action_output_mpc = traj_opt_out['action_sequence']
            action = action_output_mpc[0]

            if verbose:
                # print("action_output_mpc", action_output_mpc)
                print("n_look_ahead", n_look_ahead)
                print("action", action)
                print("obs_pred.shape", traj_opt_out['observation_sequence'].shape)

            # record some data
            debug_data_one_step = {'obs': env_obs,
                                   'state_input_mpc': np.copy(state_input_mpc),
                                   'action_input_mpc': np.copy(action_input_mpc),
                                   'action_output_mpc': np.copy(action_output_mpc),
                                   'reward': traj_opt_out['reward'],
                                   'traj_opt_out': traj_opt_out,
                                   }
            debug_data.append(debug_data_one_step)

        if visualize and step >= n_his:

            print("Visualizing:")
            print("camera name:", camera_name)

            # note it comes out as tensor since it went through dataloader
            rgb_goal = debug_dict['goal_data']['visual_observations'][camera_name]['rgb'].squeeze().numpy()
            # add keypoint annotations to rgb_goal
            num_keypoints = int(len(obs_goal) / 2)
            keypoints_goal = obs_goal.reshape([num_keypoints, 2])

            # num_keypoints = keypoints_goal.shape[0]
            rgb_goal_wk = np.copy(rgb_goal)
            draw_reticles(rgb_goal_wk,
                          keypoints_goal[:, 0],
                          keypoints_goal[:, 1],
                          label_color=[0, 255, 0],  # green
                          )

            rgb_current = env_obs['images'][camera_name]['rgb']  # current rgb image
            keypoints_cur = vision_out_dict['keypoints_out']['best_match_dict']['indices'].squeeze()
            rgb_current_wk = np.copy(rgb_current)
            draw_reticles(rgb_current_wk,
                          keypoints_cur[:, 0],
                          keypoints_cur[:, 1],
                          label_color=[255, 0, 0],  # red
                          )

            # keypoints_pred for last timestep
            obs_seq_best_action = traj_opt_out['observation_sequence']
            keypoints_pred = obs_seq_best_action[-1]  # prediction at final timestep
            keypoints_pred = keypoints_pred[:(num_keypoints * 2)]  # trim off extra observations
            keypoints_pred = keypoints_pred.reshape([num_keypoints, 2])  # [K, 2]

            # debug print statements
            # print("obs_seq_best_action.shape", obs_seq_best_action.shape)
            # print("keypoints_pred.shape", keypoints_pred.shape)
            # print("keypoints_pred.dtype", keypoints_pred.dtype)  # they are float I think

            # cast to int so we can visualize
            keypoints_pred = keypoints_pred.astype(np.int32)

            rgb_goal_w_pred_keypoints = np.copy(rgb_goal_wk)
            draw_reticles(rgb_goal_w_pred_keypoints,
                          keypoints_pred[:, 0],
                          keypoints_pred[:, 1],
                          label_color=[0, 0, 255],  # blue
                          )

            # print("type(rgb_goal)", type(rgb_goal))
            # print("type(rgb_current)", type(rgb_current))

            # blend the two of them
            alpha = 0.5
            rgb_blend = (alpha * rgb_goal + (1 - alpha) * rgb_current).astype(rgb_current.dtype)

            # img_list = [rgb_goal_wk, rgb_current, rgb_blend]
            img_list = [rgb_goal_w_pred_keypoints, rgb_current_wk, rgb_blend]
            for idx_tmp, img_tmp in enumerate(img_list):
                # print("idx_tmp", idx_tmp)
                # print("img_tmp.dtype", img_tmp.dtype)
                image_visualizer.draw_image(idx_tmp, 0, img_tmp)

            image_visualizer.visualize_interactive()

            print("action_output_mpc\n", traj_opt_out['action_sequence'])
            print("action:", action)
            print("reward:", traj_opt_out['reward'])

            if video:
                filename = os.path.join(save_dir, "fig%d.png" % (step))
                image_filenames.append(filename)
                image_visualizer.fig.savefig(filename)

        if wait_for_user_input and step >= n_his:
            input("Press Enter to step simulation...")

        # step the simulation forwards
        obs, reward, done, info = env.step(action)
        action_container[step] = action
        print("-----------------------")

    # make the video
    if video:
        video_filename = os.path.join(save_dir, "video.mp4")
        fps = int(1.0 / config['env']['step_dt'])
        vis_utils.create_video(image_filenames,
                               video_filename,
                               fps=fps)

    if wait_for_user_input:
        input("\n\nEnd of episode, press Enter to continue")

    pandas_data = {'reward': debug_data[-1]['reward']}

    # maybe store it in an EpisodeContainer
    return {'debug_data': debug_data,
            'reward_scale_factor': reward_scale_factor,
            'pandas_data': pandas_data,  # will be added to later
            }


def evaluate_mpc(config,  # the global config
                 dynamics_net,  # the dynamics model
                 vision_net,  # the vision model
                 save_dir,  # str: directory to store results
                 observation_function,
                 env,
                 dataset,  # dataset
                 ):
    # save config
    os.makedirs(save_dir)
    save_yaml(config, os.path.join(save_dir, 'config.yaml'))

    n_history = config['train']['n_history']
    start_idx = n_history - 1 + config['eval']['start_idx']
    end_idx = start_idx + config['eval']['episode_length']

    camera_name = config['vision_net']['camera_name']

    # build the planner
    planner = planner_from_config(config)

    episode_names = dataset.get_episode_names()
    episode_names.sort()
    num_episodes = min(config['eval']['num_episodes'], len(episode_names))

    mpc_idx = 0

    pandas_data_list = []

    for i in range(num_episodes):
        mpc_idx += 1

        episode_name = episode_names[i]
        episode = dataset.episode_dict[episode_name]

        data_goal = dataset._getitem(episode,
                                     end_idx,
                                     rollout_length=0,
                                     n_history=1)

        # goal_keypoints
        visual_observations = data_goal['visual_observations']
        vision_net_out = vision_net.forward_visual_obs(data_goal['visual_observations'])
        goal_keypoints = vision_net_out['dynamics_net_input'].squeeze().cpu().numpy()

        debug_dict = {'goal_data': data_goal,
                      'goal_vision_net_out': vision_net_out}

        # reset the simulator state
        env.reset()
        observation_full = episode.get_observation(start_idx)
        context = env.get_mutable_context()
        env.set_simulator_state_from_observation_dict(context,
                                                      observation_full)

        folder_name = "episode_%d" % mpc_idx
        save_dir_tmp = os.path.join(save_dir, folder_name)
        os.makedirs(save_dir_tmp)

        # run the simulation for this episode
        mpc_out = mpc_episode_keypoint_observation(config=config,
                                                   model_dy=dynamics_net,
                                                   model_vision=vision_net,
                                                   planner=planner,
                                                   obs_goal=goal_keypoints,
                                                   observation_function=observation_function,
                                                   env=env,
                                                   save_dir=save_dir_tmp,
                                                   use_gpu=True,
                                                   wait_for_user_input=False,
                                                   debug_dict=debug_dict,
                                                   visualize=True,
                                                   verbose=False,
                                                   video=True
                                                   )

        # ground truth slider to world
        obs_goal = episode.get_observation(end_idx)
        T_W_S_goal = transform_utils.transform_from_pose_dict(obs_goal['slider']['position'])

        obs_final = mpc_out['debug_data'][-1]['obs']
        # actual T_W_S at end of MPC rollout
        T_W_S = transform_utils.transform_from_pose_dict(obs_final['slider']['position'])

        # error between target and actual
        T_goal_S = np.matmul(np.linalg.inv(T_W_S_goal), T_W_S)

        pos_err = np.linalg.norm(T_goal_S[:3, 3])
        axis, angle = transforms3d.axangles.mat2axangle(T_goal_S[:3, :3])

        print("T_W_S[:3, 3]", T_W_S[:3, 3])
        print("T_W_S_goal[:3, 3]", T_W_S_goal[:3, 3])

        data = {'position_error': pos_err,
                'angle_error': abs(angle),
                'angle_error_degrees': np.rad2deg(abs(angle)),
                }

        print("\ndata\n:", data)

        # parse the pandas data out
        pandas_data = mpc_out['pandas_data']
        pandas_data.update(data)

        # record some additional data
        pandas_data['episode_name'] = episode_name
        pandas_data['mpc_idx'] = mpc_idx
        pandas_data['start_idx'] = start_idx
        pandas_data['end_idx'] = end_idx
        pandas_data['output_dir'] = folder_name
        pandas_data_list.append(pandas_data)

    # create dataframe and save to csv
    df = pd.DataFrame(pandas_data_list)
    df.to_csv(os.path.join(save_dir, "data.csv"))

    # record some simple info in a metadata.yaml
    reward_vec = np.array(df['reward'])

    metadata = dict()
    for key in ['reward', 'position_error', 'angle_error_degrees']:
        vec = df[key]
        metadata[key] = {'mean': float(np.mean(vec)),
                          'median': float(np.median(vec)),
                          'std_dev': float(np.std(vec))}

    save_yaml(metadata, os.path.join(save_dir, 'metadata.yaml'))



def mpc_single_episode(model_dy,  # dynamics model
                       env,  # the environment
                       action_sequence,  # [N, 2], numpy array
                       action_zero,
                       episode,  # OnlineEpisodeReader
                       mpc_input_builder,  # DynamicsModelInputBuilder
                       planner,
                       eval_indices=None,
                       goal_func=None,  # function that gets goal from observation
                       config=None,
                       wait_for_user_input=True,
                       ):

    obs_init = env.get_observation() # initial observation
    N = action_sequence.shape[0]
    episode.clear()

    if wait_for_user_input:
        input("initial state, press Enter to continue")


    # rollout single action sequence using the simulator
    obs_rollout_gt = env_utils.rollout_action_sequence(env, action_sequence)[
        'observations']

    # check if state is valid, i.e. it hasn't gone off end of platform
    if not env.state_is_valid():
        return {'valid': False}

    obs_goal = obs_rollout_gt[-1]

    # check what pose error is
    pose_error = compute_pose_error(obs=obs_init,
                                    obs_goal=obs_goal,
                                    )
    # pose change of object must be above a threshold
    if pose_error['position_error'] < 0.05 and pose_error['angle_error_degrees'] < 15:
        return {'valid': False}


    state_goal = goal_func(obs_goal)

    if wait_for_user_input:
        input("goal state, press Enter to continue")

    n_history = config['train']['n_history']

    # reset environment for mpc
    context = env.get_mutable_context()
    env.set_simulator_state_from_observation_dict(context,
                                                  obs_init)
    env.step(action_zero, dt=2.0) # stand still for 2 seconds

    if wait_for_user_input:
        input("initial state, starting MPC")

    # accumulate the observations
    for i in range(n_history):
        obs = env.get_observation()
        action = action_zero
        episode.add_observation_action(obs, action)

    mpc_observations = []
    mpc_observations.append(obs_init)

    # used in the MPPI and CEM planners
    action_seq_rollout_init = None

    # run MPC
    mpc_out = None
    for i in range(N):

        obs_cur = env.get_observation()
        episode.add_observation_only(obs_cur)
        if (i == 0) or config['eval']['replan']:

            mpc_horizon = None
            if config['mpc']['use_fixed_mpc_horizon']:
                mpc_horizon = config['mpc']['n_look_ahead']
            else:
                mpc_horizon = N - i

            idx = episode.get_latest_idx()
            mpc_input_data = mpc_input_builder.get_dynamics_model_input(idx, n_history=n_history)
            state_cur = mpc_input_data['states']
            action_his = mpc_input_data['actions']

            # compute input to planners (if needed)
            if mpc_out is not None:
                # move it over by 1
                action_seq_rollout_init = mpc_out['action_seq'][1:]

                # add zeros to the end if needed
                n, action_dim = action_seq_rollout_init.shape
                if n < mpc_horizon:
                    action_seq_zeros = torch.zeros([mpc_horizon - n, action_dim])
                    action_seq_rollout_init = torch.cat((action_seq_rollout_init, action_seq_zeros), dim=0)
            else:
                action_seq_rollout_init = None

            z_cur = model_dy.compute_z_state(state_cur)['z']
            mpc_out = planner.trajectory_optimization(state_cur=z_cur,
                                                      action_his=action_his,
                                                      obs_goal=state_goal,
                                                      model_dy=model_dy,
                                                      action_seq_rollout_init=action_seq_rollout_init,
                                                      n_look_ahead=mpc_horizon,
                                                      eval_indices=eval_indices,
                                                      return_debug_info=True,
                                                      verbose=False,
                                                      )

        action_seq_mpc = mpc_out['action_seq'].cpu().detach().numpy()
        action_cur = action_seq_mpc[0]

        obs_cur = env.get_observation()
        episode.replace_observation_action(obs_cur, action_cur)

        env.step(action_cur)
        mpc_observations.append(obs_cur)

        if wait_for_user_input:
            input("press Enter to continue MPC")

    return {'obs_init': obs_init,
            'obs_goal': obs_goal,
            'state_goal': state_goal,
            'mpc_observations': mpc_observations,
            'obs_mpc_final': mpc_observations[-1],
            'valid': True,
            }