import os
import random
import itertools
import cv2
import numpy as np
from six.moves import cPickle as pickle
from progressbar import ProgressBar
import time
from PIL import Image
from gym.spaces import Box

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle, Circle

import torch
import torch.nn as nn


from key_dynam.dynamics.models_dy import DynaNetMLP
from key_dynam.dynamics.utils import count_trainable_parameters, Tee, AverageMeter, to_np, to_var, norm, set_seed
from key_dynam.utils.utils import get_project_root, load_pickle
from key_dynam.dataset.function_factory import ObservationFunctionFactory, ActionFunctionFactory, pusher_slider_pose_from_tensor, pusher_pose_slider_keypoints_from_tensor
from key_dynam.dataset.episode_reader import PyMunkEpisodeReader
from key_dynam.dataset.episode_dataset import MultiEpisodeDataset
from key_dynam.envs.pusher_slider import PusherSlider
from key_dynam.utils.utils import numpy_to_PIL
from key_dynam.dataset.vis_utils import pusher_slider_keypoints_image

from key_dynam.planner.planner_factory import planner_from_config


def sample_pusher_velocity():
    angle_threshold = np.deg2rad(30)
    angle_space = Box(-angle_threshold, angle_threshold)
    angle = angle_space.sample()

    magnitude = 100
    pusher_velocity = magnitude * np.array([np.cos(angle), np.sin(angle)])

    return pusher_velocity


def mpc_episode_keypoint_observation(
    config,     # the global config
    mpc_idx,    # int: index of this trial for logging purposes
    model_dy,   # the dynamics model
    mpc_dir,    # str: directory to store results
    planner,
    obs_goal,
    action_function,
    observation_function,
    video=True,
    image=True,
    use_gpu=True):

    '''
    setup
    '''

    # optionally add noise to observation as specified in config
    add_noise_to_observation = False
    if "add_noise_to_observation" in config["mpc"]:
        # raise ValueError("testing")
        add_noise_to_observation = config["mpc"]["add_noise_to_observation"]

    # basic info
    n_his = config['train']['n_history']
    num_steps = config['mpc']['num_timesteps'] + n_his

    # create PusherSlider env
    env = PusherSlider(config)
    env.setup_environment()
    env.reset()
    env.step_space_to_initialize_render()


    # set up episode container
    action_zero = np.zeros(2)

    # states used to roll out the MPC model
    states_roll_mpc = np.zeros((num_steps, config['dataset']['state_dim']))

    # gt states that actually occurred
    states_roll_gt = np.zeros(states_roll_mpc.shape)
    actions_roll = np.zeros((num_steps, config['dataset']['action_dim']))

    gt_obs_list = []
    img_gt_PIL_list = []


    actions_roll = np.zeros((num_steps, config['dataset']['action_dim']))

    for i in range(n_his, num_steps):
        actions_roll[i] = sample_pusher_velocity()

    action_lower_lim = np.array(config['mpc']['action_lower_lim'])
    action_upper_lim = np.array(config['mpc']['action_upper_lim'])


    '''
    model predictive control
    '''
    for step in range(num_steps):

        print('roll_step %d/%d' % (step, num_steps))

        states_roll_mpc[step] = observation_function(env.get_observation(), data_augmentation=add_noise_to_observation)
        img_gt_PIL_list.append(numpy_to_PIL(env.render(mode='rgb_array')))


        # stand still for n_his timesteps
        if step < n_his:
            action = action_zero
        else:
            actions_roll[step:] = planner.trajectory_optimization(
                states_roll_mpc[step - n_his + 1:step + 1],

                obs_goal,
                model_dy,
                actions_roll[step - n_his + 1:],
                n_sample=config['mpc']['n_sample'],
                n_look_ahead=min(num_steps-step, config['mpc']['n_look_ahead']),
                n_update_iter=config['mpc']['n_update_iter_init'] if step == n_his else config['mpc']['n_update_iter'],
                action_lower_lim=action_lower_lim,
                action_upper_lim=action_upper_lim,
                use_gpu=use_gpu)

            action = actions_roll[step]

        obs, reward, done, info = env.step(action)
        gt_obs_list.append(obs)



    '''
    render the prediction
    '''

    # reset PusherSlider env
    env.reset()

    # setup rendering data

    width = config['env']['display_size'][0]
    height = config['env']['display_size'][1]
    n_split = 3
    split = 4

    if image:
        eval_img_dir = os.path.join(mpc_dir, '%d_img' % mpc_idx)
        os.system('mkdir -p ' + eval_img_dir)
        print('Save images to %s' % eval_img_dir)

    if video:
        eval_vid_path = os.path.join(mpc_dir, '%d_vid.avi' % mpc_idx)
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        print('Save video as %s' % eval_vid_path)
        fps = int(env.fps)
        print("fps:", fps)
        out = cv2.VideoWriter(eval_vid_path, fourcc, fps, (
            width * n_split + split * (n_split - 1), height))

    # gt represents goal
    for i, start_idx in enumerate(range(num_steps)):

        # get ground truth positions at goal
        goal_gt_state_dict = pusher_pose_slider_keypoints_from_tensor(obs_goal)

        # now render them
        img_goal_gt_PIL = pusher_slider_keypoints_image(
            config,
            goal_gt_state_dict['pusher']['position'],
            goal_gt_state_dict['keypoint_positions'])

        img_goal_gt = np.array(img_goal_gt_PIL)

        #
        img_gt_PIL = img_gt_PIL_list[i]

        # # actual GT positions that happened
        state_roll = states_roll_mpc[i, :]
        state_roll_dict = pusher_pose_slider_keypoints_from_tensor(state_roll)
        img_gt_PIL = pusher_slider_keypoints_image(
            config,
            state_roll_dict['pusher']['position'],
            state_roll_dict['keypoint_positions'],
            img=img_gt_PIL)

        img_gt = np.array(img_gt_PIL)

        # blend the images
        img_merge_PIL = Image.blend(img_goal_gt_PIL, img_gt_PIL, 0.5)

        # get numpy image
        img_merge = np.array(img_merge_PIL)

        img_output = np.zeros((
            img_goal_gt.shape[0],
            img_goal_gt.shape[1] * n_split + split * (n_split - 1), 3)).astype(np.uint8)

        img_output[:, :img_goal_gt.shape[1]] = img_goal_gt # goal keypoints image
        img_output[:, img_goal_gt.shape[1] + split: img_goal_gt.shape[1] * 2 + split] = img_gt
        img_output[:, (img_goal_gt.shape[1] + split) * 2:] = img_merge


        if image:
            # convert to PIL then save
            img_output_PIL = numpy_to_PIL(img_output)
            img_output_PIL.save(os.path.join(eval_img_dir, 'fig_%d.png' % i))

        if video:
            # convert from RGB to BGR since video writer wants BGR
            img_output_bgr = cv2.cvtColor(img_output, cv2.COLOR_RGB2BGR)
            out.write(img_output_bgr)

    if video:
        out.release()


def mpc_w_learned_dynamics(config, train_dir, mpc_dir, state_dict_path=None,
                           keypoint_observation=False):

    # set random seed for reproduction
    set_seed(config['train']['random_seed'])

    tee = Tee(os.path.join(mpc_dir, 'mpc.log'), 'w')

    print(config)

    use_gpu = torch.cuda.is_available()


    '''
    model
    '''
    if config['dynamics']['model_type'] == 'mlp':
        model_dy = DynaNetMLP(config)
    else:
        raise AssertionError("Unknown model type %s" % config['dynamics']['model_type'])

    # print model #params
    print("model #params: %d" % count_trainable_parameters(model_dy))

    if state_dict_path is None:
        if config['mpc']['mpc_dy_epoch'] == -1:
            state_dict_path = os.path.join(train_dir, 'net_best_dy.pth')
        else:
            state_dict_path = os.path.join(
                train_dir, 'net_dy_epoch_%d_iter_%d.pth' % \
                (config['mpc']['mpc_dy_epoch'], config['mpc']['mpc_dy_iter']))

        print("Loading saved ckp from %s" % state_dict_path)

    model_dy.load_state_dict(torch.load(state_dict_path))
    model_dy.eval()

    if use_gpu:
        model_dy.cuda()

    criterionMSE = nn.MSELoss()

    # generate action/observation functions
    action_function = ActionFunctionFactory.function_from_config(config)
    observation_function = ObservationFunctionFactory.function_from_config(config)

    # planner
    planner = planner_from_config(config)


    '''
    env
    '''
    # set up goal
    obs_goals = np.array([
        [262.9843, 267.3102, 318.9369, 351.1229, 360.2048, 323.5128, 305.6385, 240.4460, 515.4230, 347.8708],
        [381.8694, 273.6327, 299.6685, 331.0925, 328.7724, 372.0096, 411.0972, 314.7053, 517.7299, 268.4953],
        [284.8728, 275.7985, 374.0677, 320.4990, 395.4019, 275.4633, 306.2896, 231.4310, 507.0849, 312.4057],
        [313.1638, 271.4258, 405.0255, 312.2325, 424.7874, 266.3525, 333.6973, 225.7708, 510.1232, 305.3802],
        [308.6859, 270.9629, 394.2789, 323.2781, 419.7905, 280.1602, 333.8901, 228.1624, 519.1964, 321.5318],
        [386.8067, 284.8947, 294.2467, 323.2223, 313.3221, 368.9970, 405.9415, 330.9298, 495.9970, 268.9920],
        [432.0219, 299.6021, 340.8581, 339.4676, 360.2354, 384.5515, 451.4394, 345.2190, 514.6357, 291.2043],
        [351.3389, 264.5325, 267.5279, 318.2321, 293.7460, 360.0423, 378.4428, 306.9586, 516.4390, 259.7810],
        [521.1902, 254.0693, 492.7884, 349.7861, 539.6320, 364.5190, 569.2258, 268.8824, 506.9431, 286.9752],
        [264.8554, 275.9547, 338.1317, 345.3435, 372.7012, 308.4648, 299.3454, 239.9245, 506.2117, 373.8413]
    ])

    for mpc_idx in range(config['mpc']['num_episodes']):
        if keypoint_observation:
            mpc_episode_keypoint_observation(
                config, mpc_idx, model_dy, mpc_dir, planner, obs_goals[mpc_idx],
                action_function, observation_function, use_gpu=use_gpu)
        else:
            # not supported for now
            raise AssertionError("currently only support keypoint observation")


