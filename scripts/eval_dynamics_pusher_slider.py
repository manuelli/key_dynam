from __future__ import print_function

import os
import random
import itertools
import cv2
import numpy as np
from six.moves import cPickle as pickle
from progressbar import ProgressBar
import time
from PIL import Image


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
from key_dynam.dataset.utils import load_episodes_from_config
from key_dynam.dataset.vis_utils import pusher_slider_keypoints_image



def render_img(state, action, width, height, config):
    # temporary solution
    # probably easiest to just use the same pygame and pymunk setup that was
    # used to generate the images in the first place

    raise ValueError("method deprecated")

    assert config['env']['type'] == 'PusherSlider'

    size = config['env']['slider']['size']
    fig, ax = plt.subplots(1)

    plt.xlim(0, 600)
    plt.ylim(0, 600)

    # draw pusher
    pusher = Circle((action[0], action[1]), radius=5)
    pc = PatchCollection([pusher], facecolor=['tomato'], edgecolor='k', linewidth=0.3)
    ax.add_collection(pc)

    # draw slider
    t = mpl.transforms.Affine2D().rotate_deg_around(
        state[0], state[1], np.rad2deg(state[2]))
    slider = Rectangle((state[0] - size[0] / 2., state[1] - size[1] / 2.), size[0], size[1], transform=t)
    pc = PatchCollection([slider], facecolor=['royalblue'], edgecolor='k', linewidth=0.3)
    ax.add_collection(pc)

    ax.set_aspect('equal')
    plt.axis('off')
    plt.tight_layout()

    fig.canvas.draw()
    frame = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    plt.close()

    return frame[140:-140, 170:-170]

def eval_episode(config, # the global config
                 dataset, # MultiEpisodeDataset
                 episode_name, # str,
                 roll_idx, # int: index of this rollout for logging purposes
                 model_dy, # dynamics model
                 eval_dir, # str: directory to store results
                 video=True, # save a video of rollout
                 image=True, # save images of rollout
                 use_gpu=True,# bool: whether or not to use the gpu
                 start_idx=None, # int: starting index for rollout
                 n_prediction=5, # prediction horizon,
                 render_human=False, # whether to render in human visible mode
                 ):

    episode = dataset._episodes[episode_name]
    n_his = config["train"]["n_history"]
    n_roll = config["train"]["n_rollout"]

    if start_idx is None:
        low, high = dataset.get_valid_idx_range_for_episode(episode)
        start_idx = int((low + high) / 2.0)

    data = dataset.__getitem__(0, episode_name=episode_name, episode_idx=start_idx, rollout_length=0)

    # need to do the unsqueeze to make it look like they
    # are in batch form for doing forward pass through the
    # network
    states_unnorm = data['observations'].unsqueeze(0)
    actions_unnorm = data['actions'].unsqueeze(0)

    if use_gpu:
        device = 'cuda'
        states_unnorm = states_unnorm.cuda()
        actions_unnorm = actions_unnorm.cuda()
    else:
        device='cpu'


    # normalize states for the entire rollout
    states = model_dy.state_normalizer.normalize(states_unnorm)

    # state_cur: B x n_his x state_dim
    state_cur = states[:, :n_his]

    # states.shape[2] should be state_dim
    states_predicted = torch.zeros((n_prediction, states.shape[2]), device=device)
    states_predicted_idx = [None]*n_prediction

    print("states_predicted.shape", states_predicted.shape)

    # this code is exactly copied from the training setup
    # ropagate dynamics forward through time using the learned model
    with torch.no_grad():
        for j in range(n_prediction):

            # action_cur: B x n_his x action_dim
            # get actions from the dataloader
            action_cur = None
            data = dataset.__getitem__(0, episode_name=episode_name, episode_idx=(start_idx+j), rollout_length=0)
            actions_unnorm = data['actions'].unsqueeze(0).cuda()
            actions = model_dy.action_normalizer.normalize(actions_unnorm)
            action_cur = actions[:, :n_his]


            # state_pred: B x state_dim
            # state_cur: B x n_his x state_dim
            state_pred = model_dy(state_cur, action_cur)
            states_predicted[j] = state_pred.squeeze()
            states_predicted_idx[j] = start_idx + j + 1

            # update state_cur
            state_cur = torch.cat([state_cur[:, 1:], state_pred.unsqueeze(1)], 1)


    states_predicted_unnorm = model_dy.state_normalizer.denormalize(states_predicted)

    # create PusherSlider env
    env = PusherSlider(config)
    env.setup_environment()
    env.reset()

    # setup rendering data

    '''
    render the prediction
    '''
    width = config['env']['display_size'][0]
    height = config['env']['display_size'][1]
    n_split = 3
    split = 4

    if image:
        eval_img_dir = os.path.join(eval_dir, '%d_img' % roll_idx)
        os.system('mkdir -p ' + eval_img_dir)
        print('Save images to %s' % eval_img_dir)

    if video:
        eval_vid_path = os.path.join(eval_dir, '%d_vid.avi' % roll_idx)
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        print('Save video as %s' % eval_vid_path)
        fps = int(env.fps)
        print("fps:", fps)
        out = cv2.VideoWriter(eval_vid_path, fourcc, fps, (
            width * n_split + split * (n_split - 1), height))

    # can easily read the ground truth states directly from the episode . . .
    for i, start_idx in enumerate(states_predicted_idx):
        # get predicted positions
        pred_state_np = states_predicted_unnorm[i, :].cpu().numpy()
        predicted_state_dict = pusher_slider_pose_from_tensor(pred_state_np)

        # get ground truth positions
        gt_state_dict = episode.get_observation(start_idx)

        # now render them
        env.set_object_positions_for_rendering(gt_state_dict)
        img_gt = env.render(mode='rgb_array')
        img_gt_PIL = numpy_to_PIL(img_gt)
        if render_human:
            print("rendering human")
            env.render(mode='human')


        # predicted image
        env.set_object_positions_for_rendering(predicted_state_dict)
        img_pred = env.render(mode='rgb_array')
        img_pred_PIL = numpy_to_PIL(img_pred)

        if render_human:
            env.render(mode='human')

        # blend the images
        img_merge_PIL = Image.blend(img_gt_PIL, img_pred_PIL, 0.5)

        # # show the predictions
        # if SHOW_BLENDED_IMAGE:
        #     print("showing PIL image")
        #     img_merge_PIL.show()
        #     time.sleep(1.0)

        # get numpy image
        img_merge = np.array(img_merge_PIL)

        img_output = np.zeros((
            img_gt.shape[0],
            img_gt.shape[1] * n_split + split * (n_split - 1), 3)).astype(np.uint8)

        img_output[:, :img_gt.shape[1]] = img_gt
        img_output[:, img_gt.shape[1] + split: img_gt.shape[1] * 2 + split] = img_pred
        img_output[:, (img_gt.shape[1] + split) * 2:] = img_merge

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


def eval_episode_keypoint_observations(config, # the global config
                 dataset, # MultiEpisodeDataset
                 episode_name, # str,
                 roll_idx, # int: index of this rollout for logging purposes
                 model_dy, # dynamics model
                 eval_dir, # str: directory to store results
                 video=True, # save a video of rollout
                 image=True, # save images of rollout
                 use_gpu=True,# bool: whether or not to use the gpu
                 start_idx=None, # int: starting index for rollout
                 n_prediction=5, # prediction horizon,
                 render_human=False, # whether to render in human visible mode
                 ):
    """
    Similar to 'eval_episode' but specialized to work with
    keypoint obersvations. Predicton step is exactly the same, but rendering
    step is a bit different
    """

    episode = dataset._episodes[episode_name]
    n_his = config["train"]["n_history"]
    n_roll = config["train"]["n_rollout"]

    if start_idx is None:
        low, high = dataset.get_valid_idx_range_for_episode(episode)
        start_idx = int((low + high) / 2.0)

    if "add_noise_to_observation" in config["eval"]:
        # raise ValueError("testing")
        dataset.data_augmentation_enabled = config["eval"]["add_noise_to_observation"]

    data = dataset.__getitem__(0, episode_name=episode_name, episode_idx=start_idx, rollout_length=0)



    # need to do the unsqueeze to make it look like they
    # are in batch form for doing forward pass through the
    # network
    states_unnorm = data['observations'].unsqueeze(0)
    actions_unnorm = data['actions'].unsqueeze(0)

    if use_gpu:
        device = 'cuda'
        states_unnorm = states_unnorm.cuda()
        actions_unnorm = actions_unnorm.cuda()
    else:
        device='cpu'


    # normalize states for the entire rollout
    states = model_dy.state_normalizer.normalize(states_unnorm)

    # state_cur: B x n_his x state_dim
    state_cur = states[:, :n_his]

    # states.shape[2] should be state_dim
    states_predicted = torch.zeros((n_prediction, states.shape[2]), device=device)
    states_predicted_idx = [None]*n_prediction

    print("states_predicted.shape", states_predicted.shape)

    # this code is exactly copied from the training setup
    # propagate dynamics forward through time using the learned model
    with torch.no_grad():
        for j in range(n_prediction):

            # action_cur: B x n_his x action_dim
            # get actions from the dataloader
            action_cur = None
            data = dataset.__getitem__(0, episode_name=episode_name, episode_idx=(start_idx+j), rollout_length=0)
            actions_unnorm = data['actions'].unsqueeze(0).cuda()
            actions = model_dy.action_normalizer.normalize(actions_unnorm)
            action_cur = actions[:, :n_his]

            # state_pred: B x state_dim
            # state_cur: B x n_his x state_dim
            state_pred = model_dy(state_cur, action_cur)
            states_predicted[j] = state_pred.squeeze()
            states_predicted_idx[j] = start_idx + j + 1

            # update state_cur
            state_cur = torch.cat([state_cur[:, 1:], state_pred.unsqueeze(1)], 1)


    states_predicted_unnorm = model_dy.state_normalizer.denormalize(states_predicted)

    print(states_predicted_unnorm)

    # reset dataset to not have data augmentation enabled
    dataset.data_augmentation_enabled = False

    # create PusherSlider env
    env = PusherSlider(config)
    env.setup_environment()
    env.reset()

    # setup rendering data

    '''
    render the prediction
    '''
    width = config['env']['display_size'][0]
    height = config['env']['display_size'][1]
    n_split = 3
    split = 4

    if image:
        eval_img_dir = os.path.join(eval_dir, '%d_img' % roll_idx)
        os.system('mkdir -p ' + eval_img_dir)
        print('Save images to %s' % eval_img_dir)

    if video:
        eval_vid_path = os.path.join(eval_dir, '%d_vid.avi' % roll_idx)
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        print('Save video as %s' % eval_vid_path)
        fps = int(env.fps)
        print("fps:", fps)
        out = cv2.VideoWriter(eval_vid_path, fourcc, fps, (
            width * n_split + split * (n_split - 1), height))

    # can easily read the ground truth states directly from the episode . . .
    for i, start_idx in enumerate(states_predicted_idx):

        # get ground truth positions
        gt_state_dict = episode.get_observation(start_idx)

        # now render them
        env.set_object_positions_for_rendering(gt_state_dict)
        img_gt = env.render(mode='rgb_array')
        img_gt_PIL = numpy_to_PIL(img_gt)
        if render_human:
            env.render(mode='human')
            # time.sleep(1.0)

        # predicted positions
        pred_state_np = states_predicted_unnorm[i, :].cpu().numpy()
        predicted_state_dict = pusher_pose_slider_keypoints_from_tensor(pred_state_np)
        img_pred_PIL = pusher_slider_keypoints_image(config,
                                                     predicted_state_dict['pusher']['position'],
                                                     predicted_state_dict['keypoint_positions'])


        img_pred = np.array(img_pred_PIL)

        # env.set_object_positions_for_rendering(predicted_state_dict)
        # img_pred = env.render(mode='rgb_array')
        # img_pred_PIL = numpy_to_PIL(img_pred)
        #
        # if render_human:
        #     env.render(mode='human')
        #     # time.sleep(1.0)

        # blend the images
        img_merge_PIL = Image.blend(img_gt_PIL, img_pred_PIL, 0.5)

        # # show the predictions
        # if SHOW_BLENDED_IMAGE:
        #     print("showing PIL image")
        #     img_merge_PIL.show()
        #     time.sleep(1.0)

        # get numpy image
        img_merge = np.array(img_merge_PIL)

        img_output = np.zeros((
            img_gt.shape[0],
            img_gt.shape[1] * n_split + split * (n_split - 1), 3)).astype(np.uint8)

        img_output[:, :img_gt.shape[1]] = img_gt
        img_output[:, img_gt.shape[1] + split: img_gt.shape[1] * 2 + split] = img_pred
        img_output[:, (img_gt.shape[1] + split) * 2:] = img_merge

        if image:
            # convert to PIL then save
            # this step might be slow
            img_output_PIL = numpy_to_PIL(img_output)
            img_output_PIL.save(os.path.join(eval_img_dir, 'fig_%d.png' % i))

        if video:
            # convert from RGB to BGR since video writer wants BGR
            img_output_bgr = cv2.cvtColor(img_output, cv2.COLOR_RGB2BGR)
            out.write(img_output_bgr)

    if video:
        out.release()

def eval_dynamics(config,
                  train_dir,
                  eval_dir,
                  state_dict_path=None,
                  keypoint_observation=False,
                  debug=False,
                  render_human=False):

    # set random seed for reproduction
    set_seed(config['train']['random_seed'])

    tee = Tee(os.path.join(eval_dir, 'eval.log'), 'w')

    print(config)

    use_gpu = torch.cuda.is_available()

    '''
    model
    '''
    model_dy = DynaNetMLP(config)

    # print model #params
    print("model #params: %d" % count_trainable_parameters(model_dy))

    if state_dict_path is None:
        if config['eval']['eval_dy_epoch'] == -1:
            state_dict_path = os.path.join(train_dir, 'net_best_dy.pth')
        else:
            state_dict_path = os.path.join(
                train_dir, 'net_dy_epoch_%d_iter_%d.pth' % \
                (config['eval']['eval_dy_epoch'], config['eval']['eval_dy_iter']))

        print("Loading saved ckp from %s" % state_dict_path)

    model_dy.load_state_dict(torch.load(state_dict_path))
    model_dy.eval()

    if use_gpu:
        model_dy.cuda()

    criterionMSE = nn.MSELoss()
    bar = ProgressBar()

    st_idx = config['eval']['eval_st_idx']
    ed_idx = config['eval']['eval_ed_idx']

    # load the data
    episodes = load_episodes_from_config(config)

    # generate action/observation functions
    action_function = ActionFunctionFactory.function_from_config(config)
    observation_function = ObservationFunctionFactory.function_from_config(config)

    dataset = MultiEpisodeDataset(config,
                                  action_function=action_function,
                                  observation_function=observation_function,
                                  episodes=episodes,
                                  phase="valid")

    episode_names = dataset.get_episode_names()
    episode_names.sort()

    num_episodes = None
    # for backwards compatibility
    if "num_episodes" in config["eval"]:
        num_episodes = config["eval"]["num_episodes"]
    else:
        num_episodes = 10


    episode_list = []
    if debug:
        episode_list = [episode_names[0]]
    else:
        episode_list = episode_names[:num_episodes]

    for roll_idx, episode_name in enumerate(episode_list):
        print("episode_name", episode_name)
        if keypoint_observation:
            eval_episode_keypoint_observations(config,
                                               dataset,
                                               episode_name,
                                               roll_idx,
                                               model_dy,
                                               eval_dir,
                                               start_idx=9,
                                               n_prediction=30,
                                               render_human=render_human)
        else:
            eval_episode(config, dataset, episode_name, roll_idx, model_dy, eval_dir,
                         start_idx=9,
                         n_prediction=30,
                         render_human=render_human)

