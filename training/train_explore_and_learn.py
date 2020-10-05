import os
import time
import random

import numpy as np

from gym.spaces import Box
import transforms3d.derivations.eulerangles

# utils
from key_dynam.utils.utils import get_current_YYYY_MM_DD_hh_mm_ss_ms, get_project_root, load_pickle, save_yaml

# env
from key_dynam.envs.drake_pusher_slider import DrakePusherSliderEnv

# dynamics
from key_dynam.dynamics.utils import rand_int, count_trainable_parameters, Tee, AverageMeter, get_lr, to_np, set_seed
from key_dynam.dynamics.models_dy import DynaNetMLP, rollout_model

# planner
from key_dynam.planner.planner_factory import planner_from_config

# dataset
from key_dynam.dataset.episode_dataset import MultiEpisodeDataset
from key_dynam.dataset.episode_container import EpisodeContainer, MultiEpisodeContainer
from key_dynam.dataset.drake_sim_episode_reader import DrakeSimEpisodeReader
from key_dynam.dataset.function_factory import ObservationFunctionFactory, ActionFunctionFactory


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter




DEBUG = False



class DrakePusherSliderEpisodeCollector(object):

    def __init__(self, config):
        self._env = DrakePusherSliderEnv(config, visualize=False)
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

    def sample_pusher_velocity(self):
        angle_threshold = np.deg2rad(30)
        magnitude = 0.2

        angle_space = Box(-angle_threshold, angle_threshold)
        angle = angle_space.sample()

        pusher_velocity = magnitude * np.array([np.cos(angle), np.sin(angle)])

        return pusher_velocity




def collect_episodes(
    config,
    metadata,
    data_collector,
    n_episode=1,
    data_dir=None,
    visualize=False,
    episode_name=None,
    exploration_type='random',
    model_dy=None):

    """
    This collects a single episode by performing the following steps

    - sample initial conditions for environment
    - sample action from controller for environment
    - collect the interaction
    """

    if data_dir is None:
        data_dir = os.path.join(os.getcwd(), 'data')

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)


    for idx_episode in range(n_episode):

        start_time = time.time()
        print("collecting episode %d of %d" % (idx_episode + 1, n_episode))
        episode_name = "%s_idx_%d" % (get_current_YYYY_MM_DD_hh_mm_ss_ms(), idx_episode)

        env = data_collector._env
        env.reset()

        if visualize:
            print("setting target realtime rate 1.0")
            env.simulator.set_target_realtime_rate(1.0)


        q_pusher, q_slider = data_collector.get_pusher_slider_initial_positions()

        context = env.get_mutable_context()
        env.set_pusher_position(context, q_pusher)
        env.set_slider_position(context, q_slider)
        action_zero = np.zeros(2)
        env.step(action_zero, dt=1.0)    # to get the box to drop down

        # log data into EpisodeContainer object
        episode_container = EpisodeContainer()
        episode_container.set_config(env.config)
        episode_container.set_name(episode_name)
        episode_container.set_metadata(env.get_metadata())

        obs_prev = env.get_observation()

        num_timesteps = config['mpc']['num_timesteps'] + config['train']['n_history']
        for i in range(num_timesteps):

            # stand still for n_history timesteps
            if i < config['train']['n_history']:
                action = action_zero

            # this makes it easier to train our dynamics model
            # since we need to get a history of observations
            else:
                if exploration_type == 'random':
                    action = data_collector.sample_pusher_velocity()
                elif exploration_type == 'mppi':
                    action = data_collector.sample_pusher_velocity()

            # single sim time
            obs, reward, done, info = env.step(action)
            episode_container.add_obs_action(obs_prev, action)
            obs_prev = obs


            # terminate if outside boundary
            if not env.slider_within_boundary():
                print('slider outside boundary, terminating episode')
                break

            if not env.pusher_within_boundary():
                print('pusher ouside boundary, terminating episode')
                break

        # print("saving to disk")
        metadata['episodes'][episode_name] = dict()

        image_data_file = episode_container.save_images_to_hdf5(data_dir)
        non_image_data_file = episode_container.save_non_image_data_to_pickle(data_dir)

        metadata['episodes'][episode_name]['non_image_data_file'] = non_image_data_file
        metadata['episodes'][episode_name]['image_data_file'] = image_data_file

        # print("done saving to disk")
        elapsed = time.time() - start_time
        print("single episode took: %.2f seconds" %(elapsed))



def train_dynamics(
    config,
    train_dir,
    data_dir,
    model_dy,
    global_iteration,
    writer):

    # load the data
    multi_episode_dict = DrakeSimEpisodeReader.load_dataset(data_dir, load_image_data=False)

    '''
    for episode_name in list(multi_episode_dict.keys()):
        print("episode name", episode_name)
        episode = multi_episode_dict[episode_name]
        obs = episode.get_observation(34)
        print(obs)
    '''

    action_function = ActionFunctionFactory.function_from_config(config)
    observation_function = ObservationFunctionFactory.function_from_config(config)


    datasets = {}
    dataloaders = {}
    data_n_batches = {}
    for phase in ['train', 'valid']:
        print("Loading data for %s" % phase)
        datasets[phase] = MultiEpisodeDataset(
            config,
            action_function=action_function,
            observation_function=observation_function,
            episodes=multi_episode_dict,
            phase=phase)

        # print(config['train'])

        dataloaders[phase] = DataLoader(
            datasets[phase], batch_size=config['train']['batch_size'],
            shuffle=True if phase == 'train' else False,
            num_workers=config['train']['num_workers'], drop_last=True)

        data_n_batches[phase] = len(dataloaders[phase])

    use_gpu = torch.cuda.is_available()


    '''
    define model for dynamics prediction
    '''
    if model_dy is None:
        model_dy = DynaNetMLP(config)


    # criterion
    MSELoss = nn.MSELoss()
    L1Loss = nn.L1Loss()

    # optimizer
    params = model_dy.parameters()
    lr = float(config['train']['lr'])
    optimizer = optim.Adam(params, lr=lr, betas=(config['train']['adam_beta1'], 0.999))

    # setup scheduler
    sc = config['train']['lr_scheduler']
    scheduler = None

    if config['train']['lr_scheduler']['enabled']:
        if config['train']['lr_scheduler']['type'] == "ReduceLROnPlateau":
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=sc['factor'],
                patience=sc['patience'],
                threshold_mode=sc['threshold_mode'],
                cooldown= sc['cooldown'],
                verbose=True)
        elif config['train']['lr_scheduler']['type'] == "StepLR":
            step_size = config['train']['lr_scheduler']['step_size']
            gamma = config['train']['lr_scheduler']['gamma']
            scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        else:
            raise ValueError("unknown scheduler type: %s" %(config['train']['lr_scheduler']['type']))

    if use_gpu:
        print("using gpu")
        model_dy = model_dy.cuda()


    best_valid_loss = np.inf
    counters = {'train': 0, 'valid': 0}


    try:
        for epoch in range(config['train']['n_epoch']):
            phases = ['train', 'valid']

            writer.add_scalar("Training Params/epoch", epoch, global_iteration)
            for phase in phases:
                model_dy.train(phase == 'train')

                meter_loss_rmse = AverageMeter()
                step_duration_meter = AverageMeter()

                # bar = ProgressBar(max_value=data_n_batches[phase])
                loader = dataloaders[phase]

                for i, data in enumerate(loader):

                    loss_container = dict() # store the losses for this step

                    step_start_time = time.time()

                    global_iteration += 1
                    counters[phase] += 1

                    with torch.set_grad_enabled(phase == 'train'):
                        n_his, n_roll = config['train']['n_history'], config['train']['n_rollout']
                        n_samples = n_his + n_roll

                        if DEBUG:
                            print("global iteration: %d" % global_iteration)
                            print("n_samples", n_samples)


                        # [B, n_samples, obs_dim]
                        observations = data['observations']

                        # [B, n_samples, action_dim]
                        actions = data['actions']
                        B = actions.shape[0]

                        if use_gpu:
                            observations = observations.cuda()
                            actions = actions.cuda()

                        # states, actions = data
                        assert actions.shape[1] == n_samples
                        loss_mse = 0.


                        # we don't have any visual observations, so states are observations
                        states = observations

                        # [B, n_his, state_dim]
                        state_init = states[:, :n_his]

                        # We want to rollout n_roll steps
                        # actions = [B, n_his + n_roll, -1]
                        # so we want action_seq.shape = [B, n_roll, -1]
                        action_start_idx = 0
                        action_end_idx = n_his + n_roll - 1
                        action_seq = actions[:, action_start_idx:action_end_idx, :]

                        if DEBUG:
                            print("states.shape", states.shape)
                            print("state_init.shape", state_init.shape)
                            print("actions.shape", actions.shape)
                            print("action_seq.shape", action_seq.shape)

                        # try using models_dy.rollout_model instead of doing this manually
                        rollout_data = rollout_model(
                            state_init=state_init,
                            action_seq=action_seq,
                            dynamics_net=model_dy,
                            compute_debug_data=False)

                        # [B, n_roll, state_dim]
                        state_rollout_pred = rollout_data['state_pred']

                        # [B, n_roll, state_dim]
                        state_rollout_gt = states[:, n_his:]


                        if DEBUG:
                            print("state_rollout_gt.shape", state_rollout_gt.shape)
                            print("state_rollout_pred.shape", state_rollout_pred.shape)

                        # the loss function is between
                        # [B, n_roll, state_dim]
                        state_pred_err = state_rollout_pred - state_rollout_gt

                        # everything is in 3D space now so no need to do any scaling
                        # all the losses would be in meters . . . .
                        loss_mse = MSELoss(state_rollout_pred, state_rollout_gt)
                        loss_l1 = L1Loss(state_rollout_pred, state_rollout_gt)
                        meter_loss_rmse.update(np.sqrt(loss_mse.item()), B)


                        # compute losses at final step of the rollout
                        mse_final_step = MSELoss(state_rollout_pred[:, -1, :], state_rollout_gt[:, -1, :])
                        l2_final_step = torch.norm(state_pred_err[:, -1], dim=-1).mean()
                        l1_final_step = L1Loss(state_rollout_pred[:, -1, :], state_rollout_gt[:, -1, :])

                        loss_container['mse'] = loss_mse
                        loss_container['l1'] = loss_l1
                        loss_container['mse_final_step'] = mse_final_step
                        loss_container['l1_final_step'] = l1_final_step
                        loss_container['l2_final_step'] = l2_final_step

                    step_duration_meter.update(time.time() - step_start_time)
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss_mse.backward()
                        optimizer.step()

                    if i % config['train']['log_per_iter'] == 0:
                        log = '%s %d [%d/%d][%d/%d] LR: %.6f' % (
                            phase, global_iteration, epoch, config['train']['n_epoch'], i, data_n_batches[phase],
                            get_lr(optimizer))
                        log += ', rmse: %.6f (%.6f)' % (
                            np.sqrt(loss_mse.item()), meter_loss_rmse.avg)

                        log += ', step time %.6f' % (step_duration_meter.avg)
                        step_duration_meter.reset()


                        print(log)

                        # log data to tensorboard
                        # only do it once we have reached 100 iterations
                        if global_iteration > 100:
                            writer.add_scalar("Params/learning rate", get_lr(optimizer), global_iteration)
                            writer.add_scalar("Loss_MSE/%s" % (phase), loss_mse.item(), global_iteration)
                            writer.add_scalar("L1/%s" % (phase), loss_l1.item(), global_iteration)
                            writer.add_scalar("RMSE average loss/%s" % (phase), meter_loss_rmse.avg, global_iteration)

                            writer.add_scalar("n_taj", len(multi_episode_dict), global_iteration)

                            for loss_type, loss_obj in loss_container.items():
                                plot_name = "Loss/%s/%s" % (loss_type, phase)
                                writer.add_scalar(plot_name, loss_obj.item(), global_iteration)

                    if phase == 'train' and global_iteration % config['train']['ckp_per_iter'] == 0:
                        save_model(model_dy, '%s/net_dy_iter_%d' % (train_dir, global_iteration))



                log = '%s %d [%d/%d] Loss: %.6f, Best valid: %.6f' % (
                    phase, global_iteration, epoch, config['train']['n_epoch'], meter_loss_rmse.avg, best_valid_loss)
                print(log)


                if phase == "train":
                    if (scheduler is not None) and (config['train']['lr_scheduler']['type'] == "StepLR"):
                        scheduler.step()

                if phase == 'valid':
                    if (scheduler is not None) and (config['train']['lr_scheduler']['type'] == "ReduceLROnPlateau"):
                        scheduler.step(meter_loss_rmse.avg)

                    if meter_loss_rmse.avg < best_valid_loss:
                        best_valid_loss = meter_loss_rmse.avg
                        save_model(model_dy, '%s/net_best_dy' % (train_dir))

                writer.flush() # flush SummaryWriter events to disk

    except KeyboardInterrupt:
        # save network if we have a keyboard interrupt
        save_model(model_dy, '%s/net_dy_iter_%d_keyboard_interrupt' % (train_dir, global_iteration))
        writer.flush() # flush SummaryWriter events to disk


    return model_dy, global_iteration



def save_model(model, save_base_path):
    # save both the model in binary form, and also the state dict
    torch.save(model.state_dict(), save_base_path + "_state_dict.pth")
    torch.save(model, save_base_path + "_model.pth")



def train_explore_and_learn(config,
                            train_dir,  # str: directory to save output
                            data_dir,
                            visualize=False
                            ):

    # set random seed for reproduction
    set_seed(config['train_explore_and_learn']['random_seed'])

    tensorboard_dir = os.path.join(train_dir, "tensorboard")
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    writer = SummaryWriter(log_dir=tensorboard_dir)

    # save the config
    save_yaml(config, os.path.join(train_dir, "config.yaml"))

    print(config)

    num_exploration_rounds = config['train_explore_and_learn']['num_exploration_rounds']
    num_episodes_per_exploration_round = config['train_explore_and_learn']['num_episodes_per_exploration_round']
    num_timesteps = config['train_explore_and_learn']['num_timesteps']


    # model_folder = os.path.join(train_dir, "../2020-04-05-23-00-30-887903")
    # model_file = os.path.join(model_folder, "net_best_dy_model.pth")
    # model_dy = torch.load(model_file)
    model_dy = None

    global_iteration = 0


    ##### setup to store the dataset
    metadata = dict()
    metadata['episodes'] = dict()

    # data collector
    data_collector = DrakePusherSliderEpisodeCollector(config)


    ##### explore and learn
    for idx_exploration_round in range(num_exploration_rounds):

        print("Exploration round %d / %d" % (
            idx_exploration_round, num_exploration_rounds))

        ### exploration

        if idx_exploration_round == 0:
            # initial exploration
            exploration_type = 'random'
        else:
            exploration_type = 'mppi'

        collect_episodes(
            config,
            metadata,
            data_collector,
            num_episodes_per_exploration_round,
            data_dir,
            visualize,
            exploration_type,
            model_dy=None if exploration_type=='random' else model_dy)

        save_yaml(metadata, os.path.join(data_dir, 'metadata.yaml'))


        ### optimize the dynamics model
        model_dy, global_iteration = train_dynamics(
            config,
            train_dir,
            data_dir,
            model_dy,
            global_iteration,
            writer)



