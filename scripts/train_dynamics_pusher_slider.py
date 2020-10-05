import os
import random
import numpy as np

# from progressbar import ProgressBar

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from key_dynam.dataset.episode_dataset import MultiEpisodeDataset
from key_dynam.dynamics.models_dy import DynaNetMLP
from key_dynam.dataset.episode_reader import PyMunkEpisodeReader
from key_dynam.dynamics.utils import rand_int, count_trainable_parameters, Tee, AverageMeter, get_lr, to_np, set_seed
from key_dynam.utils.utils import get_project_root, load_pickle
from key_dynam.dynamics.data_normalizer import DataNormalizer
from key_dynam.dataset.function_factory import ObservationFunctionFactory, ActionFunctionFactory
from key_dynam.dataset.utils import load_episodes_from_config
from key_dynam.utils.utils import save_yaml

def save_model(model, save_base_path):
    # save both the model in binary form, and also the state dict
    torch.save(model.state_dict(), save_base_path + "_state_dict.pth")
    torch.save(model, save_base_path + "_model.pth")

def train_dynamics(config,
                   train_dir, # str: directory to save output
                   ):

    # set random seed for reproduction
    set_seed(config['train']['random_seed'])

    st_epoch = config['train']['resume_epoch'] if config['train']['resume_epoch'] > 0 else 0
    tee = Tee(os.path.join(train_dir, 'train_st_epoch_%d.log' % st_epoch), 'w')

    tensorboard_dir = os.path.join(train_dir, "tensorboard")
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    writer = SummaryWriter(log_dir=tensorboard_dir)

    # save the config
    save_yaml(config, os.path.join(train_dir, "config.yaml"))


    print(config)

    # load the data
    episodes = load_episodes_from_config(config)

    action_function = ActionFunctionFactory.function_from_config(config)
    observation_function = ObservationFunctionFactory.function_from_config(config)

    datasets = {}
    dataloaders = {}
    data_n_batches = {}
    for phase in ['train', 'valid']:
        print("Loading data for %s" % phase)
        datasets[phase] = MultiEpisodeDataset(config,
                                              action_function=action_function,
                                              observation_function=observation_function,
                                              episodes=episodes,
                                              phase=phase)

        dataloaders[phase] = DataLoader(
            datasets[phase], batch_size=config['train']['batch_size'],
            shuffle=True if phase == 'train' else False,
            num_workers=config['train']['num_workers'])

        data_n_batches[phase] = len(dataloaders[phase])

    use_gpu = torch.cuda.is_available()

    # compute normalization parameters if not starting from pre-trained network . . .


    '''
    define model for dynamics prediction
    '''
    model_dy = None


    if config['train']['resume_epoch'] >= 0:
        # if resume from a pretrained checkpoint
        state_dict_path = os.path.join(
            train_dir, 'net_dy_epoch_%d_iter_%d_state_dict.pth' % (
                config['train']['resume_epoch'], config['train']['resume_iter']))
        print("Loading saved ckp from %s" % state_dict_path)

        # why is this needed if we already do torch.load???
        model_dy.load_state_dict(torch.load(state_dict_path))

        # don't we also need to load optimizer state from pre-trained???
    else:
        # not starting from pre-trained create the network and compute the
        # normalization parameters
        model_dy = DynaNetMLP(config)

        # compute normalization params
        stats = datasets["train"].compute_dataset_statistics()

        obs_mean = stats['observations']['mean']
        obs_std = stats['observations']['std']
        observations_normalizer = DataNormalizer(obs_mean, obs_std)

        action_mean = stats['actions']['mean']
        action_std = stats['actions']['std']
        actions_normalizer = DataNormalizer(action_mean, action_std)

        model_dy.action_normalizer = actions_normalizer
        model_dy.state_normalizer = observations_normalizer

    print("model_dy #params: %d" % count_trainable_parameters(model_dy))

    # criterion
    criterionMSE = nn.MSELoss()

    # optimizer
    params = model_dy.parameters()
    optimizer = optim.Adam(params, lr=config['train']['lr'], betas=(config['train']['adam_beta1'], 0.999))
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.9, patience=10, verbose=True)

    if use_gpu:
        model_dy = model_dy.cuda()


    best_valid_loss = np.inf
    global_iteration = 0

    epoch_counter_external = 0

    try:
        for epoch in range(st_epoch, config['train']['n_epoch']):
            phases = ['train', 'valid']
            epoch_counter_external = epoch

            writer.add_scalar("Training Params/epoch", epoch, global_iteration)
            for phase in phases:
                model_dy.train(phase == 'train')

                meter_loss_rmse = AverageMeter()

                # bar = ProgressBar(max_value=data_n_batches[phase])
                loader = dataloaders[phase]

                for i, data in enumerate(loader):

                    global_iteration += 1

                    with torch.set_grad_enabled(phase == 'train'):
                        n_his, n_roll = config['train']['n_history'], config['train']['n_rollout']
                        n_samples = n_his + n_roll

                        if config['env']['type'] in ['PusherSlider']:
                            states = data['observations']
                            actions = data['actions']

                            if use_gpu:
                                states = states.cuda()
                                actions = actions.cuda()

                            # states, actions = data
                            assert states.size(1) == n_samples

                            # normalize states and actions once for entire rollout
                            states = model_dy.state_normalizer.normalize(states)
                            actions = model_dy.action_normalizer.normalize(actions)

                            B = states.size(0)
                            loss_mse = 0.

                            # state_cur: B x n_his x state_dim
                            state_cur = states[:, :n_his]

                            for j in range(n_roll):

                                state_des = states[:, n_his + j]

                                # action_cur: B x n_his x action_dim
                                action_cur = actions[:, j : j + n_his] if actions is not None else None

                                # state_pred: B x state_dim
                                # state_cur: B x n_his x state_dim
                                # state_pred: B x state_dim
                                state_pred = model_dy(state_cur, action_cur)


                                loss_mse_cur = criterionMSE(state_pred, state_des)
                                loss_mse += loss_mse_cur / n_roll

                                # update state_cur
                                # state_pred.unsqueeze(1): B x 1 x state_dim
                                state_cur = torch.cat([state_cur[:, 1:], state_pred.unsqueeze(1)], 1)

                            meter_loss_rmse.update(np.sqrt(loss_mse.item()), B)

                    if phase == 'train':
                        optimizer.zero_grad()
                        loss_mse.backward()
                        optimizer.step()

                    if i % config['train']['log_per_iter'] == 0:
                        log = '%s [%d/%d][%d/%d] LR: %.6f' % (
                            phase, epoch, config['train']['n_epoch'], i, data_n_batches[phase],
                            get_lr(optimizer))
                        log += ', rmse: %.6f (%.6f)' % (
                            np.sqrt(loss_mse.item()), meter_loss_rmse.avg)

                        print(log)

                        # log data to tensorboard
                        # only do it once we have reached 500 iterations
                        if global_iteration > 500:
                            writer.add_scalar("Params/learning rate", get_lr(optimizer), global_iteration)
                            writer.add_scalar("Loss/train", loss_mse.item(), global_iteration)
                            writer.add_scalar("RMSE average loss/train", meter_loss_rmse.avg, global_iteration)

                    if phase == 'train' and i % config['train']['ckp_per_iter'] == 0:
                        save_model(model_dy, '%s/net_dy_epoch_%d_iter_%d' % (train_dir, epoch, i))



                log = '%s [%d/%d] Loss: %.6f, Best valid: %.6f' % (
                    phase, epoch, config['train']['n_epoch'], meter_loss_rmse.avg, best_valid_loss)
                print(log)

                if phase == 'valid':
                    scheduler.step(meter_loss_rmse.avg)
                    writer.add_scalar("RMSE average loss/valid", meter_loss_rmse.avg, global_iteration)
                    if meter_loss_rmse.avg < best_valid_loss:
                        best_valid_loss = meter_loss_rmse.avg
                        save_model(model_dy, '%s/net_best_dy' % (train_dir))

                writer.flush() # flush SummaryWriter events to disk

    except KeyboardInterrupt:
        # save network if we have a keyboard interrupt
        save_model(model_dy, '%s/net_dy_epoch_%d_keyboard_interrupt' % (train_dir, epoch_counter_external))
        writer.flush() # flush SummaryWriter events to disk
