import os
import random
import numpy as np

from progressbar import ProgressBar
from easydict import EasyDict as edict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from key_dynam.dataset.episode_dataset_bak import MultiEpisodeDataset
from key_dynam.dynamics.models_dy import DynaNetMLP
from key_dynam.dynamics.utils import rand_int, count_trainable_parameters, Tee, AverageMeter, get_lr, to_np, set_seed



def train_dynamics(config, data_path, train_dir):

    # access dict values as attributes
    config = edict(config)

    # set random seed for reproduction
    set_seed(config.train.random_seed)

    st_epoch = config.train.resume_epoch if config.train.resume_epoch > 0 else 0
    tee = Tee(os.path.join(train_dir, 'train_st_epoch_%d.log' % st_epoch), 'w')

    print(config)

    datasets = {}
    dataloaders = {}
    data_n_batches = {}
    for phase in ['train', 'valid']:
        print("Loading data for %s" % phase)
        datasets[phase] = MultiEpisodeDataset(config, data_path, phase=phase)

        dataloaders[phase] = DataLoader(
            datasets[phase], batch_size=config.train.batch_size,
            shuffle=True if phase == 'train' else False,
            num_workers=config.train.num_workers)

        data_n_batches[phase] = len(dataloaders[phase])

    use_gpu = torch.cuda.is_available()


    '''
    define model for dynamics prediction
    '''
    model_dy = DynaNetMLP(config)
    print("model_dy #params: %d" % count_trainable_parameters(model_dy))

    if config.train.resume_epoch >= 0:
        # if resume from a pretrained checkpoint
        model_dy_path = os.path.join(
            train_dir, 'net_dy_epoch_%d_iter_%d.pth' % (
                config.train.resume_epoch, config.train.resume_iter))
        print("Loading saved ckp from %s" % model_dy_path)
        model_dy.load_state_dict(torch.load(model_dy_path))


    # criterion
    criterionMSE = nn.MSELoss()

    # optimizer
    params = model_dy.parameters()
    optimizer = optim.Adam(params, lr=config.train.lr, betas=(config.train.adam_beta1, 0.999))
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.9, patience=10, verbose=True)

    if use_gpu:
        model_dy = model_dy.cuda()


    best_valid_loss = np.inf

    for epoch in range(st_epoch, config.train.n_epoch):
        phases = ['train', 'valid']

        for phase in phases:
            model_dy.train(phase == 'train')

            meter_loss_rmse = AverageMeter()

            bar = ProgressBar(max_value=data_n_batches[phase])
            loader = dataloaders[phase]

            for i, data in bar(enumerate(loader)):

                if use_gpu:
                    if isinstance(data, list):
                        data = [d.cuda() for d in data]
                    else:
                        data = data.cuda()

                with torch.set_grad_enabled(phase == 'train'):
                    n_his, n_roll = config.train.n_history, config.train.n_rollout
                    n_samples = n_his + n_roll

                    if config.env.type in ['PusherSlider']:
                        states, actions = data
                        assert states.size(1) == n_samples

                        B = states.size(0)
                        loss_mse = 0.

                        # state_cur: B x n_his x state_dim
                        state_cur = states[:, :n_his]

                        for j in range(n_roll):

                            state_des = states[:, n_his + j]

                            # action_cur: B x n_his x action_dim
                            action_cur = actions[:, j : j + n_his] if actions is not None else None

                            # state_pred: B x state_dim
                            state_pred = model_dy(state_cur, action_cur)

                            loss_mse_cur = criterionMSE(state_pred, state_des)
                            loss_mse += loss_mse_cur / config.train.n_rollout

                            # update state_cur
                            state_cur = torch.cat([state_cur[:, 1:], state_pred.unsqueeze(1)], 1)

                        meter_loss_rmse.update(np.sqrt(loss_mse.item()), B)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss_mse.backward()
                    optimizer.step()

                if i % config.train.log_per_iter == 0:
                    log = '%s [%d/%d][%d/%d] LR: %.6f' % (
                        phase, epoch, config.train.n_epoch, i, data_n_batches[phase],
                        get_lr(optimizer))
                    log += ', rmse: %.6f (%.6f)' % (
                        np.sqrt(loss_mse.item()), meter_loss_rmse.avg)

                    print(log)

                if phase == 'train' and i % config.train.ckp_per_iter == 0:
                    torch.save(model_dy.state_dict(), '%s/net_dy_epoch_%d_iter_%d.pth' % (train_dir, epoch, i))

            log = '%s [%d/%d] Loss: %.6f, Best valid: %.6f' % (
                phase, epoch, config.train.n_epoch, meter_loss_rmse.avg, best_valid_loss)
            print(log)

            if phase == 'valid':
                scheduler.step(meter_loss_rmse.avg)
                if meter_loss_rmse.avg < best_valid_loss:
                    best_valid_loss = meter_loss_rmse.avg
                    torch.save(model_dy.state_dict(), '%s/net_best_dy.pth' % (train_dir))

