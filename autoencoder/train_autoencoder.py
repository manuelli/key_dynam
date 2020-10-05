import os
import time
import random
import numpy as np
import functools
import matplotlib.pyplot as plt

import pydrake

from progressbar import ProgressBar

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from key_dynam.utils.utils import get_project_root, save_yaml, load_yaml, get_current_YYYY_MM_DD_hh_mm_ss_ms, \
    get_data_root, convert_float_image_to_uint8
from key_dynam.transporter.utils import rand_int, count_parameters, AverageMeter, get_lr, to_np
from key_dynam.utils import torch_utils
from key_dynam.dynamics.utils import set_seed
from key_dynam.autoencoder.dataset import AutoencoderImageDataset, spatial_autoencoder_image_preprocessing, AutoencoderImagePreprocessFunctionFactory
from key_dynam.autoencoder.autoencoder_models import SpatialAutoencoder, ConvolutionalAutoencoder


def train_autoencoder(config,
                      train_dir,
                      ckp_dir=None,
                      multi_episode_dict=None,
                      type=None,  # ["SpatialAutoencoder", . . .]
                      ):
    assert multi_episode_dict is not None

    if ckp_dir is None:
        ckp_dir = os.path.join(train_dir, 'checkpoints')

    if not os.path.exists(ckp_dir):
        os.makedirs(ckp_dir)

    tensorboard_dir = os.path.join(train_dir, "tensorboard")
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    images_dir = os.path.join(train_dir, 'images')
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    # save the config
    save_yaml(config, os.path.join(train_dir, 'config.yaml'))

    # set random seed for reproduction
    set_seed(config['train_autoencoder']['random_seed'])
    camera_names = [config['perception']['camera_name']]

    model = None
    image_preprocess_func = None
    if type == "SpatialAutoencoder":
        model = SpatialAutoencoder.from_global_config(config)
        image_preprocess_func = functools.partial(spatial_autoencoder_image_preprocessing,
                                                  H_in=model.input_image_shape[0],
                                                  W_in=model.input_image_shape[1],
                                                  H_out=model.output_image_shape[0],
                                                  W_out=model.output_image_shape[1])
    elif type == "ConvolutionalAutoencoder":
        model = ConvolutionalAutoencoder.from_global_config(config)
        image_preprocess_func = AutoencoderImagePreprocessFunctionFactory.convolutional_autoencoder(config)
    else:
        raise ValueError("unknown model type: %s" % (type))

    writer = SummaryWriter(log_dir=tensorboard_dir)

    # only use images from this specific config

    ### data
    datasets = {}
    dataloaders = {}
    for phase in ['train', 'valid']:
        datasets[phase] = AutoencoderImageDataset(config,
                                                  phase=phase,
                                                  episodes=multi_episode_dict,
                                                  camera_names=camera_names,
                                                  image_preprocess_func=image_preprocess_func)

        dataloaders[phase] = DataLoader(
            datasets[phase], batch_size=config['train_autoencoder']['batch_size'],
            shuffle=True if phase == 'train' else False,
            num_workers=config['train_autoencoder']['batch_size'])

    use_gpu = torch.cuda.is_available()

    params = model.parameters()
    optimizer = optim.Adam(
        params, lr=float(config['train_autoencoder']['lr']),
        betas=(config['train_autoencoder']['adam_beta1'], 0.999))

    scheduler = None
    if config['train_autoencoder']['lr_scheduler']['enabled']:
        scheduler = ReduceLROnPlateau(
            optimizer, 'min', factor=0.6, patience=2, verbose=True)

    if use_gpu:
        model = model.cuda()

    best_valid_loss = np.inf
    global_iteration = 0
    log_fout = open(os.path.join(ckp_dir, 'log.txt'), 'w')

    # criterion
    criterionMSE = nn.MSELoss()


    # a little test
    if False:
        data = datasets['train'][0]
        print(data.keys())

        print("data['target_tensor'].shape", data['target_tensor'].shape)
        print("data['target_mask'].shape", data['target_mask'].shape)
        fig = plt.figure()
        ax = fig.subplots(2)
        target_img = data['target']
        print("target_img.dtype", target_img.dtype)
        ax[0].imshow(data['input'])
        ax[1].imshow(data['target'], cmap='gray', vmin=0, vmax=255)
        plt.show()

        quit()

    # a little test
    if False:
        data = datasets['train'][0]
        print(data.keys())

        print("data['target_tensor'].shape", data['target_tensor'].shape)
        print("data['target_mask'].shape", data['target_mask'].shape)
        fig = plt.figure()
        ax = fig.subplots(2)
        target_img = data['target']
        target_tensor = data['target_tensor'].unsqueeze(0)
        target_tensor_np = torch_utils.convert_torch_image_to_numpy(target_tensor).squeeze()
        print("target_img.dtype", target_img.dtype)
        ax[0].imshow(target_img)
        ax[1].imshow(target_tensor_np)
        plt.show()

        quit()

    counters = {'train': 0, 'valid': 0}


    n_epoch = config['train_autoencoder']['n_epoch']
    for epoch in range(n_epoch):
        phases = ['train', 'valid']

        writer.add_scalar("Training Params/epoch", epoch, global_iteration)

        for phase in phases:
            model.train(phase == 'train')

            meter_loss = AverageMeter()

            loader = dataloaders[phase]
            bar = ProgressBar(max_value=len(loader))

            step_duration_meter = AverageMeter()
            epoch_start_time = time.time()
            prev_time = time.time()
            print("\n\n")
            for i, data in bar(enumerate(loader)):
                loss_container = dict() # store the losses for this step
                counters[phase] += 1

                with torch.set_grad_enabled(phase == 'train'):
                    input = data['input_tensor']
                    target = data['target_tensor']

                    if use_gpu:
                        input = input.cuda()
                        target = target.cuda()

                    out = model(input)
                    target_pred = out['output']

                    # print("target.shape", target.shape)
                    # print("target_pred.shape", target_pred.shape)

                    # reconstruction loss
                    l2_recon = criterionMSE(target, target_pred)
                    loss_container['l2_recon'] = l2_recon


                    # loss_masked
                    # [B, H', W']
                    mask = data['target_mask'].to(target.device)
                    mask_idx = mask > 0

                    # convert to BHWC ordering so we can directly index
                    target_masked = target.permute(0, 2, 3, 1)[mask_idx]
                    target_pred_masked = target_pred.permute(0, 2, 3, 1)[mask_idx]

                    # print('target_masked.shape', target_masked.shape)
                    # print("target_pred_masked.shape", target_pred_masked.shape)
                    l2_recon_masked = criterionMSE(target_masked, target_pred_masked)

                    loss_container['l2_recon_masked'] = l2_recon_masked

                    # compute the loss
                    loss = 0
                    for key, val in config['train_autoencoder']['loss_function'].items():
                        if val['enabled']:
                            loss += loss_container[key] * val['weight']

                    meter_loss.update(loss.item())

                step_duration_meter.update(time.time() - prev_time)
                prev_time = time.time()

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(params, 1)
                    optimizer.step()

                if global_iteration > 100:
                    writer.add_scalar("Params/learning rate", get_lr(optimizer), global_iteration)
                    # writer.add_scalar("Loss/%s" % (phase), loss.item(), global_iteration)

                    writer.add_scalar("Loss_train/%s" % (phase), loss.item(), counters[phase])

                    for loss_type, loss_obj in loss_container.items():
                        plot_name = "Loss/%s/%s" % (loss_type, phase)
                        writer.add_scalar(plot_name, loss_obj.item(), counters[phase])

                if i % config['train_autoencoder']['log_per_iter'] == 0:
                    log = '%s [%d/%d][%d/%d] LR: %.6f, Loss: %.6f (%.6f)' % (
                        phase, epoch, n_epoch, i, len(loader), get_lr(optimizer),
                        loss.item(), meter_loss.avg)

                    log += ', step time %.6f' % (step_duration_meter.avg)
                    step_duration_meter.reset()

                    print(log)

                    log_fout.write(log + '\n')
                    log_fout.flush()

                if phase == 'train' and i % config['train_autoencoder']['ckp_per_iter'] == 0:
                    torch.save(
                        model.state_dict(),
                        '%s/net_kp_epoch_%d_iter_%d.pth' % (ckp_dir, epoch, i))

                if i % config['train_autoencoder']['img_save_per_iter'] == 0:

                    nrows = 4
                    ncols = 2
                    fig_width = 5

                    B, _, H, W = target.shape
                    fig_height = fig_width * ((nrows * H) / (ncols * W))
                    figsize = (fig_width, fig_height)
                    fig = plt.figure(figsize=figsize)

                    ax = fig.subplots(nrows=nrows, ncols=ncols)

                    target_np = torch_utils.convert_torch_image_to_numpy(target)
                    target_pred_np = torch_utils.convert_torch_image_to_numpy(target_pred)

                    for n in range(nrows):
                        ax[n, 0].imshow(target_np[n])
                        ax[n, 1].imshow(target_pred_np[n])

                    save_file = os.path.join(images_dir,
                                             '%s_epoch_%d_iter_%d.png' %(phase, epoch, i))

                    fig.savefig(save_file)
                    plt.close(fig)


                writer.flush()  # flush SummaryWriter events to disk
                global_iteration += 1

            log = '%s [%d/%d] Loss: %.6f, Best valid: %.6f' % (
                phase, epoch, n_epoch, meter_loss.avg, best_valid_loss)
            print(log)
            print("Epoch Duration:", time.time() - epoch_start_time)
            log_fout.write(log + '\n')
            log_fout.flush()

            if phase == 'valid':
                if scheduler is not None:
                    scheduler.step(meter_loss.avg)
                if meter_loss.avg < best_valid_loss:
                    best_valid_loss = meter_loss.avg

                    torch.save(model.state_dict(), '%s/net_best.pth' % ckp_dir)

    log_fout.close()
