import os
import time
import random
import numpy as np
import cv2
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


from key_dynam.utils.utils import get_project_root, save_yaml, load_yaml, get_current_YYYY_MM_DD_hh_mm_ss_ms, get_data_root
from key_dynam.transporter.models_kp import Transporter
from key_dynam.transporter.utils import rand_int, count_parameters, AverageMeter, get_lr, to_np, visualize_transporter_output
from key_dynam.dynamics.utils import set_seed
from key_dynam.transporter.dataset import ImageTupleDataset
from key_dynam.dense_correspondence.dc_drake_sim_episode_reader import DCDrakeSimEpisodeReader



def train_transporter(config,
                      train_dir,
                      ckp_dir=None,
                      multi_episode_dict=None,
                      ):

    assert multi_episode_dict is not None

    if ckp_dir is None:
        ckp_dir = os.path.join(train_dir)

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
    set_seed(config['train_transporter']['random_seed'])

    writer = SummaryWriter(log_dir=tensorboard_dir)

    # only use images from this specific config
    camera_names = [config['perception']['camera_name']]

    ### data
    datasets = {}
    dataloaders = {}
    for phase in ['train', 'valid']:
        datasets[phase] = ImageTupleDataset(config,
                                            phase=phase,
                                            episodes=multi_episode_dict,
                                            tuple_size=2,
                                            camera_names=camera_names)

        dataloaders[phase] = DataLoader(
            datasets[phase], batch_size=config['train_transporter']['batch_size'],
            shuffle=True if phase == 'train' else False,
            num_workers=config['train_transporter']['batch_size'])

    use_gpu = torch.cuda.is_available()


    crop_enabled = datasets['train'].crop_enabled
    rgb_tensor_key = None
    if crop_enabled:
        rgb_image_key = "rgb_crop"
        rgb_tensor_key = "rgb_crop_tensor"
    else:
        rgb_image_key = "rgb_masked_scaled"
        rgb_tensor_key = "rgb_masked_scaled_tensor"



    if False:
        dataset = datasets["train"]

        dataset_size = len(dataset)
        print("len(dataset)", len(dataset))
        print("len(dataset._image_dataset)", len(dataset._image_dataset))

        print("len(dataset['valid'])", len(datasets['valid']))
        print("len(dataset['train'])", len(datasets['train']))

        print("dataset.crop_enabled", dataset.crop_enabled)

        data = dataset[0]
        print("data.keys()", data.keys())
        print("data[0].keys()", data[0].keys())

        # rgb_crop_tensor = data[0]['rgb_crop_tensor']
        # print("rgb_crop_tensor.max()", rgb_crop_tensor.max())
        # print("rgb_crop_tensor.min()", rgb_crop_tensor.min())
        #
        # rgb_image = data[0]['rgb_masked_scaled']
        # print("rgb_crop.dtype", rgb_image.dtype)
        # print("rgb_image.shape", rgb_image.shape)

        rgb_image = data[0][rgb_image_key]
        rgb_tensor = data[0][rgb_tensor_key]
        print("rgb_image.shape", rgb_image.shape)
        print("rgb_tensor.shape", rgb_tensor.shape)

        plt.figure()
        # plt.imshow(rgb_image)
        plt.imshow(data[0][rgb_image_key])
        plt.show()
        quit()
        #

    ### model
    model_kp = Transporter(config, use_gpu=use_gpu)
    print("model_kp #params: %d" % count_parameters(model_kp))

    if config['train_transporter']['resume_epoch'] >= 0:
        model_kp_path = os.path.join(
            ckp_dir, 'net_kp_epoch_%d_iter_%d.pth' % (
                config['train_transporter']['resume_epoch'],
                config['train_transporter']['resume_iter']
            ))
        print("Loading saved ckp from %s" % model_kp_path)
        model_kp.load_state_dict(torch.load(model_kp_path))

    # criterion
    criterionMSE = nn.MSELoss()

    # optimizer
    params = model_kp.parameters()
    optimizer = optim.Adam(
        params, lr=float(config['train_transporter']['lr']),
        betas=(config['train_transporter']['adam_beta1'], 0.999))
    scheduler = ReduceLROnPlateau(
        optimizer, 'min', factor=0.6, patience=2, verbose=True)

    if use_gpu:
        model_kp = model_kp.cuda()


    best_valid_loss = np.inf
    global_iteration = 0
    log_fout = open(os.path.join(ckp_dir, 'log.txt'), 'w')

    n_epoch = config['train_transporter']['n_epoch']
    for epoch in range(n_epoch):
        phases = ['train', 'valid']

        writer.add_scalar("Training Params/epoch", epoch, global_iteration)

        for phase in phases:
            model_kp.train(phase == 'train')

            meter_loss = AverageMeter()

            loader = dataloaders[phase]
            bar = ProgressBar(max_value=len(loader))

            for i, data in bar(enumerate(loader)):

                with torch.set_grad_enabled(phase == 'train'):
                    src = data[0][rgb_tensor_key]
                    des = data[1][rgb_tensor_key]

                    if use_gpu:
                        src = src.cuda()
                        des = des.cuda()

                    des_pred = model_kp(src, des)

                    # reconstruction loss
                    loss = criterionMSE(des_pred, des)
                    meter_loss.update(loss.item(), src.size(0))

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                if global_iteration > 100:
                    writer.add_scalar("Params/learning rate", get_lr(optimizer), global_iteration)
                    writer.add_scalar("Loss/%s" % (phase), loss.item(), global_iteration)


                if i % config['train_transporter']['log_per_iter'] == 0:
                    log = '%s [%d/%d][%d/%d] LR: %.6f, Loss: %.6f (%.6f)' % (
                        phase, epoch, n_epoch, i, len(loader), get_lr(optimizer),
                        loss.item(), meter_loss.avg)

                    print()
                    print(log)

                    log_fout.write(log + '\n')
                    log_fout.flush()

                if phase == 'train' and i % config['train_transporter']['ckp_per_iter'] == 0:
                    torch.save(
                        model_kp.state_dict(),
                        '%s/net_kp_epoch_%d_iter_%d.pth' % (ckp_dir, epoch, i))

                # compute some images and draw them
                if global_iteration % config['train_transporter']['image_per_iter'] == 0:
                    with torch.no_grad():
                        kp = model_kp.predict_keypoint(des)
                        heatmap = model_kp.keypoint_to_heatmap(
                            kp, inv_std=config['perception']['inv_std'])

                        images = visualize_transporter_output(des=des,
                                                              des_pred=des_pred,
                                                              heatmap=heatmap,
                                                              kp=kp)

                        print("images[0].shape", images[0].shape)


                        save_img = np.concatenate(images[:4], axis=0)
                        print("save_img.dtype", save_img.dtype)
                        print("save_img.shape", save_img.shape)

                        save_file = os.path.join(images_dir,
                                                 '%s_epoch_%d_iter_%d.png' % (phase, epoch, i))
                        cv2.imwrite(save_file, save_img)


                    pass

                writer.flush() # flush SummaryWriter events to disk
                global_iteration += 1


            log = '%s [%d/%d] Loss: %.6f, Best valid: %.6f' % (
                phase, epoch, n_epoch, meter_loss.avg, best_valid_loss)
            print(log)
            log_fout.write(log + '\n')
            log_fout.flush()

            if phase == 'valid':
                scheduler.step(meter_loss.avg)
                if meter_loss.avg < best_valid_loss:
                    best_valid_loss = meter_loss.avg

                    torch.save(model_kp.state_dict(), '%s/net_best.pth' % ckp_dir)


    log_fout.close()
