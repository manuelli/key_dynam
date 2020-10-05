import os
import time
import random
import numpy as np
import cv2

from progressbar import ProgressBar

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


from key_dynam.utils.utils import get_project_root, save_yaml, load_yaml, get_current_YYYY_MM_DD_hh_mm_ss_ms, get_data_root
from key_dynam.transporter.models_kp import Transporter
from key_dynam.transporter.utils import rand_int, count_parameters, AverageMeter, get_lr, to_np
from key_dynam.dynamics.utils import set_seed
from key_dynam.transporter.dataset import ImageTupleDataset
from key_dynam.dense_correspondence.dc_drake_sim_episode_reader import DCDrakeSimEpisodeReader



def eval_transporter(config,
                     train_dir,
                     multi_episode_dict=None,
                     model_checkpoint_file=None, # model_checkpoint_file
                     eval_dir=None,
                     ckp_dir=None,
                     ):


    if ckp_dir is None:
        ckp_dir = os.path.join(train_dir, 'train_nKp%d_invStd%.1f' % (
            config['perception']['n_kp'], config['perception']['inv_std']))

    if eval_dir is None:
        eval_dir = os.path.join(train_dir, 'eval_nKp%d_invStd%.1f' % (
            config['perception']['n_kp'], config['perception']['inv_std']))



    ## load the model
    use_gpu = torch.cuda.is_available()

    ### model
    model_kp = Transporter(config, use_gpu=use_gpu)
    print("model_kp #params: %d" % count_parameters(model_kp))




    if model_checkpoint_file is None:
        if config['eval']['eval_epoch'] >= 0:
            model_checkpoint_file = os.path.join(
                ckp_dir, 'net_kp_epoch_%d_iter_%d.pth' % (
                    config['eval']['eval_epoch'],
                    config['eval']['eval_iter']
                ))
        else:
            model_checkpoint_file = os.path.join(
                ckp_dir, 'net_best.pth')

    print("Loading saved ckp from %s" % model_checkpoint_file)
    model_kp.load_state_dict(torch.load(model_checkpoint_file))

    # criterion
    criterionMSE = nn.MSELoss()

    if use_gpu:
        model_kp = model_kp.cuda()

    # set the model to eval mode
    model_kp = model_kp.eval()

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
            datasets[phase], batch_size=config['train']['batch_size'],
            shuffle=True if phase == 'train' else False,
            num_workers=config['train']['batch_size'])



    meter_loss = AverageMeter()

    phase = config['eval']['eval_set']
    loader = dataloaders[phase]


    bar = ProgressBar(max_value=len(loader))
    out = None # placeholder that will be assigned later

    num_episodes = config['eval']['num_episodes']
    for i, data in bar(enumerate(loader)):

        if i >= num_episodes:
            break

        with torch.set_grad_enabled(False):
            src = data[0]['rgb_crop_tensor']
            des = data[1]['rgb_crop_tensor']

            if use_gpu:
                src = src.cuda()
                des = des.cuda()

            des_pred = model_kp(src, des)
            kp = model_kp.predict_keypoint(des)
            heatmap = model_kp.keypoint_to_heatmap(
                kp, inv_std=config['perception']['inv_std'])



        # reconstruction loss
        loss = criterionMSE(des_pred, des)
        meter_loss.update(loss.item(), src.size(0))

        log = '[%d/%d] Loss: %.6f (%.6f)' % (i, num_episodes, loss.item(), meter_loss.avg)
        print()
        print(log)

        ###### generate video
        width_demo = config['eval']['width_demo']
        height_demo = config['eval']['height_demo']
        n_split, d_split = 5, 4

        c = [(255, 105, 65), (0, 69, 255), (50, 205, 50), (0, 165, 255), (238, 130, 238),
             (128, 128, 128), (30, 105, 210), (147, 20, 255), (205, 90, 106), (255, 215, 0)]

        # predicted des
        des_pred = to_np(torch.clamp(des_pred, -1., 1.)).transpose(0, 2, 3, 1)[..., ::-1]
        des_pred = (des_pred * 0.5 + 0.5) * 255.

        # the real des
        des = to_np(torch.clamp(des, -1., 1.)).transpose(0, 2, 3, 1)[..., ::-1]
        des = (des * 0.5 + 0.5) * 255.

        # predicted keypoints
        kp = to_np(kp) + 1.
        kp = kp * height_demo / 2.
        kp = np.round(kp).astype(np.int)

        # corresponding heatmap
        heatmap = to_np(heatmap).transpose(0, 2, 3, 1)
        heatmap = np.sum(heatmap, 3, keepdims=True)
        heatmap = np.clip(heatmap * 255., 0., 255.)

        if config['eval']['video']:
            video_path = os.path.join(eval_dir, '%d.avi' % i)
            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
            print('Save video as %s' % video_path)
            out = cv2.VideoWriter(video_path, fourcc, 10, (
                width_demo * n_split + d_split * (n_split - 1), height_demo))

        if config['eval']['image']:
            image_path = os.path.join(eval_dir, '%d' % i)
            os.system('mkdir -p ' + image_path)
            print('Save image to %s' % image_path)

        for j in range(des_pred.shape[0]):
            des_pred_cur = des_pred[j]
            des_pred_cur = cv2.resize(des_pred_cur, (width_demo, height_demo))

            des_cur = des[j]
            des_cur = cv2.resize(des_cur, (width_demo, height_demo))

            kp_cur = kp[j]

            heatmap_cur = heatmap[j]
            heatmap_cur = cv2.resize(heatmap_cur, (width_demo, height_demo),
                                     interpolation=cv2.INTER_NEAREST)

            # visualization
            kp_map = np.zeros((des_cur.shape[0], des_cur.shape[1], 3))

            for k in range(kp_cur.shape[0]):
                cv2.circle(kp_map, (kp_cur[k, 0], kp_cur[k, 1]), 6, c[k], -1)

            overlay_gt = des_cur * 0.5 + kp_map * 0.5

            h, w = des_cur.shape[:2]
            merge = np.zeros((h, w * n_split + d_split * (n_split - 1), 3))

            merge[:, :w] = des_cur
            merge[:, (w + d_split) * 1 : (w + d_split) * 1 + w] = kp_map
            merge[:, (w + d_split) * 2 : (w + d_split) * 2 + w] = overlay_gt
            merge[:, (w + d_split) * 3 : (w + d_split) * 3 + w] = heatmap_cur[..., None]
            merge[:, (w + d_split) * 4:] = des_pred_cur
            merge = merge.astype(np.uint8)

            if config['eval']['video']:
                out.write(merge)

            if config['eval']['image']:
                cv2.imwrite(os.path.join(image_path, '%d.png' % j), merge)


    log = 'Loss: %.6f' % meter_loss.avg
    print(log)

    if config['eval']['video']:
        out.release()

