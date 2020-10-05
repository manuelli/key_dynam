import os
import random
import itertools
import matplotlib.pyplot as plt

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from progressbar import ProgressBar
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from key_dynam.dynamics.config import gen_args
from key_dynam.dynamics.data import PhysicsDataset, load_data, store_data, resize_and_crop, pil_loader
from key_dynam.dynamics.models_dy import DynaNetGNN
from key_dynam.dynamics.utils import count_trainable_parameters, Tee, AverageMeter, to_np, to_var, norm, set_seed

from key_dynam.dynamics.data import normalize, denormalize

args = gen_args()

use_gpu = torch.cuda.is_available()


'''
model
'''
model_dy = DynaNetGNN(args, use_gpu=use_gpu)

# print model #params
print("model #params: %d" % count_trainable_parameters(model_dy))

if args.eval_dy_epoch == -1:
    model_dy_path = os.path.join(args.outf_dy, 'net_best_dy.pth')
else:
    model_dy_path = os.path.join(
        args.outf_dy, 'net_dy_epoch_%d_iter_%d.pth' % \
        (args.eval_dy_epoch, args.eval_dy_iter))

print("Loading saved ckp from %s" % model_dy_path)
model_dy.load_state_dict(torch.load(model_dy_path))
model_dy.eval()

if use_gpu:
    model_dy.cuda()

criterionMSE = nn.MSELoss()


'''
data
'''
data_dir = os.path.join(args.dataf, args.eval_set)

stat_path = os.path.join(args.dataf, 'stat.h5')
if args.env in ['BallAct']:
    data_names = ['attrs', 'states', 'actions', 'rels']
    stat = load_data(data_names[:2], stat_path)

loader = pil_loader

trans_to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


'''
store results
'''
os.system('mkdir -p ' + args.evalf)

log_path = os.path.join(args.evalf, 'log.txt')
tee = Tee(log_path, 'w')


def evaluate(roll_idx, video=True, image=True):

    eval_path = os.path.join(args.evalf, str(roll_idx))

    n_split = 3
    split = 4

    if image:
        os.system('mkdir -p ' + eval_path)
        print('Save images to %s' % eval_path)

    if video:
        video_path = eval_path + '.avi'
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        print('Save video as %s' % video_path)
        out = cv2.VideoWriter(video_path, fourcc, 25, (
            args.width_raw * n_split + split * (n_split - 1), args.height_raw))

    if args.env in ['BallAct']:
        metadata_path = os.path.join(data_dir, str(roll_idx) + '.h5')
        metadata = load_data(data_names, metadata_path)

        states = metadata[data_names.index('states')]
        states = normalize([states], [stat[data_names.index('states')]])[0]

        actions = metadata[data_names.index('actions')] / 600.
        actions = torch.FloatTensor(actions).cuda()

    '''
    model prediction
    '''
    if args.env in ['BallAct']:
        # graph_gt
        edge_type = metadata[data_names.index('rels')][0, :, 0].astype(np.int)
        edge_attr = metadata[data_names.index('rels')][0, :, 1:]
        edge_type_gt = np.zeros((args.n_kp, args.n_kp, args.edge_type_num))
        edge_attr_gt = np.zeros((args.n_kp, args.n_kp, edge_attr.shape[1]))
        cnt = 0
        # print(edge_type)
        # print(edge_attr)
        for x in range(args.n_kp):
            for y in range(x):
                edge_type_gt[x, y, edge_type[cnt]] = 1.
                edge_type_gt[y, x, edge_type[cnt]] = 1.
                edge_attr_gt[x, y] = edge_attr[cnt]
                edge_attr_gt[y, x] = edge_attr[cnt]
                cnt += 1

        edge_type_gt = torch.FloatTensor(edge_type_gt).cuda()
        edge_attr_gt = torch.FloatTensor(edge_attr_gt).cuda()

        node_attr_gt = torch.zeros((args.n_kp, args.node_attr_dim)).cuda()

        graph_gt = node_attr_gt.unsqueeze(0), edge_type_gt.unsqueeze(0), edge_attr_gt.unsqueeze(0)

        # kps
        kps = metadata[1][args.eval_st_idx:args.eval_ed_idx, :, :2] / 80.
        kps[:, :, 1] *= -1
        kps = torch.FloatTensor(kps).cuda()
        kps_gt = kps

        n_kp = args.n_kp

    # the current keypoints
    eps = 5e-2
    kp_cur = kps_gt[:args.n_his].view(1, args.n_his, args.n_kp, 2)

    loss_kp_acc = 0.
    for i in range(args.eval_ed_idx - args.eval_st_idx):

        if i >= args.n_his:
            with torch.set_grad_enabled(False):
                # predict the feat and hmap at the next time step
                action_cur = actions[i - args.n_his : i].unsqueeze(0) if actions is not None else None
                kp_pred = model_dy.dynam_prediction(kp_cur, graph_gt, action_cur, env=args.env)
                mean_pred = kp_pred[:, :, :2]

            # compare with the ground truth
            kp_des = kps_gt[i : i + 1]

            loss_kp = criterionMSE(mean_pred, kp_des)

            n_roll = args.eval_ed_idx - args.eval_st_idx - args.n_his
            loss_kp_acc += loss_kp.item() / n_roll

            if i == args.n_his or i % 10 == 0:
                print("step %d, kp: %.6f" % (i, loss_kp.item()))

            # update feat_cur and hmap_cur
            kp_cur = torch.cat([kp_cur[:, 1:], kp_pred.unsqueeze(1)], 1)

            # record the prediction
            keypoint = mean_pred

        else:
            keypoint = kps_gt[i:i+1]

        # transform the numpy
        lim = args.lim
        keypoint = to_np(keypoint)[0] - [lim[0], lim[2]]
        keypoint *= args.height_raw / 2.
        keypoint = np.round(keypoint).astype(np.int)

        # generate the visualization
        img_path = os.path.join(data_dir, str(roll_idx), 'fig_%d.png' % (i + args.eval_st_idx))
        img = cv2.imread(img_path).astype(np.float)
        kp_map = np.zeros((img.shape[0], img.shape[1], 3))

        c = [(255, 105, 65), (0, 69, 255), (50, 205, 50), (0, 165, 255), (238, 130, 238),
             (128, 128, 128), (210, 105, 30), (255, 20, 147), (106, 90, 205), (255, 215, 0)]
        for j in range(keypoint.shape[0]):
            cv2.circle(kp_map, (keypoint[j, 0], keypoint[j, 1]), 4, c[j], -1)

        overlay_pred = img * 0.5 + kp_map * 0.5

        merge = np.ones((img.shape[0], img.shape[1] * n_split + split * (n_split - 1), 3)) * 255.
        merge[:, :img.shape[1]] = img
        merge[:, img.shape[1] + 4 : img.shape[1] * 2 + 4] = kp_map
        merge[:, img.shape[1] * 2 + 8 : img.shape[1] * 3 + 8] = overlay_pred

        merge = merge.astype(np.uint8)

        if image:
            cv2.imwrite(os.path.join(eval_path, 'fig_%d.png' % i), merge)

        if video:
            out.write(merge)

    if video:
        out.release()

    print("kp: %.6f" % (loss_kp_acc))


bar = ProgressBar()
ls_rollout_idx = np.arange(args.store_st_idx, args.store_ed_idx)

for roll_idx in bar(ls_rollout_idx):
    print()
    print("Eval # %d / %d" % (roll_idx, ls_rollout_idx[-1]))
    evaluate(roll_idx, video=True, image=True)

