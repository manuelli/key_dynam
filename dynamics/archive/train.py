import os
import random
import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from progressbar import ProgressBar
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torch.distributions.multivariate_normal import MultivariateNormal

from key_dynam.dynamics.config import gen_args
from key_dynam.dynamics.data import PhysicsDataset, load_data
from key_dynam.dynamics.models_dy import DynaNetGNN
from key_dynam.dynamics.utils import rand_int, count_trainable_parameters, Tee, AverageMeter, get_lr, to_np, set_seed

args = gen_args()
set_seed(args.random_seed)

torch.manual_seed(args.random_seed)
np.random.seed(args.random_seed)

os.system('mkdir -p ' + args.dataf)
os.system('mkdir -p ' + args.outf_dy)
tee = Tee(os.path.join(args.outf_dy, 'train.log'), 'w')

print(args)

# generate data
trans_to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

datasets = {}
dataloaders = {}
data_n_batches = {}
for phase in ['train', 'valid']:
    datasets[phase] = PhysicsDataset(args, phase=phase, trans_to_tensor=trans_to_tensor)

    if args.gen_data:
        datasets[phase].gen_data()
    else:
        datasets[phase].load_data()

    dataloaders[phase] = DataLoader(
        datasets[phase], batch_size=args.batch_size,
        shuffle=True if phase == 'train' else False,
        num_workers=args.num_workers)

    data_n_batches[phase] = len(dataloaders[phase])

args.stat = datasets['train'].stat

use_gpu = torch.cuda.is_available()


'''
define model for dynamics prediction
'''
model_dy = DynaNetGNN(args, use_gpu=use_gpu)
print("model_dy #params: %d" % count_trainable_parameters(model_dy))

if args.dy_epoch >= 0:
    # if resume from a pretrained checkpoint
    model_dy_path = os.path.join(
        args.outf_dy, 'net_dy_epoch_%d_iter_%d.pth' % (args.dy_epoch, args.dy_iter))
    print("Loading saved ckp from %s" % model_dy_path)
    model_dy.load_state_dict(torch.load(model_dy_path))


# criterion
criterionMSE = nn.MSELoss()

# optimizer
params = model_dy.parameters()
optimizer = optim.Adam(params, lr=args.lr, betas=(args.beta1, 0.999))
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.6, patience=2, verbose=True)

if use_gpu:
    model_dy = model_dy.cuda()


st_epoch = args.dy_epoch if args.dy_epoch > 0 else 0
log_fout = open(os.path.join(args.outf_dy, 'log_st_epoch_%d.txt' % st_epoch), 'w')

best_valid_loss = np.inf

for epoch in range(st_epoch, args.n_epoch):
    phases = ['train', 'valid'] if args.eval == 0 else ['valid']

    for phase in phases:
        model_dy.train(phase == 'train')

        meter_loss = AverageMeter()
        meter_loss_rmse = AverageMeter()

        bar = ProgressBar(max_value=data_n_batches[phase])
        loader = dataloaders[phase]

        for i, data in bar(enumerate(loader)):

            if use_gpu:
                if isinstance(data, list):
                    # nested transform
                    data = [[d.cuda() for d in dd] if isinstance(dd, list) else dd.cuda() for dd in data]
                else:
                    data = data.cuda()

            with torch.set_grad_enabled(phase == 'train'):
                n_his, n_kp = args.n_his, args.n_kp
                n_samples = args.n_his + args.n_roll

                if args.env in ['BallAct']:
                    kps_gt, graph_gt, actions = data
                    B = kps_gt.size(0)

                    eps = args.gauss_std
                    kp_cur = kps_gt[:, :n_his].view(B, n_his, n_kp, 2)

                    loss_kp = 0.
                    loss_mse = 0.

                    for j in range(args.n_roll):

                        kp_des = kps_gt[:, n_his + j]

                        action_cur = actions[:, j : j + n_his] if actions is not None else None

                        # kp_pred: B x n_kp x 2
                        kp_pred = model_dy.dynam_prediction(kp_cur, graph_gt, action_cur, env=args.env)
                        mean_cur = kp_pred

                        mean_des = kp_des

                        loss_mse_cur = criterionMSE(mean_cur, mean_des)
                        loss_mse += loss_mse_cur / args.n_roll

                        # update feat_cur and hmap_cur
                        kp_cur = torch.cat([kp_cur[:, 1:], kp_pred.unsqueeze(1)], 1)

                    # summarize the losses
                    loss = loss_mse

                    # update meter
                    meter_loss_rmse.update(np.sqrt(loss_mse.item()), B)
                    meter_loss.update(loss.item(), B)

            if phase == 'train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if i % args.log_per_iter == 0:
                log = '%s [%d/%d][%d/%d] LR: %.6f' % (
                    phase, epoch, args.n_epoch, i, data_n_batches[phase],
                    get_lr(optimizer))
                log += ', rmse: %.6f (%.6f)' % (
                    np.sqrt(loss_mse.item()), meter_loss_rmse.avg)

                print()
                print(log)
                log_fout.write(log + '\n')
                log_fout.flush()

            if phase == 'train' and i % args.ckp_per_iter == 0:
                torch.save(model_dy.state_dict(), '%s/net_dy_epoch_%d_iter_%d.pth' % (args.outf_dy, epoch, i))

        log = '%s [%d/%d] Loss: %.6f, Best valid: %.6f' % (
            phase, epoch, args.n_epoch, np.sqrt(meter_loss.avg), best_valid_loss)
        print(log)
        log_fout.write(log + '\n')
        log_fout.flush()

        if phase == 'valid' and not args.eval:
            scheduler.step(meter_loss.avg)
            if meter_loss.avg < best_valid_loss:
                best_valid_loss = meter_loss.avg
                torch.save(model_dy.state_dict(), '%s/net_best_dy.pth' % (args.outf_dy))


log_fout.close()
