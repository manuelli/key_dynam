import argparse
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--env', default='')
parser.add_argument('--time_step', type=int, default=0)
parser.add_argument('--dt', type=float, default=1./50.)
parser.add_argument('--nf_hidden', type=int, default=32)
parser.add_argument('--norm_layer', default='Batch', help='Batch|Instance')
parser.add_argument('--n_kp', type=int, default=0, help="number of objects")

parser.add_argument('--outf', default='train')
parser.add_argument('--dataf', default='data')

'''
train
'''
parser.add_argument('--random_seed', type=int, default=1024)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--n_epoch', type=int, default=1000)
parser.add_argument('--beta1', type=float, default=0.9)

parser.add_argument('--num_workers', type=int, default=10)
parser.add_argument('--gen_data', type=int, default=0, help="whether to generate new data")
parser.add_argument('--train_valid_ratio', type=float, default=0.95, help="percentage of training data")

parser.add_argument('--log_per_iter', type=int, default=100, help="print log every x iterations")
parser.add_argument('--ckp_per_iter', type=int, default=5000, help="save checkpoint every x iterations")

parser.add_argument('--dy_epoch', type=int, default=-1)
parser.add_argument('--dy_iter', type=int, default=-1)

parser.add_argument('--height_raw', type=int, default=0)
parser.add_argument('--width_raw', type=int, default=0)
parser.add_argument('--height', type=int, default=0)
parser.add_argument('--width', type=int, default=0)
parser.add_argument('--scale_size', type=int, default=0)
parser.add_argument('--crop_size', type=int, default=0)

parser.add_argument('--eval', type=int, default=0)

# for dynamics prediction
parser.add_argument('--n_his', type=int, default=5, help='number of frames used as input')
parser.add_argument('--n_roll', type=int, default=5, help='number of rollout steps for training')
parser.add_argument('--gauss_std', type=float, default=5e-2)

parser.add_argument('--pstep', type=int, default=2)
parser.add_argument('--node_attr_dim', type=int, default=0)
parser.add_argument('--edge_attr_dim', type=int, default=0)
parser.add_argument('--edge_type_num', type=int, default=0)

'''
eval
'''
parser.add_argument('--evalf', default='eval')
parser.add_argument('--eval_dy_epoch', type=int, default=-1)
parser.add_argument('--eval_dy_iter', type=int, default=-1)

parser.add_argument('--eval_set', default='valid', help='train|valid')
parser.add_argument('--eval_st_idx', type=int, default=0)
parser.add_argument('--eval_ed_idx', type=int, default=0)

parser.add_argument('--vis_edge', type=int, default=1)
parser.add_argument('--store_demo', type=int, default=1)
parser.add_argument('--store_result', type=int, default=0)
parser.add_argument('--store_st_idx', type=int, default=0)
parser.add_argument('--store_ed_idx', type=int, default=0)


'''
model
'''
# object attributes:
parser.add_argument('--attr_dim', type=int, default=0)

# object state:
parser.add_argument('--state_dim', type=int, default=0)

# action:
parser.add_argument('--action_dim', type=int, default=0)

# relation:
parser.add_argument('--relation_dim', type=int, default=0)



def gen_args():
    args = parser.parse_args()

    if args.env == 'BallAct':
        args.data_names = ['attrs', 'states', 'actions', 'rels']

        args.n_rollout = 2000
        args.time_step = 300
        args.train_valid_ratio = 0.9

        # radius
        args.attr_dim = 1
        # x, y, xdot, ydot
        args.state_dim = 4
        # ddx, ddy
        args.action_dim = 2
        # none, spring, rod
        args.relation_dim = 4

        args.height_raw = 110
        args.width_raw = 110
        args.height = 64
        args.width = 64
        args.scale_size = 64
        args.crop_size = 64

        args.lim = [-1., 1., -1., 1.]

    else:
        raise AssertionError("Unsupported env %s" % args.env)


    # path to data
    args.dataf = 'data/' + args.dataf + '_' + args.env + '_nkp_' + str(args.n_kp)

    # path to train
    dump_prefix = 'dump_{}/'.format(args.env)

    args.outf_dy = dump_prefix + args.outf
    args.outf_dy += '_' + args.env + '_dy'
    args.outf_dy += '_nkp_' + str(args.n_kp)
    args.outf_dy += '_nHis_' + str(args.n_his)

    # path to eval
    args.evalf = dump_prefix + args.evalf
    args.evalf += '_' + args.env

    args.evalf += '_' + str(args.eval_set)

    args.evalf += '_nkp_' + str(args.n_kp)
    args.evalf += '_nHis_' + str(args.n_his)

    if args.eval_dy_epoch > -1:
        args.evalf += '_dyEpoch_' + str(args.eval_dy_epoch)
        args.evalf += '_dyIter_' + str(args.eval_dy_iter)
    else:
        args.evalf += '_dyEpoch_best'

    return args

