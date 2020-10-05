import os
import math
import time
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from key_dynam.dynamics.data import denormalize, normalize
from key_dynam.dynamics.utils import load_data, count_trainable_parameters


class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=2, drop_prob=0.2):

        super(GRUNet, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.output_dim = output_dim

        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, h=None):
        # x: B x T x nf
        # h: n_layers x B x nf
        B, T, nf = x.size()

        if h is None:
            h = self.init_hidden(B)

        out, h = self.gru(x, h)

        # out: B x T x nf
        # h: n_layers x B x nf
        out = self.fc(self.relu(out.contiguous().view(B * T, self.hidden_dim)))
        out = out.view(B, T, self.output_dim)

        # out: B x output_dim
        return out[:, -1]

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda()
        return hidden


class PropNet(nn.Module):

    def __init__(self, node_dim_in, edge_dim_in, nf_hidden, node_dim_out, edge_dim_out,
                 edge_type_num=1, pstep=2, batch_norm=1, use_gpu=True):

        super(PropNet, self).__init__()

        self.node_dim_in = node_dim_in
        self.edge_dim_in = edge_dim_in
        self.nf_hidden = nf_hidden

        self.node_dim_out = node_dim_out
        self.edge_dim_out = edge_dim_out

        self.edge_type_num = edge_type_num
        self.pstep = pstep

        # node encoder
        modules = [
            nn.Linear(node_dim_in, nf_hidden),
            nn.ReLU()]
        if batch_norm == 1:
            modules.append(nn.BatchNorm1d(nf_hidden))
        self.node_encoder = nn.Sequential(*modules)

        # edge encoder
        self.edge_encoders = nn.ModuleList()
        for i in range(edge_type_num):
            modules = [
                nn.Linear(node_dim_in * 2 + edge_dim_in, nf_hidden),
                nn.ReLU()]
            if batch_norm == 1:
                modules.append(nn.BatchNorm1d(nf_hidden))

            self.edge_encoders.append(nn.Sequential(*modules))

        # node propagator
        modules = [
            # input: node_enc, node_rep, edge_agg
            nn.Linear(nf_hidden * 3, nf_hidden),
            nn.ReLU(),
            nn.Linear(nf_hidden, nf_hidden),
            nn.ReLU()]
        if batch_norm == 1:
            modules.append(nn.BatchNorm1d(nf_hidden))
        self.node_propagator = nn.Sequential(*modules)

        # edge propagator
        self.edge_propagators = nn.ModuleList()
        for i in range(pstep):
            edge_propagator = nn.ModuleList()
            for j in range(edge_type_num):
                modules = [
                    # input: node_rep * 2, edge_enc, edge_rep
                    nn.Linear(nf_hidden * 3, nf_hidden),
                    nn.ReLU(),
                    nn.Linear(nf_hidden, nf_hidden),
                    nn.ReLU()]
                if batch_norm == 1:
                    modules.append(nn.BatchNorm1d(nf_hidden))
                edge_propagator.append(nn.Sequential(*modules))

            self.edge_propagators.append(edge_propagator)

        # node predictor
        modules = [
            nn.Linear(nf_hidden * 2, nf_hidden),
            nn.ReLU()]
        if batch_norm == 1:
            modules.append(nn.BatchNorm1d(nf_hidden))
        modules.append(nn.Linear(nf_hidden, node_dim_out))
        self.node_predictor = nn.Sequential(*modules)

        # edge predictor
        modules = [
            nn.Linear(nf_hidden * 2, nf_hidden),
            nn.ReLU()]
        if batch_norm == 1:
            modules.append(nn.BatchNorm1d(nf_hidden))
        modules.append(nn.Linear(nf_hidden, edge_dim_out))
        self.edge_predictor = nn.Sequential(*modules)

    def forward(self, node_rep, edge_rep=None, edge_type=None, start_idx=0,
                ignore_node=False, ignore_edge=False):
        # node_rep: B x N x node_dim_in
        # edge_rep: B x N x N x edge_dim_in
        # edge_type: B x N x N x edge_type_num
        # start_idx: whether to ignore the first edge type
        B, N, _ = node_rep.size()

        # node_enc
        node_enc = self.node_encoder(node_rep.view(-1, self.node_dim_in)).view(B, N, self.nf_hidden)

        # edge_enc
        node_rep_r = node_rep[:, :, None, :].repeat(1, 1, N, 1)
        node_rep_s = node_rep[:, None, :, :].repeat(1, N, 1, 1)
        if edge_rep is not None:
            tmp = torch.cat([node_rep_r, node_rep_s, edge_rep], 3)
        else:
            tmp = torch.cat([node_rep_r, node_rep_s], 3)

        edge_encs = []
        for i in range(start_idx, self.edge_type_num):
            edge_enc = self.edge_encoders[i](tmp.view(B * N * N, -1)).view(B, N, N, 1, self.nf_hidden)
            edge_encs.append(edge_enc)
        # edge_enc: B x N x N x edge_type_num x nf
        edge_enc = torch.cat(edge_encs, 3)

        if edge_type is not None:
            edge_enc = edge_enc * edge_type.view(B, N, N, self.edge_type_num, 1)[:, :, :, start_idx:]

        # edge_enc: B x N x N x nf
        edge_enc = edge_enc.sum(3)

        for i in range(self.pstep):
            if i == 0:
                node_effect = node_enc
                edge_effect = edge_enc

            # calculate edge_effect
            node_effect_r = node_effect[:, :, None, :].repeat(1, 1, N, 1)
            node_effect_s = node_effect[:, None, :, :].repeat(1, N, 1, 1)
            tmp = torch.cat([node_effect_r, node_effect_s, edge_effect], 3)

            edge_effects = []
            for j in range(start_idx, self.edge_type_num):
                edge_effect = self.edge_propagators[i][j](tmp.view(B * N * N, -1))
                edge_effect = edge_effect.view(B, N, N, 1, self.nf_hidden)
                edge_effects.append(edge_effect)
            # edge_effect: B x N x N x edge_type_num x nf
            edge_effect = torch.cat(edge_effects, 3)

            if edge_type is not None:
                edge_effect = edge_effect * edge_type.view(B, N, N, self.edge_type_num, 1)[:, :, :, start_idx:]

            # edge_effect: B x N x N x nf
            edge_effect = edge_effect.sum(3)

            # calculate node_effect
            edge_effect_agg = edge_effect.sum(2)
            tmp = torch.cat([node_enc, node_effect, edge_effect_agg], 2)
            node_effect = self.node_propagator(tmp.view(B * N, -1)).view(B, N, self.nf_hidden)

        node_effect = torch.cat([node_effect, node_enc], 2).view(B * N, -1)
        edge_effect = torch.cat([edge_effect, edge_enc], 3).view(B * N * N, -1)

        # node_pred: B x N x node_dim_out
        # edge_pred: B x N x N x edge_dim_out
        if ignore_node:
            edge_pred = self.edge_predictor(edge_effect)
            return edge_pred.view(B, N, N, -1)
        if ignore_edge:
            node_pred = self.node_predictor(node_effect)
            return node_pred.view(B, N, -1)

        node_pred = self.node_predictor(node_effect).view(B, N, -1)
        edge_pred = self.edge_predictor(edge_effect).view(B, N, N, -1)
        return node_pred, edge_pred


class DynaNetGNN(nn.Module):

    def __init__(self, args, use_gpu=True):
        super(DynaNetGNN, self).__init__()

        self.args = args
        nf = args.nf_hidden * 4

        self.ratio = (args.height // 64) * (args.width // 64)

        # dynamics modeling
        self.model_dynam_encode = PropNet(
            node_dim_in=args.node_attr_dim + 2,
            edge_dim_in=args.edge_attr_dim + 4,
            nf_hidden=nf * 3,
            node_dim_out=nf,
            edge_dim_out=nf,
            edge_type_num=args.edge_type_num,
            pstep=1,
            batch_norm=1)

        self.model_dynam_node_forward = GRUNet(
            nf + 2 + args.node_attr_dim + args.action_dim, nf * 2, nf)
        self.model_dynam_edge_forward = GRUNet(
            nf + 4 + args.edge_attr_dim + args.action_dim * 2, nf * 2, nf)

        self.model_dynam_decode = PropNet(
            node_dim_in=nf + args.node_attr_dim + args.action_dim + 2,
            edge_dim_in=nf + args.edge_attr_dim + args.action_dim * 2 + 4,
            nf_hidden=nf * 3,
            node_dim_out=2,
            edge_dim_out=0,
            edge_type_num=args.edge_type_num,
            pstep=2,
            batch_norm=0)

        print('model_dynam_encode #params', count_trainable_parameters(self.model_dynam_encode))
        print('model_dynam_node_forward #params', count_trainable_parameters(self.model_dynam_node_forward))
        print('model_dynam_edge_forward #params', count_trainable_parameters(self.model_dynam_edge_forward))
        print('model_dynam_decode #params', count_trainable_parameters(self.model_dynam_decode))

        # for generating gaussian heatmap
        lim = args.lim
        x = np.linspace(lim[0], lim[1], args.width // 4)
        y = np.linspace(lim[2], lim[3], args.height // 4)

        if use_gpu:
            self.x = Variable(torch.FloatTensor(x)).cuda()
            self.y = Variable(torch.FloatTensor(y)).cuda()
        else:
            self.x = Variable(torch.FloatTensor(x))
            self.y = Variable(torch.FloatTensor(y))

        self.graph = [None, None, None]

    def dynam_prediction(self, kp, graph, action=None, eps=5e-2, env=None):
        # kp: B x n_his x n_kp x (2 + 4)
        # action:
        #   BallAct: B x n_his x n_kp x action_dim
        args = self.args
        nf = args.nf_hidden * 4
        action_dim = args.action_dim
        node_attr_dim = args.node_attr_dim
        edge_attr_dim = args.edge_attr_dim
        edge_type_num = args.edge_type_num

        B, n_his, n_kp, _ = kp.size()

        # node_attr: B x n_kp x node_attr_dim
        # edge_attr: B x n_kp x n_kp x edge_attr_dim
        # edge_type: B x n_kp x n_kp x edge_type_num
        node_attr, edge_type, edge_attr = graph

        # node_enc: B x n_his x n_kp x nf
        # edge_enc: B x n_his x (n_kp * n_kp) x nf
        node_enc = torch.cat([kp, node_attr.view(B, 1, n_kp, node_attr_dim).repeat(1, n_his, 1, 1)], 3)
        edge_enc = torch.cat([
            torch.cat([kp[:, :, :, None, :].repeat(1, 1, 1, n_kp, 1),
                       kp[:, :, None, :, :].repeat(1, 1, n_kp, 1, 1)], 4),
            edge_attr.view(B, 1, n_kp, n_kp, edge_attr_dim).repeat(1, n_his, 1, 1, 1)], 4)

        node_enc, edge_enc = self.model_dynam_encode(
            node_enc.view(B * n_his, n_kp, node_attr_dim + 2),
            edge_enc.view(B * n_his, n_kp, n_kp, edge_attr_dim + 4),
            edge_type[:, None, :, :, :].repeat(1, n_his, 1, 1, 1).view(B * n_his, n_kp, n_kp, edge_type_num),
            start_idx=0)

        node_enc = node_enc.view(B, n_his, n_kp, nf)
        edge_enc = edge_enc.view(B, n_his, n_kp * n_kp, nf)

        # node_enc: B x n_kp x n_his x nf
        # edge_enc: B x (n_kp * n_kp) x n_his x nf
        node_enc = node_enc.transpose(1, 2).contiguous().view(B, n_kp, n_his, nf)
        edge_enc = edge_enc.transpose(1, 2).contiguous().view(B, n_kp * n_kp, n_his, nf)

        # node_enc: B x n_kp x n_his x (nf + node_attr_dim + action_dim)
        # kp_node: B x n_kp x n_his x 2
        kp_node = kp.transpose(1, 2).contiguous().view(B, n_kp, n_his, 2)

        node_enc = torch.cat([
            kp_node, node_enc, node_attr.view(B, n_kp, 1, node_attr_dim).repeat(1, 1, n_his, 1)], 3)

        # edge_enc: B x (n_kp * n_kp) x n_his x (nf + edge_attr_dim + action_dim)
        # kp_edge: B x (n_kp * n_kp) x n_his x (2 + 2)
        kp_edge = torch.cat([
            kp_node[:, :, None, :, :].repeat(1, 1, n_kp, 1, 1),
            kp_node[:, None, :, :, :].repeat(1, n_kp, 1, 1, 1)], 4)
        kp_edge = kp_edge.view(B, n_kp**2, n_his, 4)

        edge_enc = torch.cat([
            kp_edge, edge_enc, edge_attr.view(B, n_kp**2, 1, edge_attr_dim).repeat(1, 1, n_his, 1)], 3)

        # append action
        if action is not None:
            if env in ['BallAct']:
                action_t = action.transpose(1, 2).contiguous()
                action_t_r = action_t[:, :, None, :, :].repeat(1, 1, n_kp, 1, 1).view(B, n_kp**2, n_his, action_dim)
                action_t_s = action_t[:, None, :, :, :].repeat(1, n_kp, 1, 1, 1).view(B, n_kp**2, n_his, action_dim)
                # print('node_enc', node_enc.size(), 'edge_enc', edge_enc.size())
                # print('action_t', action_t.size(), 'action_t_r', action_t_r.size(), 'action_t_s', action_t_s.size())
                node_enc = torch.cat([node_enc, action_t], 3)
                edge_enc = torch.cat([edge_enc, action_t_r, action_t_s], 3)

        # node_enc: B x n_kp x nf
        # edge_enc: B x n_kp x n_kp x nf
        node_enc = self.model_dynam_node_forward(node_enc.view(B * n_kp, n_his, -1)).view(B, n_kp, nf)
        edge_enc = self.model_dynam_edge_forward(edge_enc.view(B * n_kp**2, n_his, -1)).view(B, n_kp, n_kp, nf)

        # kp_pred: B x n_kp x (2 + 3)
        node_enc = torch.cat([node_enc, node_attr, kp_node[:, :, -1]], 2)
        edge_enc = torch.cat([edge_enc, edge_attr, kp_edge[:, :, -1].view(B, n_kp, n_kp, 4)], 3)

        if action is not None:
            if env in ['BallAct']:
                # print('node_enc', node_enc.size(), 'edge_enc', edge_enc.size(), 'action', action.size())
                action_r = action[:, :, :, None, :].repeat(1, 1, 1, n_kp, 1)
                action_s = action[:, :, None, :, :].repeat(1, 1, n_kp, 1, 1)
                node_enc = torch.cat([node_enc, action[:, -1]], 2)
                edge_enc = torch.cat([edge_enc, action_r[:, -1], action_s[:, -1]], 3)

        kp_pred = self.model_dynam_decode(
            node_enc, edge_enc, edge_type,
            start_idx=0, ignore_edge=True)

        # kp_pred: B x n_kp x 2
        kp_pred = kp[:, -1, :, :2] + kp_pred[:, :, :2]

        return kp_pred

