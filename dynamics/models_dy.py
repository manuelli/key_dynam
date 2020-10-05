import os
import math
import time
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from key_dynam.dynamics.data_normalizer import DataNormalizer


def rollout_model(state_init,  # [B, n_his, obs_dim]
                  action_seq,  # [B, n_his + n_rollout - 1, action_dim]
                  dynamics_net,
                  compute_debug_data=False,
                  ):
    """
    Rolls out the dynamics model for the given number of steps
    """

    assert len(state_init.shape) == 3
    assert len(action_seq.shape) == 3

    # cast it to the same device as action_seq
    state_init = state_init.to(action_seq.device)

    B, n_his, obs_dim = state_init.shape
    _, n_tmp, action_dim = action_seq.shape

    # if state_init and action_seq have same size in dim=1
    # then we are just doing 1 step prediction
    n_rollout = n_tmp - n_his + 1
    assert n_rollout > 0, "n_rollout = %d must be greater than 0" % (n_rollout)

    state_cur = state_init
    state_pred_list = []
    debug_data_list = []
    for i in range(n_rollout):

        # [B, n_his, action_dim]
        actions_cur = action_seq[:, i:i+n_his]
        # state_cur is [B, n_his, obs_dim]

        dyna_net_input = {'observation': state_cur,
                          'action': actions_cur}

        # [B, obs_dim]
        obs_pred = dynamics_net.forward(dyna_net_input)

        if compute_debug_data:
            debug_data = {'model_input': dyna_net_input,
                          'model_out': obs_pred}
            debug_data_list.append(debug_data)

        state_cur = torch.cat([state_cur[:, 1:], obs_pred.unsqueeze(1)], 1)
        state_pred_list.append(obs_pred)

    # [B, n_rollout, obs_dim]
    state_pred_tensor = torch.stack(state_pred_list, axis=1)

    return {'state_pred': state_pred_tensor,
            'debug_data': debug_data_list}


def rollout_action_sequences(state_init,  # [n_his, obs_dim]
                             action_seq,  # [B, n_his + n_rollout -1, action_dim]
                             model_dy,
                             ):
    """
    Rollout many different action sequences from a single initial state
    This function is almost identical to 'rollout_model' but expands
    state_init to have the appropriate batch size. Does torch.no_grad() if so desired
    """
    # need to expand states_cur to have batch_size dimension
    # currently has shape [n_his, state_dim]
    n_his, state_dim = state_init.shape

    # trim it to have the correct shape
    # [n_sample, n_his + (n_look_ahead - 1), action_dim]]
    n_sample = action_seq.shape[0]

    device = action_seq.device

    # expand to have shape [n_sample, n_his, state_dim]
    state_init = state_init.to(device).unsqueeze(0).expand([n_sample, -1, -1])

    out = rollout_model(state_init,
                        action_seq,
                        model_dy,
                        compute_debug_data=True,  # for now
                        )

    # [n_sample, n_look_ahead, state_dim]
    state_pred = out['state_pred']

    return {'state_pred': state_pred,
            'model_rollout': out}


def get_object_and_robot_state_indices(config):

    object_state_size = 1
    for x in config['dataset']['object_state_shape']:
        object_state_size *= x

    object_indices = list(range(0, object_state_size))

    robot_state_size = 1
    for x in config['dataset']['robot_state_shape']:
        robot_state_size *= x

    robot_indices = list(range(len(object_indices), len(object_indices) + robot_state_size))

    return {'object_indices': object_indices,
            'robot_indices': robot_indices}


class DynaNetMLP(nn.Module):

    def __init__(self, config):

        super(DynaNetMLP, self).__init__()

        self.config = config
        nf = config['train']['nf_hidden']

        self.state_dim = config['dataset']['state_dim']
        self.action_dim = config['dataset']['action_dim']
        self.n_his = config['train']['n_history']

        # create default DataNormalizers
        self.action_normalizer = DataNormalizer(torch.zeros(self.action_dim), torch.ones(self.action_dim))
        self.state_normalizer = DataNormalizer(torch.zeros(self.state_dim), torch.zeros(self.state_dim))

        # get sizes of things
        self._index_dict = get_object_and_robot_state_indices(config)

        # self.register_buffer('_object_indices', torch.LongTensor(self._index_dict['object_indices']))
        # self.register_buffer('_robot_indices', torch.LongTensor(self._index_dict['robot_indices']))

        self._object_indices = torch.LongTensor(self._index_dict['object_indices']).cuda()
        self._robot_indices = torch.LongTensor(self._index_dict['robot_indices']).cuda()

        self._object_state_shape = tuple(config['dataset']['object_state_shape'])

        input_dim = (self.state_dim + self.action_dim) * self.n_his

        layers = [
            nn.Linear(input_dim, nf),
            nn.ReLU(),
            nn.Linear(nf, nf * 2),
            nn.ReLU(),
            nn.Linear(nf * 2, nf * 2),
            nn.ReLU(),
            nn.Linear(nf * 2, nf),
            nn.ReLU(),
            nn.Linear(nf, self.state_dim),]

        # use batch norm by default
        if "batch_norm" not in config['train']:
            config['train']['batch_norm'] = True

        # add batchnorm layer if it was specified
        if config['train']['batch_norm']:
            layers.insert(0, nn.BatchNorm1d(input_dim))

        self.model = nn.Sequential(*layers)

    def forward(self,
                input,  # dict: w/ keys ['observation', 'action']
                ):  # torch.FloatTensor B x state_dim

        """
        input['observation'].shape is [B, n_his, obs_dim]
        input['action'].shape is [B, n_his, action_dim]
        :param input:
        :type input:
        :return:
        :rtype:
        """

        # [B, n_his, obs_size]
        state = input['observation']
        action = input['action']

        B, n_his, state_size = state.shape

        # flatten the observation and action inputs
        # then concatenate them
        input = torch.cat([state.view(B, -1), action.view(B, -1)], 1)
        output = self.model(input)

        # output: B x state_dim
        # always predict the residual
        output = output + state[:, -1]

        return output

    def compute_z_state(self,
                        x):
        """
        Computes the z state. This is used in some of the other models (e.g. DynaNetMLPWeightMatrix)
        :param x:
        :type x:
        :return:
        :rtype:
        """

        # cast self._object_indices to the correct device
        self._object_indices = self._object_indices.to(x.device)

        x_object_flat = x.index_select(-1, self._object_indices)

        shape = x_object_flat.shape[:-1] + self._object_state_shape
        x_object = x_object_flat.reshape(shape)

        return {'z': x,
                'z_object': x_object,
                'z_object_flat': x_object_flat}



class DynaNetMLPWeighted(DynaNetMLP):

    def __init__(self, config):
        super(DynaNetMLPWeighted, self).__init__(config)
        raise ValueError("DEPRECATED")

        # add the weights
        self._K = self.config['dataset']['object_state_shape'][0]
        self._alpha = torch.nn.Parameter(torch.ones(self._K)) # they all start out equally weighted

        if "freeze_keypoint_weights" in self.config['dynamics_net'] and self.config['dynamics_net']['freeze_keypoint_weights']:
            self._alpha.requires_grad = False

    @property
    def weights(self):
        return torch.nn.functional.softmax(self._alpha)



class DynaNetMLPWeightMatrix(DynaNetMLP):

    def __init__(self, config):
        super(DynaNetMLPWeightMatrix, self).__init__(config)

        # add the weights
        self._K = self.config['dataset']['object_state_shape'][0]
        self._M = self.config['dataset']['object_state_shape'][0] # same for now


        self._index_dict = get_object_and_robot_state_indices(config)
        self._object_indices = torch.LongTensor(self._index_dict['object_indices']).cuda()
        self._robot_indices = torch.LongTensor(self._index_dict['robot_indices']).cuda()
        self._object_state_shape = tuple(config['dataset']['object_state_shape'])


        # add a simple check on size of relative tensors

        assert self.config['dataset']['state_dim'] == len(self._object_indices) + len(self._robot_indices)

        assert self._M == self._K # this is the only thing we support for now


        # internal weights
        self._alpha = torch.nn.Parameter(torch.zeros([self._M, self._K]))

        # initialize the weights, set all off diagonal elements
        self._alpha.data = math.log(1.0/self._K) * torch.ones([self._M, self._K])
        for i in range(self._K):
            self._alpha.data[i, i] = 2.0


        if 'weight_matrix_init' in self.config['dynamics_net']:
            init_type = self.config['dynamics_net']['weight_matrix_init']
            if init_type == "uniform":
                torch.nn.init.normal_(self._alpha)


        if "freeze_keypoint_weights" in self.config['dynamics_net'] and self.config['dynamics_net']['freeze_keypoint_weights']:
            self._alpha.requires_grad = False

    @property
    def weight_matrix(self): # weight matrix W with shape [M, K], rows sum to one
        return torch.nn.functional.softmax(self._alpha, dim=1)


    def compute_z_state(self,
                        x,  # [B, state_dim] or [B, N, state_dim]
                        ): # torch.Tensor with shape [B, z_dim] or [B, N, z_dim]


        # do a size check

        state_dim = x.shape[-1]
        num_indices = self._robot_indices.numel() + self._object_indices.numel()



        assert state_dim == num_indices, "size mismatch len(x) = %d, (len(self._robot_indices) + len(self._object_indices)) = %d" %(state_dim, num_indices)


        device = x.device

        # cast it to the right device if it isn't already
        # x = x.to(self._object_indices.device)

        x_robot_flat = x.index_select(-1, self._robot_indices.to(device))
        x_object_flat = x.index_select(-1, self._object_indices.to(device))

        shape = x_object_flat.shape[:-1] + self._object_state_shape

        # [B, K, 3] or [B, N, K, 3]
        x_object = x_object_flat.reshape(shape)

        # https://pytorch.org/docs/stable/torch.html?highlight=mm#torch.matmul
        # [B, M, 3] or [B, N, M, 3]
        z_object = torch.matmul(self.weight_matrix.to(device), x_object)


        # flatten the last two dimension
        z_object_flat = torch.flatten(z_object, start_dim=-2, end_dim=-1)
        z = torch.cat((z_object_flat, x_robot_flat), dim=-1)



        return {'z': z,
                'z_object': z_object,
                'z_object_flat': z_object_flat,}




class VisualNet(nn.Module):
    """
    Visual dynamics net
    """

    def __init__(self):
        super(VisualNet, self).__init__()

    @property
    def output_size(self):
        raise NotImplementedError("sublass must implement this")

    def forward(self):  # return: dict
        """

        :return: dict
        should have at least one key which is 'dynamics_net_input'
        This should be tensor with shape [B, N_his, output_size]
        :rtype: dict:

        out should have (at least) key 'dynamics_net_input'
        of shape [B, N_his, output_size]
        """
        raise NotImplementedError("subclass must implement this")


class VisualDynamicsNet(nn.Module):

    def __init__(self,
                 config,  # dict: can contain arbitrary information
                 vision_net,  # nn.Module
                 dynamics_net,  # nn.Module
                 ):
        super(VisualDynamicsNet, self).__init__()
        self._config = config
        self._vision_net = vision_net
        self._dynamics_net = dynamics_net

    @property
    def vision_net(self):
        return self._vision_net

    @property
    def dynamics_net(self):
        return self._dynamics_net

    def forward(self,
                input,  # dict: data
                ):  # dict:

        raise ValueError("DEPRECATED")

        """

        :param input: dict w/ keys
        - 'observation'
        - 'action'
        - 'visual_observation'

        visual_observation is a dict
        Has entries like visual_observation[<camera_name>]['rgb']
        Each entry has shape [B, N_his, 3, H, W]

        Note currently this is very inefficient in that in calls `forward' repeatedly
        on the vision model for all entries in N_his. We could offer an option to provide
        previously computed output of the vision model. That way one only needs to run the vision
        model on the new visual observation [B, -1, 3, H, W] instead of all N_his images

        :type input:
        :return:
        :rtype:
        """

        vision_net_out = self._vision_net.forward(input['visual_observation'])
        observation = torch.cat((vision_net_out['dynamics_net_input'], input['observation']))

        dyna_net_input = {'observation': observation,
                          'action': input['action']}

        dyna_net_out = self._dynamics_net.forward(dyna_net_input)

        out = {'vision_net_out': vision_net_out,
               'dynamics_net_out:': dyna_net_out}

        return out
