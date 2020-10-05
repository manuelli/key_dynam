from __future__ import print_function

import random
import numpy as np

from six.moves import cPickle as pickle

import torch
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader

from key_dynam.dynamics.utils import rand_float, rand_int


class MultiEpisodeDataset(data.Dataset):

    def __init__(self, config, data_path, phase, action_function=None, observation_function=None):
        self._action_function = action_function
        self._observation_function = observation_function
        self._config = config
        self._phase = phase

        self.n_timesteps = config['dataset']['num_timesteps']
        self.n_his = config['train']['n_history']
        self.n_rol = config['train']['n_rollout']

        self.state_dim = config['dataset']['state_dim']
        self.action_dim = config['dataset']['action_dim']

        self.load_data(data_path)

    @property
    def n_history(self):
        return self._config['train']['n_history']

    @property
    def n_rollout(self):
        return self._config['train']['n_rollout']

    def load_data(self, data_path):
        d = pickle.load(open(data_path, 'rb'))

        n_train = int(len(d.keys()) * self._config['train']['train_valid_ratio'])
        n_valid = len(d.keys()) - n_train

        self._episodes = []

        if self._phase == 'train':
            self.n_episodes = n_train
            st_idx, ed_idx = 0, n_train
        elif self._phase == 'valid':
            self.n_episodes = n_valid
            st_idx, ed_idx = n_train, n_train + n_valid
        else:
            raise AssertionError("Unknown phase %s" % self._phase)

        for i in range(st_idx, ed_idx):
            episode = d[d.keys()[i]]

            states = []
            actions = []
            for j in range(len(episode['trajectory'])):
                frame = episode['trajectory'][j]

                state = np.zeros(self.state_dim)
                action = np.zeros(self.action_dim)

                state[:2] = frame['observation']['slider']['position']
                state[2] = frame['observation']['slider']['angle']

                action[:] = frame['observation']['pusher']['position']

                states.append(state)
                actions.append(action)

            states = np.stack(states)
            actions = np.stack(actions)

            self._episodes.append([states, actions])

        '''
        # calculate mean and std for state and action
        states_acc = []
        actions_acc = []
        for i in range(self.n_episode):
            states_acc.append(self._episodes[i][0])
            actions_acc.append(self._episodes[i][1])
        states_acc = np.concatenate(states_acc, 0)
        actions_acc = np.concatenate(actions_acc, 0)
        print('state stat:', np.mean(states_acc, 0), np.std(states_acc, 0))
        print('action stat:', np.mean(actions_acc, 0), np.std(actions_acc, 0))
        '''

        self.mean_state = np.array([3.21818600e+02, 2.99796867e+02, -1.15948190e-02])
        self.std_state = np.array([25.06492009, 10.80764359, 0.50278145])
        self.mean_action = np.array([308.62541692, 299.19894001])
        self.std_action = np.array([46.8802937, 39.80293088])

    def __len__(self):
        return self.n_episodes * (self.n_timesteps - self.n_his - self.n_rol + 1)

    def __getitem__(self, idx):
        config = self._config

        offset = self.n_timesteps - self.n_his - self.n_rol + 1
        idx_episode = idx // offset
        idx_timestep = idx % offset

        states, actions = self._episodes[idx_episode]

        states_tensor = states[idx_timestep:idx_timestep + self.n_his + self.n_rol]
        actions_tensor = actions[idx_timestep:idx_timestep + self.n_his + self.n_rol]

        states_tensor = (states_tensor - self.mean_state) / self.std_state
        actions_tensor = (actions_tensor - self.mean_action) / self.std_action

        states_tensor = torch.FloatTensor(states_tensor)
        actions_tensor = torch.FloatTensor(actions_tensor)

        return states_tensor, actions_tensor

