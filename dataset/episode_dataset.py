from __future__ import print_function

import random
import numpy as np
import transforms3d
import functools

# torch
import torch
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
import torch.utils.data.dataloader
from torch.utils.data.dataloader import default_collate

# key_dynam
from key_dynam.utils.torch_utils import make_default_image_to_tensor_transform
from key_dynam.utils import transform_utils
from key_dynam.utils.utils import random_sample_in_range

DEBUG = False


class MultiEpisodeDataset(data.Dataset):

    def __init__(self,
                 config,  # type: dict
                 action_function=None,  # type: function
                 observation_function=None,  # type: function
                 episodes=None,  # dict, keys of type EpisodeReader
                 phase="train",
                 sample_T_aug=None,  # function: (optional) otherwise it's constructed from config
                 visual_observation_function=None,  # type function, extracts visual information episode, returns a dict
                 ):

        self._action_function = action_function
        self._observation_function = observation_function
        self._visual_observation_function = visual_observation_function
        self._config = config

        # dict of episodes
        # key: episode name
        # value: EpisodeReader
        assert episodes is not None
        self._episodes = episodes
        self._length = None
        self._phase = phase
        self._data_augmentation_enabled = (phase == "train")
        self.set_train_test_splits()
        self._rgb_image_to_tensor = make_default_image_to_tensor_transform()

        # construct the sampler for data augmentation if specified in config
        self._sample_T_aug = sample_T_aug
        if sample_T_aug is None and 'data_augmentation' in self.config['dataset'] and \
                self.config['dataset']['data_augmentation']['enabled']:
            self._sample_T_aug = MultiEpisodeDataset.construct_T_aug_sampler(config)

    @property
    def n_history(self):
        return self._config["train"]["n_history"]

    @property
    def n_rollout(self):
        return self._config["train"]["n_rollout"]

    @property
    def episode_dict(self):
        return self._episodes

    @property
    def observation_function(self):
        return self._observation_function

    @property
    def data_augmentation_enabled(self):
        return self._data_augmentation_enabled

    @data_augmentation_enabled.setter
    def data_augmentation_enabled(self, val):
        self._data_augmentation_enabled = val

    @property
    def index(self):
        if self._phase == "train":
            return self._train_index
        else:
            return self._valid_index

    @property
    def config(self):
        return self._config

    def set_train_test_splits(self):
        """
        Divides episodes into train/test splits
        :return:
        :rtype:
        """
        episode_names = list(self._episodes)
        episode_names.sort()  # to make sure train/test splits are deterministic
        n_train = int(len(episode_names) * self._config['train']['train_valid_ratio'])
        n_valid = len(episode_names) - n_train

        self._train_episode_names = episode_names[0:n_train]
        self._valid_episode_names = episode_names[n_train:]

        self._train_index = self.make_index(self._train_episode_names)
        self._valid_index = self.make_index(self._valid_episode_names)

    def get_episode_names(self):
        """
        Returns a list of episode names
        :return:
        :rtype:
        """
        if self._phase == "train":
            return self._train_episode_names
        elif self._phase == "valid":
            return self._valid_episode_names

    def compute_length(self):
        """
        Computes length of dataset
        :return:
        :rtype:
        """
        length = 0
        for episode_name in self.get_episode_names():
            episode = self._episodes[episode_name]
            length += len(episode)

        return length

    def __len__(self):
        return len(self.index)

    def make_index(self,
                   episode_names):
        index = []

        for name in episode_names:
            episode = self._episodes[name]
            index.extend(episode.make_index(episode_name=name, config=self.config))

        return index

    def get_random_episode(self):
        """
        returns a random episode from test or train depending on the value of
        self._phase.

        This is not used during regular usage. Instead we let the DataLoader
        randomly sample from the index.
        :return:
        :rtype:
        """
        # returns a random episode from test or train depending on the value of
        # self._phase
        if self._phase == "train":
            episode_name = random.sample(self._train_episode_names, 1)[0]
        elif self._phase == "valid":
            episode_name = random.sample(self._valid_episode_names, 1)[0]

        episode = self._episodes[episode_name]
        return episode

    def sample_index_from_episode(self,
                                  episode,  # EpisodeReader object
                                  rollout_length,  # steps to rollout
                                  ):
        """
        Sample random index from the episode. This idx must be such that
        there are n_history indices before and rollout_length indices after.

        This is not used during regular usage. Instead we let the DataLoader
        randomly sample from the index.
        :param episode:
        :type episode:
        :return:
        :rtype:
        """
        n_history = self.n_history
        low, high = self.get_valid_idx_range_for_episode(episode, n_history, rollout_length)
        idx = random.randint(low, high)
        return idx

    def get_valid_idx_range_for_episode(self,
                                        episode,  # EpisodeReader object
                                        n_history,
                                        rollout_length
                                        ):
        """
        Returns range of indices for which there are at least n_history indices before (inclusive)
        and n_roll indices after.
        :param episode:
        :type episode:
        :return:
        :rtype:
        """
        low = n_history - 1
        high = len(episode) - (rollout_length + 1)
        return (low, high)

    def _getitem(self,
                 episode,
                 idx,
                 rollout_length,
                 n_history=None,  #
                 visual_observation=None,  # bool whether or not to get visual observation specified in config
                 T_aug=None,
                 ):

        if n_history is None:
            n_history = self.n_history

        if visual_observation is None:
            if 'visual_observation' in self._config['dataset']:
                visual_observation = self._config['dataset']['visual_observation']['enabled']
            else:
                visual_observation = False

        visual_observation_config = None
        if visual_observation:
            visual_observation_config = self._config['dataset']['visual_observation']

        # check episode_idx is in valid range
        low, high = self.get_valid_idx_range_for_episode(episode, n_history, rollout_length)
        assert (low <= idx <= high), "Index %d is out of range [%d, %d]" % (idx, low, high)

        obs_list = [] # stores list of observation tensors
        action_list = [] # stores list of action tensors
        visual_obs_tensor_list = []  # tensors that come from images
        visual_obs_func_list = [] # stores output that comes from visual_observation_func
        visual_obs_list = [] # old version of this


        # if this is not None then T_aug type data augmentation is enabled

        if (T_aug is None) and \
                ('data_augmentation' in self.config['dataset']) and \
                (self.config['dataset']['data_augmentation']['enabled']) and self.data_augmentation_enabled:

            T_aug = self._sample_T_aug()

        if not self.data_augmentation_enabled:
            T_aug = None # don't do T_aug if we have this flag set

        # parse the observations
        start_idx = idx - (n_history - 1)
        end_idx = idx + rollout_length
        idx_range = range(start_idx, end_idx + 1)
        for j in idx_range:
            obs_raw = episode.get_observation(j)
            data_raw = episode.get_data(j)
            obs = self._observation_function(obs_raw,
                                             data_augmentation=self.data_augmentation_enabled,
                                             data_raw=data_raw,
                                             T=T_aug,
                                             episode=episode,
                                             episode_idx=j,
                                             )
            obs_list.append(obs)

            action_raw = episode.get_action(j)
            action = self._action_function(action_raw,
                                           data_raw=data_raw,
                                           T=T_aug,
                                           data_augmentation=self.data_augmentation_enabled,
                                           episode=episode,
                                           episode_idx=j,
                                           )
            action_list.append(action)

            if self._visual_observation_function is not None:
                visual_obs_tensor_dict = self._visual_observation_function(episode,
                                                                           episode_idx=j,
                                                                           T=T_aug,
                                                                           data_augmentation=self.data_augmentation_enabled,
                                                                           )

                # print("visual_obs_tensor_dict.keys()", visual_obs_tensor_dict.keys())
                # print("visual_obs_tensor_dict['tensor'].shape", visual_obs_tensor_dict['tensor'].shape)
                visual_obs_tensor_list.append(visual_obs_tensor_dict['tensor'])
                visual_obs_func_list.append(visual_obs_tensor_dict)

            # handle the visual observation
            if visual_observation:
                raise ValueError("deprecated")
                # note you can only use T_aug for the descriptor_keypoints_3d_world_frame
                # image type. The other ones can't be adjusted.

                image_data = dict()
                image_episode = episode.image_episode

                for camera_name in visual_observation_config['camera_names']:
                    image_data_local = image_episode.get_image_data_specified_in_config(camera_name, j,
                                                                                        visual_observation_config)

                    # add noise to keypoints if requested
                    config_data_aug = self.config['dataset']['visual_observation']['data_augmentation']

                    if self.data_augmentation_enabled and config_data_aug['keypoints']['augment']:
                        descriptor_keypoints_key = "descriptor_keypoints_3d_world_frame"
                        if descriptor_keypoints_key in image_data_local:
                            keypoints = image_data_local[descriptor_keypoints_key]
                            keypoints_aug = MultiEpisodeDataset.add_noise_to_tensor(keypoints,
                                                                                    config_data_aug['keypoints'][
                                                                                        'std_dev'])
                            image_data_local[descriptor_keypoints_key] = keypoints_aug

                    # only support T_aug for descriptor_keypoints_3d_world_frame
                    if T_aug is not None:
                        for key in image_data_local:
                            if key not in ["descriptor_keypoints_3d_world_frame", "T_W_C", "camera_name", "idx", 'K']:
                                raise ValueError("T_aug data augmentation not supported for visual observations yet")
                            if key == "descriptor_keypoints_3d_world_frame":
                                keypoints_3d_world_frame = image_data_local[key]

                                # transform them
                                keypoints_3d_aug = transform_utils.transform_points_3D(T_aug,
                                                                                       keypoints_3d_world_frame)

                                image_data_local[key] = keypoints_3d_aug

                    # apply image normalization if we have loaded an rgb image
                    if 'rgb' in image_data_local:
                        rgb_copy = np.copy(image_data_local['rgb'])
                        image_data_local['rgb_tensor'] = self._rgb_image_to_tensor(rgb_copy)

                    image_data[camera_name] = image_data_local

                visual_obs_list.append(image_data)

            if DEBUG:
                print("obs.shape", obs.shape)
                print("action.shape", action.shape)

        # use default_collate to append first dimension and concatenate them
        # e.g. observations [M, obs_dim]
        # actions: [M, action_dim]
        visual_observation_collated = []
        if len(visual_obs_list) > 0:
            visual_observation_collated = default_collate(visual_obs_list)

        # [n_sample, obs_dim]
        obs_collated = default_collate(obs_list)

        visual_obs_tensor_collated = []
        visual_obs_func_collated = []
        if len(visual_obs_func_list) > 0:
            visual_obs_tensor_collated = default_collate(visual_obs_tensor_list)
            visual_obs_func_collated = default_collate(visual_obs_func_list)

        obs_combined_collated = None
        if len(visual_obs_tensor_list) > 0:
            # [n_sample, vis_obs_tensor_dim]
            obs_combined_collated = torch.cat((visual_obs_tensor_collated, obs_collated), dim=-1)
        else:
            obs_combined_collated = obs_collated

        return {"observations": default_collate(obs_list),
                "actions": default_collate(action_list),
                'visual_observation_tensor': visual_obs_tensor_collated,
                'visual_observation_func_collated': visual_obs_func_collated,
                'visual_obs_func_list': visual_obs_func_list,
                "visual_observations": visual_observation_collated,
                'visual_observations_list': visual_obs_list,
                'observations_combined': obs_combined_collated,
                'idx_range': list(idx_range),
                'episode_name': episode.name}

    def __getitem__(self, item_idx, episode_name=None, episode_idx=None, rollout_length=None):
        """

        :param idx:
        :type idx:
        :param episode_name: override to use specific episode
        :type episode_name:
        :param episode_idx: override to use specific index from that episode
        :type episode_idx:
        :return:
        :rtype:
        """

        entry = self.index[item_idx]
        episode_name = entry['episode_name']
        episode = self.episode_dict[episode_name]
        episode_idx = entry['idx']

        return self._getitem(episode, idx=episode_idx, rollout_length=self.n_rollout)

    @staticmethod
    def add_noise_to_tensor(x,
                            std_dev):

        noise = np.random.normal(scale=std_dev,
                                 size=x.shape)

        return x + noise

    def compute_dataset_statistics(self, num_samples=500):
        """
        Computes mean and std_dev of the dataset.
        :param num_samples:
        :type num_samples:
        :return: dict containing mean and std_dev for both observations and actions
        :rtype:
        """
        # computes the mean and std_dev of the data

        observation_list = []
        action_list = []

        for i in range(num_samples):
            j = random.randint(0, self.__len__())
            data = self[i]

            observation_list.append(data['observations'])
            action_list.append(data['actions'])

        observations = torch.cat(observation_list)
        actions = torch.cat(action_list)

        if DEBUG:
            print("observations.shape", observations.shape)
            print("actions.shape", actions.shape)

        stats = dict()
        stats['observations'] = dict()
        stats['observations']['mean'] = torch.mean(observations, 0)
        stats['observations']['std'] = torch.std(observations, 0)

        stats['actions'] = dict()
        stats['actions']['mean'] = torch.mean(actions, 0)
        stats['actions']['std'] = torch.std(actions, 0)

        return stats

    @staticmethod
    def construct_T_aug_sampler(config):  # return: func
        """
        Returns a function that samples homogeneous transforms.

        These transforms are really in the plane, just x-y translation
        and yaw
        :param config:
        :type config:
        :return:
        :rtype:
        """
        pos_min = np.array(config['dataset']['data_augmentation']['pos_min'], dtype=np.float32)
        pos_max = np.array(config['dataset']['data_augmentation']['pos_max'], dtype=np.float32)

        yaw_min = np.array([config['dataset']['data_augmentation']['yaw_min']], dtype=np.float32)
        yaw_max = np.array([config['dataset']['data_augmentation']['yaw_max']], dtype=np.float32)

        func = functools.partial(MultiEpisodeDataset.sample_T_aug, pos_min, pos_max, yaw_min, yaw_max)

        return func

    @staticmethod
    def sample_T_aug(pos_min,
                     pos_max,
                     yaw_min,
                     yaw_max):

        low = np.array(pos_min, dtype=np.float32)
        high = np.array(pos_max, dtype=np.float32)
        pos = random_sample_in_range(low, high)

        yaw_min = np.array(yaw_min, dtype=np.float32)
        yaw_max = np.array(yaw_max, dtype=np.float32)
        yaw = random_sample_in_range(yaw_min, yaw_max)

        T = np.eye(4)
        T[:3, :3] = transforms3d.euler.euler2mat(0, 0, yaw)
        T[:3, 3] = pos
        return T
