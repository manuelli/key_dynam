import random
import numpy as np
import logging
import time
from tinydb import TinyDB
from tinydb.storages import MemoryStorage

# torch
import torch
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from key_dynam.utils.torch_utils import make_default_image_to_tensor_transform


class ImageDataset(data.Dataset):
    """
    Simple Dataset class that just serves up individual images from a set
    of episodes. This simple class is useful for doing things like precomputing
    descriptor images or keypoints . . . 
    """

    def __init__(self,
                 config,
                 episodes,
                 phase="train",
                 camera_names=None, #(optional) list[str]: list of cameras to use, defaults to all
                 ):

        assert phase in ["train", "valid", "all"]

        self._config = config
        self._episodes = episodes
        self._phase = phase
        self._camera_names = camera_names
        self.verbose = False
        self.debug = False
        self.initialize()

    def initialize(self):
        """
        Initialize
        - setup train/valid splits
        - setup rgb --> tensor transform
        :return:
        :rtype:
        """
        # setup train/valid splits
        self.set_test_train_splits()

        # rgb --> tensor transform
        self.rgb_to_tensor_transform = make_default_image_to_tensor_transform()

    def set_test_train_splits(self):
        """
        Divides episodes into test/train splits
        :return:
        :rtype:
        """
        episode_names = list(self._episodes.keys())
        episode_names.sort()  # to make sure train/test splits are deterministic
        self._all_episode_names = episode_names
        self._all_index = self.make_index(self._all_episode_names)

        if self._phase in ["train", "valid"]:
            n_train = int(len(episode_names) * self._config['train']['train_valid_ratio'])
            n_valid = len(episode_names) - n_train

            self._train_episode_names = episode_names[0:n_train]
            self._valid_episode_names = episode_names[n_train:-1]

            self._train_index = self.make_index(self._train_episode_names)
            self._valid_index = self.make_index(self._valid_episode_names)

    @property
    def index(self):
        if self._phase == "train":
            return self._train_index
        elif self._phase == "valid":
            return self._valid_index
        elif self._phase == "all":
            return self._all_index

    @property
    def episode_names(self):
        if self._phase == "train":
            return self._train_episode_names
        elif self._phase == "valid":
            return self._valid_episode_names
        elif self._phase == "all":
            return self._all_episode_names
        else:
            raise ValueError("unknown phase:", self._phase)

    @staticmethod
    def make_index_for_episode(episode,
                               episode_name,
                               camera_names=None):
        if camera_names is None:
            camera_names = episode.camera_names

        index = []
        for idx in range(episode.length):
            for camera_name in camera_names:
                entry = {'episode_name': episode_name,
                         'idx': idx,
                         'camera_name': camera_name,
                         }

                index.append(entry)

        return index

    @property
    def episodes(self):
        return self._episodes

    def make_index(self,
                   episode_names):
        index = []

        for name in episode_names:
            episode = self._episodes[name]
            episode_index = ImageDataset.make_index_for_episode(episode,
                                                                name,
                                                                camera_names=self._camera_names)
            index.extend(episode_index)

        return index

    def __len__(self):
        return len(self.index)

    def _getitem(self,
                 episode,  # EpisodeReader
                 idx,  # int
                 camera_name,  # str
                 ):
        """
        Internal version of __getitem__. Returns dictionary with image data
        :param episode:
        :type episode:
        :param idx:
        :type idx:
        :param camera_name:
        :type camera_name:
        :return:
        :rtype:
        """
        data = episode.get_image_data(camera_name=camera_name, idx=idx)
        data['rgb_tensor'] = self.rgb_to_tensor_transform(data['rgb'])
        data['episode_name'] = episode.name
        data['idx'] = idx
        data['key_tree'] = episode.get_image_key_tree(camera_name, idx)
        data['key_tree_joined'] = "/".join(data['key_tree'])

        return data

    def __getitem__(self, item_idx):
        """
        For use by a torch DataLoader. Finds entry in index, calls the internal _getitem method
        :param item_idx:
        :type item_idx:
        :return:
        :rtype:
        """

        entry = self.index[item_idx]
        episode = self._episodes[entry['episode_name']]
        data = self._getitem(episode, entry['idx'], entry['camera_name'])

        return data
