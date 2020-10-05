from future.utils import iteritems
import os
import h5py
import abc


from key_dynam.utils.utils import load_yaml, load_pickle

class EpisodeReader(abc.ABC):

    def __init__(self):
        pass

    @property
    def config(self):
        raise NotImplementedError()

    @property
    def name(self):
        raise NotImplementedError()

    @property
    def length(self):
        """
        The length of the trajectory
        :return:
        :rtype:
        """
        raise NotImplementedError

    @property
    def image_episode(self):
        """
        If you want to be able to get image observations from this type of dataset
        you should pass, upon construction of the episode, an episode reader object
        that subclasses dense_correspondence.dataset.episode_reader. This is then
        used in EpisodeDataset to load images, descriptor keypoints, etc.
        :return:
        :rtype:
        """
        raise NotImplementedError

    def image_episode_idx_from_query_idx(self, idx):
        """
        The idx for the image episode data might be different from idx you
        pass in due to downsampling
        :param idx:
        :type idx:
        :return:
        :rtype:
        """
        raise NotImplementedError

    def get_observation(self, idx):
        raise NotImplementedError()

    def get_action(self, idx):
        raise NotImplementedError()

    def get_data(self, idx):
        """
        Get all data for that timestep . . .
        :param idx:
        :type idx:
        :return:
        :rtype:
        """
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError("subclass must implement this method")

    def make_index(self,
                   episode_name=None,
                   config=None,
                   ):
        """
        Makes the index, can be overwritten by base class if necessary
        :return:
        :rtype:
        """
        assert config is not None, "you must specify a config"

        if episode_name is None:
            episode_name = self.name

        low = config['train']['n_history'] - 1
        high = len(self) - (config['train']['n_rollout'] + 1)
        high -= 1 # to allow for action lookups that involve next timestep

        index = []
        for idx in range(low, high+1):
            d = {'episode_name': episode_name,
                'idx': idx,
                }
            index.append(d)

        return index


class PyMunkEpisodeReader(EpisodeReader):

    def __init__(self, data):
        super(PyMunkEpisodeReader, self).__init__()
        self._data = data

    @property
    def data(self):
        return self._data

    @property
    def name(self):
        return self.data['name']

    def get_data(self, idx):
        return self._data["trajectory"][idx]

    def get_observation(self, idx):
        return self._data["trajectory"][idx]["observation"]

    def get_action(self, idx):
        return self._data["trajectory"][idx]["action"]

    def __len__(self):
        return len(self._data["trajectory"])


    @staticmethod
    def load_pymunk_episodes_from_raw_data(raw_data):
        multi_episode_dict = dict()
        for episode_name, episode_data in iteritems(raw_data):
            episode_reader = PyMunkEpisodeReader(episode_data)
            multi_episode_dict[episode_name] = episode_reader


        return multi_episode_dict
