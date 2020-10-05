from __future__ import print_function
from future.utils import iteritems
# system
from six.moves import cPickle as pickle
import os
import h5py
import copy
import numpy as np

# key_dynam
from key_dynam.utils.utils import get_current_YYYY_MM_DD_hh_mm_ss_ms

class EpisodeContainer(object):
    """
    Container that holds data from a single episode
    """

    def __init__(self, name=None):

        if name is None:
            name = get_current_YYYY_MM_DD_hh_mm_ss_ms()

        self._data = dict()
        self._data['name'] = name
        self._data['config'] = None
        self._data['trajectory'] = []

    def add_obs_action(self, obs, action):
        d = {"observation": obs.copy(),
             "action": action.copy()}

        self._data['trajectory'].append(d)

    def add_data(self, d):
        """

        :param d:
        :type d: dict of data you want to store, usually has keys ["observation", "action", . . . ]
        :return:
        :rtype:
        """
        raise ValueError("Deprecated")

    def save_to_file(self, filename):
        # pickle or json
        pickle.dump(self.data, open(filename, "wb"))

    def set_name(self, name): # set the episode name
        self._data['name'] = name

    def set_config(self, config):
        self._data["config"] = config

    def set_metadata(self, val):
        self._data['metadata'] = val

    def save_images_to_hdf5(self, output_dir):
        """
        Saves images to hdf5
        Non image data stored in self._non_image_data attribute
        :param output_dir:
        :type output_dir:
        :return:
        :rtype:
        """

        hdf5_file = "%s.h5" % (self._data['name'])
        hdf5_file_fullpath = os.path.join(output_dir, hdf5_file)
        hf = h5py.File(hdf5_file_fullpath, 'w')
        name = self._data['name']


        self._non_image_data = copy.copy(self._data)
        self._non_image_data['hdf5_file'] = hdf5_file

        for idx, val in enumerate(self._non_image_data['trajectory']):
            if "images" in val['observation']:
                image_dict = val['observation']['images']
                # deal with images
                key_tree = [name, str(idx)]
                EpisodeContainer.recursive_image_save(hf, image_dict, key_tree, verbose=False)

        return hdf5_file

    def save_non_image_data_to_pickle(self, output_dir):
        """
        Saves non image data using pickle. Not you should have already
        called save_images_to_hdf5, which creates the self._non_image_data
        attribute
        :param output_dir:
        :type output_dir:
        :return:
        :rtype:
        """
        non_image_data = self.non_image_data
        filename = "%s.p" %(non_image_data["name"])
        full_filename = os.path.join(output_dir, filename)
        pickle.dump(non_image_data, open(full_filename, "wb"))
        return filename

    @staticmethod
    def recursive_image_save(hf, # h5py file object
                             data, # dict
                             key_tree, # list of str
                             verbose=False): # None

        """
        Recursively traverses `data`.
        If it encounters a np.ndarray it will save it to the h5py object
        and replace the value in `data` with the key that was used to store
        it in the h5py object.
        """

        for key, value in iteritems(data):
            new_key_tree = copy.deepcopy(key_tree)
            new_key_tree.append(key)


            if isinstance(value, np.ndarray):
                hf_key = "/".join(new_key_tree)
                if verbose:
                    print("(SAVING): Found numpy array, saving:\n", hf_key)

                hf.create_dataset(hf_key, data=value)
                data[key] = hf_key
            elif isinstance(value, dict):
                if verbose:
                    print("(RECURSING): Found dict array with key_tree:\n", key_tree)
                EpisodeContainer.recursive_image_save(hf, value, new_key_tree, verbose=verbose)

    @property
    def data(self):
        return self._data

    @property
    def non_image_data(self):
        return self._non_image_data

    @property
    def name(self):
        return self._data["name"]


class MultiEpisodeContainer(object):
    """
    Container that holds data from multiple episodes
    """

    def __init__(self):
        self._episode_data_dict = dict()

    def add_episode(self,
                    episode, # type: EpisodeContainer
                    ):
        """
        Adds an episode to the data
        :param episode:
        :type episode:
        :return:
        :rtype:
        """
        self._episode_data_dict[episode.name] = episode.data

    @property
    def data(self):
        return self._episode_data_dict


    def save_to_file(self,
                     filename, # type: str
                     ):
        # saves the data to disk using pickle
        # pickle or json
        pickle.dump(self.data, open(filename, "wb"))





