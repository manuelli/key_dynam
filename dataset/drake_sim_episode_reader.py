from future.utils import iteritems
import os
import h5py
import itertools
import copy
import numpy as np
from tinydb import TinyDB
from tinydb.storages import MemoryStorage

from key_dynam.dataset.episode_reader import EpisodeReader
from key_dynam.utils.utils import load_yaml, load_pickle
from key_dynam.utils import transform_utils, drake_image_utils



class DrakeSimEpisodeReader(EpisodeReader):

    def __init__(self,
                 non_image_data, # dict
                 episode_name="",
                 dc_episode_reader=None, # DCDrakeSimEpisodeReader, needed if you want to load images
                 ):
        super(DrakeSimEpisodeReader, self).__init__()
        self._non_image_data = non_image_data
        self._dc_episode_reader = dc_episode_reader


        self._has_descriptor_keypoints = False
        if self._dc_episode_reader is not None:
            self._has_descriptor_keypoints = self._dc_episode_reader.has_descriptor_keypoints

        self._episode_name = episode_name


    @property
    def name(self):
        return self._non_image_data['name']

    @property
    def trajectory(self):
        return self._non_image_data['trajectory']

    @property
    def length(self):
        return len(self.trajectory)

    @property
    def episode_name(self):
        return self._episode_name

    @property
    def image_episode(self):
        if self._dc_episode_reader is not None:
            return self._dc_episode_reader
        else:
            raise ValueError("No EpisodeReader for images was specified")

    def __len__(self):
        return self.length

    @property
    def config(self):
        return self._non_image_data['config']

    def get_observation(self, idx):
        return self._non_image_data['trajectory'][idx]["observation"]

    def get_action(self, idx):
        return self._non_image_data["trajectory"][idx]["action"]

    def get_data(self, idx):
        return self._non_image_data['trajectory'][idx]

    @staticmethod
    def metadata_file(dataset_root):
        return os.path.join(dataset_root, 'metadata.yaml')

    @staticmethod
    def config_file(dataset_root):
        return os.path.join(dataset_root, 'config.yaml')

    @staticmethod
    def load_dataset(dataset_root, # str: folder containing dataset
                     load_image_data=True,
                     descriptor_images_root=None, # str: (optional) folder containing hdf5 files with descriptors
                     descriptor_keypoints_root=None,
                     max_num_episodes=None, # int, max num episodes to load
                     precomputed_data_root=None,
                     ):
        """

        :param dataset_root: folder should contain
            - config.yaml
            - metadata.yaml
            - <episode_name.p>
            - <episode_name.h5>
        :type dataset_root:
        :return:
        :rtype:
        """

        if load_image_data:
            from key_dynam.dense_correspondence.dc_drake_sim_episode_reader import DCDrakeSimEpisodeReader

        metadata = load_yaml(DrakeSimEpisodeReader.metadata_file(dataset_root))
        multi_episode_dict = dict()
        episode_names = list(metadata['episodes'].keys())
        episode_names.sort() # sort the keys

        num_episodes = len(episode_names)

        # optionally don't read all episodes
        if (max_num_episodes is not None) and (max_num_episodes > 0):
            # compute the number of episodes to read, in sorted order
            num_episodes = int(min(len(episode_names), max_num_episodes))

        for idx in range(num_episodes):
            episode_name = episode_names[idx]
            val = metadata['episodes'][episode_name]

            # load non image data
            non_image_data_file = os.path.join(dataset_root, val['non_image_data_file'])
            assert os.path.isfile(non_image_data_file), "File doesn't exist: %s" %(non_image_data_file)
            non_image_data = load_pickle(non_image_data_file)

            dc_episode_reader = None
            if load_image_data:

                # load image data
                image_data_file = os.path.join(dataset_root, val['image_data_file'])
                assert os.path.isfile(image_data_file), "File doesn't exist: %s" % (image_data_file)

                descriptor_image_data_file = None
                if descriptor_images_root is not None:
                    descriptor_image_data_file = os.path.join(descriptor_images_root, val['image_data_file'])

                    assert os.path.isfile(descriptor_image_data_file), "File doesn't exist: %s" % (
                        descriptor_image_data_file)

                descriptor_keypoints_data = None
                descriptor_keypoints_hdf5_file = None
                if descriptor_keypoints_root is not None:

                    # replace .h5 filename with .p for pickle file
                    descriptor_keypoints_data_file = val['image_data_file'].split(".")[0] + ".p"
                    descriptor_keypoints_data_file = os.path.join(descriptor_keypoints_root,
                                                                  descriptor_keypoints_data_file)

                    descriptor_keypoints_hdf5_file = os.path.join(descriptor_keypoints_root, val['image_data_file'])

                    if os.path.isfile(descriptor_keypoints_data_file):
                        descriptor_keypoints_data = load_pickle(descriptor_keypoints_data_file)
                    else:
                        assert os.path.isfile(descriptor_keypoints_hdf5_file), "File doesn't exist: %s" % (
                            descriptor_keypoints_hdf5_file)


                ####
                precomputed_data = None
                precomputed_data_file = None
                if precomputed_data_root is not None:

                    # replace .h5 filename with .p for pickle file
                    precomputed_data_file = val['image_data_file'].split(".")[0] + ".p"
                    precomputed_data_file = os.path.join(precomputed_data_root, precomputed_data_file)

                    if os.path.isfile(precomputed_data_file):
                        precomputed_data = load_pickle(precomputed_data_file)
                    else:
                        raise ValueError("file doesn't exist: %s" % (precomputed_data_file))

                dc_episode_reader = DCDrakeSimEpisodeReader(non_image_data,
                                                             image_data_file,
                                                             descriptor_image_data_file=descriptor_image_data_file,
                                                             descriptor_keypoints_data=descriptor_keypoints_data,
                                                             descriptor_keypoints_data_file=descriptor_keypoints_hdf5_file,
                                                            precomputed_data=precomputed_data,
                                                            precomputed_data_file=precomputed_data_file)

            episode_reader = DrakeSimEpisodeReader(non_image_data=non_image_data,
                                                   episode_name=episode_name,
                                                   dc_episode_reader=dc_episode_reader)



            multi_episode_dict[episode_name] = episode_reader


        return multi_episode_dict


