import os
import copy

# dense_correspondence
import dense_correspondence.dataset.dynamic_spartan_episode_reader as dc_episode_reader

# key_dynam
from key_dynam.dataset.episode_reader import EpisodeReader
from key_dynam.utils.utils import load_yaml, load_pickle, load_json
from key_dynam.utils import transform_utils


class DynamicSpartanEpisodeReader(EpisodeReader):
    PADDED_STRING_WIDTH = 6

    """
    Dataset for static scene with a single camera
    """

    def __init__(self,
                 config,
                 root_dir,  # str: fullpath to 'processed' dir
                 name="",  # str
                 metadata=None,  # dict: optional metadata, e.g. object type etc.
                 downsample_rate=None,
                 downsample_idx=None,
                 dc_episode_reader=None, # dense_correspondence.DynamicSpartanEpisodeReader used for reading images
                 ):

        EpisodeReader.__init__(self)
        self._config = config
        self._root_dir = root_dir
        self._name = name
        self._downsample_rate = downsample_rate
        self._downsample_idx = downsample_idx
        self._dc_episode_reader = dc_episode_reader

        self._metadata = metadata

        # load camera_info
        d = load_json(self.data_file)

        # convert keys to int
        self._episode_data = {int(k): v for k,v in d.items()}
        self._indices = list(self._episode_data.keys())
        self._indices.sort()

    def query_idx_to_data_idx(self, idx):
        """
        Converts query idx to data idx
        This is what is doing the downsampling basically
        :param idx:
        :type idx:
        :return:
        :rtype:
        """
        return self._downsample_rate * idx + self._downsample_idx

    def image_episode_idx_from_query_idx(self, idx):
        return self.query_idx_to_data_idx(idx)

    @property
    def config(self):
        return self._config

    @property
    def length(self):
        # compute the length
        l = len(self._indices)//self._downsample_rate - 1
        if self.query_idx_to_data_idx(l+1) < len(self._indices):
            l = l+1

        return l

    @property
    def name(self):
        return self._name

    @property
    def camera_names(self):
        return self._dc_episode_reader.camera_names

    @property
    def image_episode(self):
        return self._dc_episode_reader

    def __len__(self):
        return self.length

    @property
    def data_file(self):
        return os.path.join(self._root_dir, 'states.json')

    def get_data(self, query_idx):
        """
        Return data for this specific timestep
        :param idx:
        :type idx:
        :return:
        :rtype:
        """
        idx = self.query_idx_to_data_idx(query_idx)
        return self._episode_data[idx]

    def get_observation(self, query_idx):
        idx = self.query_idx_to_data_idx(query_idx)
        return self._episode_data[idx]['observations']

    def get_action(self, query_idx):
        idx = self.query_idx_to_data_idx(query_idx)
        return self._episode_data[idx]['actions']

    def get_image_data(self,
                       camera_name,
                       query_idx):
        idx = self.query_idx_to_data_idx(query_idx)
        return self._dc_episode_reader.get_image_data(camera_name, idx)

    def get_image_data_specified_in_config(self,
                                           camera_name,
                                           query_idx,
                                           image_config,  # dict
                                           ):
        """
        Pass through to Dense Correspondence Episode Reader
        :param camera_name:
        :type camera_name:
        :param query_idx:
        :type query_idx:
        :param image_config:
        :type image_config:
        :return:
        :rtype:
        """

        idx = self.query_idx_to_data_idx(query_idx)
        return self._dc_episode_reader.get_image_data_specified_in_config(camera_name, idx, image_config)

    @staticmethod
    def load_dataset_old(config, # e.g. experiments/07/config.yaml
                     episodes_config,  # dict, e.g. real_push_box.yaml
                     episodes_root,  # str: root of where all the logs are stored
                     precomputed_data_root=None,
                     max_num_episodes=None,):

        dc_config = None

        multi_episode_dict = dict()
        for counter, episode_name in enumerate(episodes_config['episodes']):

            # this is for debugging purposes
            if (max_num_episodes is not None) and counter >= max_num_episodes:
                break

            episode_processed_dir = os.path.join(episodes_root, episode_name, "processed")

            descriptor_keypoints_file = None
            if descriptor_keypoints_root is not None:
                descriptor_keypoints_file = os.path.join(descriptor_keypoints_root, "%s.h5" %(episode_name))

            if precomputed_data_root is not None:
                descriptor_keypoints_file = os.path.join(precomputed_data_root, "%s.h5" %(episode_name))

            dc_episode = dc_episode_reader.DynamicSpartanEpisodeReader(dc_config,
                                                                       episode_processed_dir,
                                                                       name=episode_name,
                                                                       descriptor_keypoints_file=descriptor_keypoints_file
                                                                       )

            episode = DynamicSpartanEpisodeReader(config,
                                                  episode_processed_dir,
                                                  name=episode_name,
                                                  downsample_rate=config['dataset']['downsample_rate'],
                                                  downsample_idx=0, # hardcoded for now
                                                  dc_episode_reader=dc_episode,
                                                  )

            multi_episode_dict[episode_name] = episode

        return multi_episode_dict

    @staticmethod
    def load_dataset(config,  # e.g. experiments/07/config.yaml
                     episodes_config,  # dict, e.g. real_push_box.yaml
                     episodes_root,  # str: root of where all the logs are stored
                     precomputed_data_root=None,
                     max_num_episodes=None,
                     load_image_episode=True):

        multi_episode_dict = dict()
        for counter, episode_name in enumerate(episodes_config['episodes']):

            # this is for debugging purposes
            if (max_num_episodes is not None) and counter >= max_num_episodes:
                break

            episode_processed_dir = os.path.join(episodes_root, episode_name, "processed")

            # load the DenseCorrespondence Episode that handles image observations
            # if requested
            dc_episode = None
            if load_image_episode:

                ####
                precomputed_data = None
                precomputed_data_file = None
                if precomputed_data_root is not None:

                    # replace .h5 filename with .p for pickle file
                    precomputed_data_file = os.path.join(precomputed_data_root, "%s.p" % (episode_name))
                    precomputed_data_file_hdf5 = os.path.join(precomputed_data_root, "%s.h5" % (episode_name))


                    if os.path.isfile(precomputed_data_file):
                        precomputed_data = load_pickle(precomputed_data_file)
                    else:
                        raise ValueError("file doesn't exist: %s" % (precomputed_data_file))

                dc_episode = dc_episode_reader.DynamicSpartanEpisodeReader(config=None,
                                                                           root_dir=episode_processed_dir,
                                                                           name=episode_name,
                                                                           precomputed_data=precomputed_data,
                                                                           precomputed_data_file=precomputed_data_file
                                                                           )

            episode = DynamicSpartanEpisodeReader(config,
                                                  episode_processed_dir,
                                                  name=episode_name,
                                                  downsample_rate=config['dataset']['downsample_rate'],
                                                  downsample_idx=0,  # hardcoded for now
                                                  dc_episode_reader=dc_episode,
                                                  )

            multi_episode_dict[episode_name] = episode

        return multi_episode_dict

    @staticmethod
    def ee_to_world(data,  # type dict entry from states.yaml/states.json
                        ):  # type -> numpy.ndarray 4 x 4 homogeneous transform
        transform_dict = data['observations']['ee_to_world']
        transform = transform_utils.transform_from_pose_dict(transform_dict)
        return transform