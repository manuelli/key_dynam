from future.utils import iteritems
import os
import h5py
import itertools
import copy
import numpy as np
from tinydb import TinyDB
from tinydb.storages import MemoryStorage


# key_dynam
from key_dynam.utils.utils import load_yaml, load_pickle
from key_dynam.utils import transform_utils, drake_image_utils

# pdc
from dense_correspondence.dataset.episode_reader import EpisodeReader

class DCDrakeSimEpisodeReader(EpisodeReader):

    def __init__(self,
                 non_image_data, # dict
                 image_data_file, # str: fullpath to hdf5 file
                 descriptor_image_data_file=None, # str: fullpath to hdf5 file
                 descriptor_keypoints_data_file=None, # str: fullpath to hdf5 file
                 descriptor_keypoints_data=None,
                 episode_name="",
                 precomputed_data=None,
                 precomputed_data_file_hdf5=None,
                 precomputed_data_file=None, # optional, just for reference of where that data came from
                 ):
        super(DCDrakeSimEpisodeReader, self).__init__()

        self._non_image_data = non_image_data
        self._image_data_file = image_data_file
        self._descriptor_image_data_file = descriptor_image_data_file
        self._descriptor_keypoints_data_file = descriptor_keypoints_data_file
        self._descriptor_keypoints_data = descriptor_keypoints_data
        self._has_descriptor_keypoints = (self._descriptor_keypoints_data is not None) or (self._descriptor_keypoints_data_file is not None)
        self._initialize()
        self._episode_name = episode_name


        self._precomputed_data = precomputed_data
        self._precomputed_data_file_hdf5 = precomputed_data_file_hdf5
        self._precomputed_data_file = precomputed_data_file

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
    def indices(self):
        return list(range(self.length))

    @property
    def episode_name(self):
        return self._episode_name

    @property
    def config(self):
        return self._non_image_data['config']

    @property
    def camera_names(self):
        config = self.config
        return list(config['env']['rgbd_sensors']['sensor_list'].keys())

    @property
    def image_data_file(self):
        return self._image_data_file

    @property
    def has_descriptor_keypoints(self):
        return self._has_descriptor_keypoints

    @property
    def precomputed_data_file(self):
        return self._precomputed_data_file


    def get_observation(self, idx):
        return self._non_image_data['trajectory'][idx]["observation"]

    def _initialize(self):
        """
        Compute camera poses and K matrix.
        Store them in local variables
        :return:
        :rtype:
        """

        self._image_type_map = {"rgb": 'rgb',
                                'depth_float32': 'depth_32F',
                                'depth_int16': 'depth_16U',
                                'label': 'label',
                                'mask': 'mask'}

        self._camera_pose_dict = dict()
        self._camera_matrix_dict = dict()

        for camera_name in self.camera_names:
            self._camera_pose_dict[camera_name] = self.camera_pose(camera_name)
            self._camera_matrix_dict[camera_name] = self.camera_K_matrix(camera_name)

        # make TinyDB dataset for label etc
        keys = ['label_db', 'mask_db']
        for key in keys:
            if key in self._non_image_data['metadata']:
                data_list = self._non_image_data['metadata'][key]
                db = TinyDB(storage=MemoryStorage)
                db.insert_multiple(data_list)

        self.mask_labels = []
        for entry in self.mask_db:
            self.mask_labels.append(int(entry['label']))

    @property
    def mask_db(self):
        return self._non_image_data['metadata']['mask_db']

    def camera_pose(self,
                    camera_name, # str
                    idx=None, # not needed in this dataset
                    ): # np.array [4,4] homogeneous transform T_world_camera

        # check if we already have it
        if camera_name in self._camera_pose_dict:
            return np.copy(self._camera_pose_dict[camera_name])
        else:
            pose_dict = self.config['env']['rgbd_sensors']['sensor_list'][camera_name]
            return transform_utils.transform_from_pose_dict(pose_dict)

    def camera_K_matrix(self,
                        camera_name):

        if camera_name in self._camera_matrix_dict:
            return np.copy(self._camera_matrix_dict[camera_name])
        else:
            sensor_dict = self.config['env']['rgbd_sensors']['sensor_list'][camera_name]
            width = sensor_dict['width']
            height = sensor_dict['height']
            fov_y = sensor_dict['fov_y']

            return transform_utils.camera_K_matrix_from_FOV(width, height, fov_y)

    def get_image_from_key(self,
                           key, # str dataset_name in hdf5 file
                           ):

        """
        Note: you need to reload the h5py file each time to make this compatible
        with torch DataLoader multithreading. See https://github.com/RobotLocomotion/key_dynam/issues/41
        :param key:
        :type key:
        :return:
        :rtype:
        """

        # need the np.array otherwise it is of type HDF5 dataset
        # namely h5py._hl.dataset.Dataset
        data = None # placeholder for visibility outside with statement

        with h5py.File(self._image_data_file, 'r') as h5_file:
            data = np.array(h5_file.get(key))

        return data

    def get_descriptor_image_from_key(self,
                                      key,  # str dataset_name in hdf5 file
                                      ):

        """
        Note: you need to reload the h5py file each time to make this compatible
        with torch DataLoader multithreading. See https://github.com/RobotLocomotion/key_dynam/issues/41
        :param key:
        :type key:
        :return:
        :rtype:
        """

        # need the np.array otherwise it is of type HDF5 dataset
        # namely h5py._hl.dataset.Dataset

        data = None  # placeholder for visibility outside with statement
        if self._descriptor_image_data_file is None:
            raise RuntimeError("No descriptor_image_data_file was specified")

        with h5py.File(self._descriptor_image_data_file, 'r') as h5_file:
            data = np.array(h5_file.get(key))

        return data

    def get_descriptor_keypoints_from_key(self,
                                          key, # str: key in the dict
                                          ):

        data = None
        if self._descriptor_keypoints_data is not None:
            data = self._descriptor_keypoints_data[key]
        elif self._descriptor_keypoints_data_file is not None:
            with h5py.File(self._descriptor_keypoints_data_file, 'r') as h5_file:
                data = np.array(h5_file.get(key)) # will throw a key error if not found
        else:
            raise KeyError("No descriptor_keypoints_data_file was specified")

        return data

    def get_precomputed_data_from_key(self,
                                      key):
        """
        Tries to get the data associated with this key.
        Throws KeyError if that key isn't found
        """

        data = None
        if self._precomputed_data is not None:
            data = self._precomputed_data[key]
        elif self._precomputed_data_file_hdf5 is not None:
            with h5py.File(self._descriptor_keypoints_data_file, 'r') as h5_file:
                data = np.array(h5_file.get(key))  # will throw a key error if not found
        else:
            raise KeyError("No precomputed data file was specified")

        return data

    def get_image_key(self,
                      camera_name,
                      idx,
                      type):

        return self.get_observation(idx)['images'][camera_name][type]

    def get_image_key_tree(self,
                           camera_name, # str
                           idx, # int
                           ): # list of str

        key = self.get_image_key(camera_name, idx, 'rgb')
        key_tree = key.split("/")[:-1]
        return key_tree

    def get_image(self,
                  camera_name, # str
                  idx, # int
                  type, # string ["rgb", "depth", "mask", "label", "descriptor"]
                  ):

        obs = self.get_observation(idx)['images'][camera_name]

        if type == "rgb":
            return self.get_image_from_key(obs['rgb'])
        elif type == "depth_float32":
            data = self.get_image_from_key(obs['depth_32F'])
            return drake_image_utils.remove_depth_32F_out_of_range_and_cast(data, np.float32)
        elif type == "depth_int16":
            data = self.get_image_from_key(obs['depth_16U'])
            return drake_image_utils.remove_depth_16U_out_of_range_and_cast(data, np.int16)
        elif type == "label":
            return self.get_image_from_key(obs['label']).squeeze()
        elif type == "mask":
            label = self.get_image_from_key(obs['label']).squeeze()
            mask = self.binary_mask_from_label_image(label)
            return mask
        elif type == "descriptor":
            rgb_key = self.get_image_key(camera_name, idx, 'rgb')
            key_tree = rgb_key.split("/")[:-1]
            key_tree.append("descriptor_image")
            descriptor_image_key = "/".join(key_tree)
            return self.get_descriptor_image_from_key(descriptor_image_key)
        elif type == "descriptor_keypoints":
            rgb_key = self.get_image_key(camera_name, idx, 'rgb')
            key_tree = rgb_key.split("/")[:-1]
            key_tree.append("descriptor_keypoints")
            descriptor_keypoints_key = "/".join(key_tree)
            return self.get_descriptor_keypoints_from_key(descriptor_keypoints_key)
        elif type == "descriptor_keypoints_3d_world_frame":
            rgb_key = self.get_image_key(camera_name, idx, 'rgb')
            key_tree = rgb_key.split("/")[:-1]
            key_tree.append("descriptor_keypoints_3d_world_frame")
            descriptor_keypoints_key = "/".join(key_tree)
            return self.get_descriptor_keypoints_from_key(descriptor_keypoints_key)
        elif type == "descriptor_keypoints_3d_camera_frame":
            rgb_key = self.get_image_key(camera_name, idx, 'rgb')
            key_tree = rgb_key.split("/")[:-1]
            key_tree.append("descriptor_keypoints_3d_camera_frame")
            descriptor_keypoints_key = "/".join(key_tree)
            return self.get_descriptor_keypoints_from_key(descriptor_keypoints_key)
        else:
            raise ValueError("image type not recognized: %s" %(type))


    def get_precomputed_data(self,
                             camera_name,  # str
                             idx,  # int
                             type,  # string "transporter_keypoints/pos_world_frame", etc.
                             ):
        """
        Return some precomupted data of a specified type corresponding to this
        camera_name, idx and type
        """

        rgb_key = self.get_image_key(camera_name, idx, 'rgb')
        key_tree = rgb_key.split("/")[:-1]
        key_tree.append(type)
        key = "/".join(key_tree)

        return self.get_precomputed_data_from_key(key)

    def binary_mask_from_label_image(self, label_img):

        mask_img = np.zeros_like(label_img)

        for label_val in self.mask_labels:
            mask_img[label_img == label_val] = 1

        return mask_img

    def get_image_data(self,
                       camera_name,  # str
                       idx,  # int
                       ): # dict

        rgb = self.get_image(camera_name, idx, "rgb")
        label = self.get_image(camera_name, idx, "label")
        mask = self.get_image(camera_name, idx, 'mask')
        depth_float32 = self.get_image(camera_name, idx, "depth_float32")
        depth_int16 = self.get_image(camera_name, idx, "depth_int16")

        # optionally get descriptor image
        descriptor = [] # this means empty descriptor image
        if self._descriptor_image_data_file is not None:
            descriptor = self.get_image(camera_name, idx, 'descriptor')

        # optionally get descriptor_keypoints
        descriptor_keypoints = []

        # if self._descriptor_keypoints_data_file is not None:
        #     descriptor_keypoints = self.get_image(camera_name, idx, 'descriptor_keypoints')

        if self._descriptor_keypoints_data is not None:
            descriptor_keypoints = self.get_image(camera_name, idx, 'descriptor_keypoints')

        descriptor_keypoints_3d_world_frame = []
        try:
            descriptor_keypoints_3d_world_frame = self.get_image(camera_name, idx, "descriptor_keypoints_3d_world_frame")
        except KeyError:
            pass


        descriptor_keypoints_3d_camera_frame = []
        try:
            descriptor_keypoints_3d_camera_frame = self.get_image(camera_name, idx, "descriptor_keypoints_3d_camera_frame")
        except KeyError:
            pass

        T_W_C = self.camera_pose(camera_name, idx)
        K = self.camera_K_matrix(camera_name)

        return {'rgb': rgb,
                'label': label,
                'mask': mask,
                'depth_float32': depth_float32, # meters
                'depth_int16': depth_int16, # millimeters,
                'descriptor': descriptor,
                'descriptor_keypoints': descriptor_keypoints,
                'descriptor_keypoints_3d_world_frame': descriptor_keypoints_3d_world_frame,
                'descriptor_keypoints_3d_camera_frame': descriptor_keypoints_3d_camera_frame,
                'K': K,
                'T_world_camera': T_W_C,
                'camera_name': camera_name,
                'idx': idx,
                'episode_name': self.episode_name,
                }

    def get_image_data_specified_in_config(self,
                                           camera_name,
                                           idx,
                                           image_config,  # dict
                                           ):

        """
        See superclass method for documentation
        """

        return_data = dict()

        image_types = ['rgb', 'label', 'mask', 'depth_float32', 'depth_int16', 'descriptor',
                       'descriptor_keypoints', 'descriptor_keypoints_3d_world_frame', 'descriptor_keypoints_3d_camera_frame']

        for image_type in image_types:
            flag = False

            if image_type in image_config:
                if image_config[image_type]:
                    if image_type == "descriptor":
                        if self._descriptor_image_data_file is not None:
                            flag = True
                    else:
                        flag = True

            if flag:
                return_data[image_type] = self.get_image(camera_name, idx, image_type)


        T_W_C = self.camera_pose(camera_name, idx)
        K = self.camera_K_matrix(camera_name)

        return_data['T_W_C'] = T_W_C
        return_data['camera_name'] = camera_name
        return_data['idx'] = idx
        return_data['K'] = K

        return return_data

    def make_index(self,
                   episode_name=None, # optional episode name
                   camera_names=None, # (optional) list[str], which cameras to use
                   ):
        """
        Index used as part of dense_correspondence.dataset.dynamic_drake_sim_dataset.DynamicDrakeSimDataset
        :return:
        :rtype:
        """

        if episode_name is None:
            episode_name = self.name


        if camera_names is None:
            camera_names = self.camera_names
        else:
            camera_names = list(set(camera_names) & set(self.camera_names))

        index = []
        for idx in range(self.length):
            for camera_name_a, camera_name_b in itertools.product(camera_names, camera_names):
                if camera_name_a == camera_name_b:
                    continue
                entry = {'episode_name': episode_name,
                         'idx_a': idx,
                         'idx_b': idx,
                         'camera_name_a': camera_name_a,
                         'camera_name_b': camera_name_b}

                index.append(entry)


        return index

    def make_single_image_index(self,
                                episode_name=None,  # name to give this episode
                                camera_names=None,  # (optional) list[str]
                                ):
        """
        Makes index to iterate through all the images.

        Use as part of key_dynam.dense_correspondence.image_dataset.ImageDataset
        :param episode_name:
        :type episode_name:
        :return:
        :rtype:
        """

        if episode_name is None:
            episode_name = self.name

        index = []
        camera_names = self.camera_names
        for idx in range(self.length):
            for camera_name in camera_names:
                entry = {'episode_name': episode_name,
                         'idx': idx,
                         'camera_name': camera_name,
                         }

                index.append(entry)


        return index



    @staticmethod
    def metadata_file(dataset_root):
        return os.path.join(dataset_root, 'metadata.yaml')

    @staticmethod
    def config_file(dataset_root):
        return os.path.join(dataset_root, 'config.yaml')

    @staticmethod
    def load_dataset(dataset_root, # str: folder containing dataset
                     descriptor_images_root=None, # str: (optional) folder containing hdf5 files with descriptors
                     descriptor_keypoints_root=None,
                     precomputed_data_root=None,
                     max_num_episodes=None, # int, max num episodes to load
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

        metadata = load_yaml(DCDrakeSimEpisodeReader.metadata_file(dataset_root))
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

            # load image data
            image_data_file = os.path.join(dataset_root, val['image_data_file'])
            assert os.path.isfile(image_data_file), "File doesn't exist: %s" %(image_data_file)

            descriptor_image_data_file = None
            if descriptor_images_root is not None:
                descriptor_image_data_file = os.path.join(descriptor_images_root, val['image_data_file'])

                assert os.path.isfile(descriptor_image_data_file), "File doesn't exist: %s" %(descriptor_image_data_file)

            descriptor_keypoints_data = None
            descriptor_keypoints_hdf5_file = None
            if descriptor_keypoints_root is not None:

                # replace .h5 filename with .p for pickle file
                descriptor_keypoints_data_file = val['image_data_file'].split(".")[0] + ".p"
                descriptor_keypoints_data_file = os.path.join(descriptor_keypoints_root, descriptor_keypoints_data_file)

                descriptor_keypoints_hdf5_file = os.path.join(descriptor_keypoints_root, val['image_data_file'])

                if os.path.isfile(descriptor_keypoints_data_file):
                    descriptor_keypoints_data = load_pickle(descriptor_keypoints_data_file)
                else:
                    assert os.path.isfile(descriptor_keypoints_hdf5_file), "File doesn't exist: %s" %(descriptor_keypoints_hdf5_file)




            #############

            precomputed_data = None
            precomputed_data_file = None
            if precomputed_data_root is not None:

                # replace .h5 filename with .p for pickle file
                precomputed_data_file = val['image_data_file'].split(".")[0] + ".p"
                precomputed_data_file = os.path.join(precomputed_data_root, precomputed_data_file)

                if os.path.isfile(precomputed_data_file):
                    precomputed_data = load_pickle(precomputed_data_file)
                else:
                    raise ValueError("file doesn't exist: %s" %(precomputed_data_file))




            episode_reader = DCDrakeSimEpisodeReader(non_image_data,
                                                   image_data_file,
                                                   descriptor_image_data_file=descriptor_image_data_file, descriptor_keypoints_data=descriptor_keypoints_data,
                                                   descriptor_keypoints_data_file=descriptor_keypoints_hdf5_file,
                                                     episode_name=episode_name,
                                                     precomputed_data=precomputed_data,
                                                     precomputed_data_file=precomputed_data_file)

            multi_episode_dict[episode_name] = episode_reader


        return multi_episode_dict


