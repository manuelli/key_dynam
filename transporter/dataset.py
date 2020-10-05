import os
import cv2
import time
import numpy as np
import random



import torch
from torch.utils.data import Dataset
from torchvision import transforms

from key_dynam.utils.utils import get_project_root, number_to_base, return_value_or_default
from key_dynam.dense_correspondence.image_dataset import ImageDataset


def find_crop_param(mask):
    """
    Finds 4 corners of a square bounding box (x_min, y_min)
    (x_max, y_max) that contains the mask
    :param mask:
    :type mask:
    :return:
    :rtype:
    """
    height, width = mask.shape
    nonzero_idx = np.where(mask == 1)

    y_min = np.min(nonzero_idx[0])
    y_max = np.max(nonzero_idx[0])

    x_min = np.min(nonzero_idx[1])
    x_max = np.max(nonzero_idx[1])

    x_l = y_max - y_min
    y_l = x_max - x_min

    # ensures that it is always a square crop . . . .
    if x_l > y_l:
        x_min -= (x_l - y_l) // 2
        x_max = x_min + x_l
    if y_l > x_l:
        y_min -= (y_l - x_l) // 2
        y_max = y_min + y_l

    assert y_max - y_min == x_max - x_min

    s = y_max - y_min

    if y_min < 0:
        y_min, y_max = 0, s
    if y_max > height:
        y_min, y_max = height - s, height
    if x_min < 0:
        x_min, x_max = 0, s
    if x_max > width:
        x_min, x_max = width - s, width

    return {'x_min': x_min,
            'x_max': x_max,
            'y_min': y_min,
            'y_max': y_max,
            's_x': s,
            's_y': s}


def crop_and_resize(img, crop_param, scale_size):
    """
    Crops the image and resizes it.
    """
    # crop the images

    img = img[crop_param['y_min']:crop_param['y_max'],
              crop_param['x_min']:crop_param['x_max']]

    # resize the images
    img = cv2.resize(img, scale_size)

    return img


# crops/resizes the image as necessary according to the config
def process_image(rgb,
                  config,
                  mask=None,  # only needed if doing crop
                  ):
    H_input = config['perception']['height']
    W_input = config['perception']['width']

    scale_size = (H_input, W_input)

    crop_param = None
    img_processed = None

    crop_enabled = return_value_or_default(config['perception'],
                                           'crop_enabled',
                                           True)
    if crop_enabled:
        crop_param = find_crop_param(mask)
        img_processed = crop_and_resize(rgb,
                                        crop_param,
                                        scale_size=scale_size)
    else:
        rgb_masked = rgb * mask[..., None].astype(rgb.dtype)
        img_processed = cv2.resize(rgb_masked, scale_size)


    return {'image': img_processed,
            'crop_param': crop_param,
            }

class ImageTupleDataset(Dataset):

    def __init__(self,
                 config,
                 episodes,
                 phase="train",
                 image_data_config=None, # specifies the information to return from the dataset
                 tuple_size=2,
                 compute_K_inv=False,
                 camera_names = None, # will use all cameras unless you specify otherwise
                 ):

        self._config = config
        self._episodes = episodes
        self._tuple_size = tuple_size
        self._phase = phase
        self._compute_K_inv = compute_K_inv
        self._camera_names = camera_names

        # for backwards compatibility
        self._use_transporter_type_data_sampling = None
        try:
            self._use_transporter_type_data_sampling = self._config['dataset']['use_transporter_type_data_sampling']
        except KeyError:
            self._use_transporter_type_data_sampling = True



        self._image_dataset = ImageDataset(config,
                                           episodes,
                                           phase=phase,
                                           camera_names=self._camera_names)


        self._episode_names = self._image_dataset.episode_names

        # self.scale_size = config['perception']['scale_size']
        self.width = config['perception']['width']
        self.height = config['perception']['height']
        self.crop_enabled = True

        # for backwards compatibility
        try:
            self.crop_enabled = config['perception']['crop_enabled']
        except KeyError:
            pass

        if self.crop_enabled:
            # must have square if you want crop stuff
            assert self._config['perception']['width'] == self._config['perception']['height']

        if image_data_config is None:
            image_data_config = {'rgb': True,
                                 'mask': True,
                                 }

        self._image_data_config = image_data_config
        self._rgb_image_to_tensor = self.make_rgb_image_to_tensor_transform()


        self._epoch_size = None

        if self._phase != "all":
            if "epoch_size" in self._config['dataset'] and self._config['dataset']["epoch_size"][self._phase] > 0:
                print("setting fixed epoch size")
                self._epoch_size = self._config['dataset']["epoch_size"][self._phase]

            # set the epoch size to be the number of images in self._image_dataset, rather than
            # the number of tuples of such images
            if "set_epoch_size_to_num_images" in self._config['dataset'] and self._config['dataset']["set_epoch_size_to_num_images"]:
                self._epoch_size = len(self._image_dataset)

    @property
    def episodes(self):
        return self._episodes

    def length_actual(self):
        """
        Number of tuples of images from the ImageDataset with replacement
        :return:
        :rtype:
        """
        return len(self._image_dataset)**self._tuple_size

    def __len__(self):
        if self._epoch_size is not None:
            return min(self._epoch_size, self.length_actual())
        else:
            return self.length_actual()



    def get_single_image_data(self,
                              episode,
                              camera_name,
                              idx):

        """
        Gets the data for a single image
        """


        data = episode.get_image_data_specified_in_config(camera_name,
                                                          idx,
                                                          self._image_data_config)

        data['episode_name'] = episode.name
        data['idx'] = idx
        data['key_tree'] = episode.get_image_key_tree(camera_name, idx)
        data['key_tree_joined'] = "/".join(data['key_tree'])

        # cast it to Tensor
        data['rgb_tensor'] = self._rgb_image_to_tensor(data['rgb'])

        if self._compute_K_inv:
            data['K_inv'] = np.linalg.inv(data['K'])

        rgb = data['rgb']


        # the cast is important to ensure that rgb_zero_out_background is still of type
        # np.uint8. This is required to be able to use the self._rgb_image_to_tensor
        # function
        rgb_zero_out_background = rgb * data['mask'][...,None].astype(rgb.dtype)

        data['rgb_masked'] = rgb_zero_out_background
        data['rgb_masked_scaled'] = cv2.resize(rgb_zero_out_background, (self.height, self.width))
        data['rgb_masked_scaled_tensor'] = self._rgb_image_to_tensor(data['rgb_masked_scaled'])

        if self.crop_enabled:
            # compute the crop params
            data['crop_param'] = find_crop_param(data['mask'])
            # data['scale_size'] = self.scale_size
            data['rgb_crop'] = crop_and_resize(rgb_zero_out_background, data['crop_param'],
                                               (self.height, self.width))


            data['crop_and_resize_shape'] = data['rgb_crop'].shape
            # cast it to a tensor
            data['rgb_crop_tensor'] = self._rgb_image_to_tensor(data['rgb_crop'])


        return data

    def transporter_image_pair_index(self,
                                     episode=None, # DC EpisodeReader, if None will randomly sample one
                                     ):

        assert self._tuple_size == 2
        episode_name = None
        if episode is None:
            episode_name = random.sample(self._episode_names, 1)[0]
            episode = self._episodes[episode_name]
        else:
            episode_name = episode.name



        low = 0
        high = episode.length - 1
        mid = (low + high)//2

        idx_1 = random.randint(0, mid)
        idx_2 = random.randint(mid+1, high)

        camera_set = set(episode.camera_names)
        camera_set.intersection_update(set(self._camera_names))


        camera_name_1 = random.sample(camera_set, 1)[0]
        camera_name_2 = random.sample(camera_set, 1)[0]


        entry_1 = {'episode': episode,
                   'idx': idx_1,
                   'camera_name': camera_name_1,
                   'episode_name': episode_name}

        entry_2 = {'episode': episode,
                   'idx': idx_2,
                   'camera_name': camera_name_2,
                   'episode_name': episode_name}


        return [entry_1, entry_2]

    def get_image_tuple_index(self, idx):
        """
        Gets a tuple of image indexes

        :param idx:
        :type idx:
        :return:
        :rtype:
        """
        if self._epoch_size is not None:
            idx = random.randint(0, self.length_actual() - 1)

        # this might be less than self._tuple_size so add zeros to the front
        # if necessary
        base_repr = number_to_base(idx, len(self._image_dataset))

        if len(base_repr) < self._tuple_size:
            base_repr = [0]*(self._tuple_size - len(base_repr)) + base_repr

        entry_list = []

        for j in base_repr:
            entry_list.append(self._image_dataset.index[j])

        return entry_list

    # def get_random_image_tuple_index(self):
    #
    #     episode_names = self._episode_names
    #     entry_list = []
    #
    #     for i in range(self._tuple_size):
    #         episode_name = random.sample(self._episode_names, 1)[0]
    #         episode = self._episodes[episode_name]
    #
    #         camera_set = set(episode.camera_names)
    #         camera_set.intersection_update(set(self._camera_names))
    #         camera_name = random.sample(camera_set, 1)[0]
    #
    #         entry = {}
    def __getitem__(self, idx):

        entry_list = None
        if self._use_transporter_type_data_sampling:
            entry_list = self.transporter_image_pair_index()
        else:
            entry_list = self.get_image_tuple_index(idx)

        data_dict = {}
        for i, index_entry in enumerate(entry_list):
            episode = self._episodes[index_entry['episode_name']]
            data = self.get_single_image_data(episode,
                                              index_entry['camera_name'],
                                              index_entry['idx'],
                                              )

            data_dict[i] = data

        return data_dict


    @staticmethod
    def make_rgb_image_to_tensor_transform():
        """
        Makes the rgb --> tensor transform used in this dataset
        Normalizes the image to [-1,1]
        :return:
        :rtype:
        """
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5])

        return transforms.Compose([transforms.ToTensor(), normalize])

