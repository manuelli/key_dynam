import cv2
import functools

import torch
import torchvision

from key_dynam.dense_correspondence.image_dataset import ImageDataset
from key_dynam.utils.torch_utils import make_default_image_to_tensor_transform



class AutoencoderImagePreprocessFunctionFactory:

    @staticmethod
    def spatial_autoencoder(config):
        height = config['perception']['height']
        width = config['perception']['width']

        assert height == 240

        H_out = None
        W_out = None
        if width == 240:
            H_out = 60
            W_out = 60
        elif width == 320:
            H_out = 60
            W_out = 80


        rgb_to_tensor = make_default_image_to_tensor_transform()
        target_to_tensor = torchvision.transforms.ToTensor()

        def func(rgb, mask=None, **kwargs):


            d = spatial_autoencoder_image_preprocessing(rgb,
                                                        H_in=height,
                                                        W_in=width,
                                                        H_out=H_out,
                                                        W_out=W_out,
                                                        mask=mask,
                                                        )

            d['input_tensor'] = rgb_to_tensor(d['input'])
            d['target_tensor'] = target_to_tensor(d['target'])

            return d

        return func

    @staticmethod
    def convolutional_autoencoder(config):
        height = config['perception']['height']
        width = config['perception']['width']

        assert height == 64
        assert width == 64

        H_out = 64
        W_out = 64

        rgb_to_tensor = make_default_image_to_tensor_transform()
        target_to_tensor = torchvision.transforms.ToTensor()


        def func(rgb, mask=None, **kwargs):

            d = convolutional_autoencoder_image_preprocessing(rgb,
                                                              H_in=height,
                                                              W_in=width,
                                                              H_out=H_out,
                                                              W_out=W_out,
                                                              mask=mask)

            d['input_tensor'] = rgb_to_tensor(d['input'])
            d['target_tensor'] = target_to_tensor(d['target'])
            return d

        return func

    @staticmethod
    def convolutional_autoencoder_from_episode(config):
        """
        Returns
        :param config:
        :type config:
        :return:
        :rtype:
        """

        image_preprocess_func = AutoencoderImagePreprocessFunctionFactory.convolutional_autoencoder(config)

        camera_name = config['perception']['camera_name']

        def func(episode=None, # key_dynam.dataset.EpisodeReader
                 episode_idx=None,
                 T_aug=None,
                 **kwargs):

            # we don't support T_aug for this type of data
            assert T_aug is None

            image_episode = episode.image_episode
            image_episode_idx = episode_idx
            rgb = image_episode.get_image(camera_name=camera_name,
                                          idx=image_episode_idx,
                                          type='rgb')

            mask = image_episode.get_image(camera_name=camera_name,
                                           idx=image_episode_idx,
                                           type='mask',
                                           )

            d = image_preprocess_func(rgb=rgb, mask=mask)
            d['tensor'] = torch.Tensor([]) # empty tensor
            return d



        return func
def image_preprocess_func(rgb, *kwargs):
    """
    Example of what such a function should look like
    :param rgb:
    :type rgb:
    :return:
    :rtype:
    """
    return {'input': None,
            'input_tensor': None,
            'target': None,
            'target_tensor': None,
            'mask': None,
            'mask_tensor': None}

def spatial_autoencoder_image_preprocessing(rgb,
                                            H_in=None,
                                            W_in=None,
                                            H_out=None,
                                            W_out=None,
                                            mask=None,
                                            *kwargs):
    # get downsampled version of image

    input_img = cv2.resize(rgb, (W_in, H_in))

    img_gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    target_img = cv2.resize(img_gray, (W_out, H_out))

    target_mask = [] # empty tensor
    if mask is not None:
        target_mask = cv2.resize(mask, (W_out, H_out))

    return {'input': input_img,
            'target': target_img,
            'target_mask': target_mask}


def convolutional_autoencoder_image_preprocessing(rgb,
                                            H_in=None,
                                            W_in=None,
                                            H_out=None,
                                            W_out=None,
                                            mask=None,
                                            *kwargs):


    # get downsampled version of image
    input_img = cv2.resize(rgb, (W_in, H_in))
    target_img = cv2.resize(input_img, (W_out, H_out))

    target_mask = [] # empty tensor
    if mask is not None:
        target_mask = cv2.resize(mask, (W_out, H_out))

    return {'input': input_img,
            'target': target_img,
            'target_mask': target_mask}


class AutoencoderImageDataset(ImageDataset):

    def __init__(self,
                 config, # global
                 episodes,
                 phase="train",
                 camera_names=None, #(optional) list[str]: list of cameras to use, defaults to all
                 image_preprocess_func=None, # func as above
                 ):

        super(AutoencoderImageDataset, self).__init__(config=config,
                                                      episodes=episodes,
                                                      phase=phase,
                                                      camera_names=camera_names)


        self._image_preprocess_function = image_preprocess_func
        self._rgb_to_tensor = make_default_image_to_tensor_transform()
        self._grayscale_to_tensor = torchvision.transforms.ToTensor()


    def _getitem(self,
                 episode,  # EpisodeReader
                 idx,  # int
                 camera_name,  # str
                 ):

        rgb = episode.get_image(camera_name, idx, 'rgb')
        mask = episode.get_image(camera_name, idx, 'mask')
        d = self._image_preprocess_function(rgb, mask=mask)

        return {'rgb': rgb,
                'mask': mask,
                'target_mask': d['target_mask'],
                'input': d['input'],
                'input_tensor': d['input_tensor'],
                'target': d['target'],
                'target_tensor': d['target_tensor'],
                'camera_name': camera_name,
                'idx': idx,
                'episode_name': episode.name,
                }


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

