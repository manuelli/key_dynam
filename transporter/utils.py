import os
import cv2
import sys
import numpy as np
import h5py
import torch

from PIL import Image, ImageOps, ImageEnhance

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

# dense correspondence
from dense_correspondence_manipulation.utils.constants import DEPTH_IM_SCALE
from dense_correspondence_manipulation.utils import utils as pdc_utils
from dense_correspondence_manipulation.utils import torch_utils as pdc_torch_utils

# key_dynam
from key_dynam.utils import transform_utils, torch_utils
from key_dynam.utils.utils import return_value_or_default, load_yaml



def rand_float(lo, hi):
    return np.random.rand() * (hi - lo) + lo


def rand_int(lo, hi):
    return np.random.randint(lo, hi)


def calc_dis(a, b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


def norm(x, p=2):
    return np.power(np.sum(x ** p), 1. / p)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def to_var(tensor, use_gpu, requires_grad=False):
    if use_gpu:
        return Variable(torch.FloatTensor(tensor).cuda(), requires_grad=requires_grad)
    else:
        return Variable(torch.FloatTensor(tensor), requires_grad=requires_grad)


def to_np(x):
    return x.detach().cpu().numpy()


'''
data utils
'''

def store_data(data_names, data, path):
    hf = h5py.File(path, 'w')
    for i in range(len(data_names)):
        hf.create_dataset(data_names[i], data=data[i])
    hf.close()


def load_data(data_names, path):
    hf = h5py.File(path, 'r')
    data = []
    for i in range(len(data_names)):
        d = np.array(hf.get(data_names[i]))
        data.append(d)
    hf.close()
    return data


def combine_stat(stat_0, stat_1):
    mean_0, std_0, n_0 = stat_0[:, 0], stat_0[:, 1], stat_0[:, 2]
    mean_1, std_1, n_1 = stat_1[:, 0], stat_1[:, 1], stat_1[:, 2]

    mean = (mean_0 * n_0 + mean_1 * n_1) / (n_0 + n_1)
    std = np.sqrt(
        (std_0 ** 2 * n_0 + std_1 ** 2 * n_1 + (mean_0 - mean) ** 2 * n_0 + (mean_1 - mean) ** 2 * n_1) / (n_0 + n_1))
    n = n_0 + n_1

    return np.stack([mean, std, n], axis=-1)


def init_stat(dim):
    # mean, std, count
    return np.zeros((dim, 3))


'''
image utils
'''

def resize(img, size, interpolation=Image.BILINEAR):

    if isinstance(size, int):
        w, h = img.size
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            return img.resize((ow, oh), interpolation)
        else:
            oh = size
            ow = int(size * w / h)
            return img.resize((ow, oh), interpolation)
    else:
        return img.resize(size[::-1], interpolation)


def crop(img, i, j, h, w):
    return img.crop((j, i, j + w, i + h))


def map_cropped_pixels_to_full_pixels(kp, # [n_kp, 2], in [-1,1]
                                      crop_params,
                                      ): # [n_kp, 2] in x,y (u, v) = (x,y) coordinates on the original image
    """
    Given the output of the Transporter keypoint network, kp with shape [n_kp, 2]
    map it back to pixel coordinates in the original full-sized image.

    :param kp:
    :type kp:
    :param crop_params:
    :type crop_params:
    :return:
    :rtype:
    """

    x = crop_params['x_min'] + (kp[:, 0] + 1)*crop_params['s_x']/2.0
    y = crop_params['y_min'] + (kp[:, 1] + 1)*crop_params['s_y']/2.0

    return np.stack((x,y), axis=-1)



def map_cropped_pixels_to_full_pixels_torch(kp, # [B, n_kp, 2], in [-1,1]
                                      crop_params, #
                                      ): # [n_kp, 2] in uv coordinates on the original image
    """
    Given the output of the Transporter keypoint network, kp with shape [n_kp, 2]
    map it back to pixel coordinates in the original full-sized image.

    :param kp:
    :type kp:
    :param crop_params:
    :type crop_params:
    :return:
    :rtype:
    """

    device = kp.device

    # [B,
    # crop_params['x_min'].shape is [B,]
    # kp.shape is [B, n_kp, 2]
    # so broadcasting semantics don't work

    x_min = crop_params['x_min'].view(-1, 1).to(device).type_as(kp)
    y_min = crop_params['y_min'].view(-1, 1).to(device).type_as(kp)
    s_x = crop_params['s_x'].view(-1, 1).to(device).type_as(kp)
    s_y = crop_params['s_y'].view(-1, 1).to(device).type_as(kp)

    # view(-1,1,1) converts x_min from [B,] to [B, 1, 1] so that broadcasting semantics work
    x = x_min + (kp.select(dim=-1, index=0) + 1)*s_x/2.0
    y = y_min + (kp.select(dim=-1, index=1) + 1)*s_y/2.0


    return torch.stack((x,y), dim=-1)



def map_transporter_keypoints_to_full_image(kp, # torch.Tensor [B, n_kp, 2] in [-1, 1] (x-y coordinates in OpenCV)
                                            crop_params=None, # if they exist
                                            full_image_size=None, # (H, W)
                                            ):

    assert full_image_size is not None
    H, W = full_image_size

    # no crop
    xy = None
    uv = None
    if crop_params is None:
        xy = kp
        u = (xy[:, :, 0] + 1)/2.0 * W
        v = (xy[:, :, 1] + 1)/2.0 * H
        uv = torch.stack((u, v), dim=-1)
    else: # there was a crop
        uv = map_cropped_pixels_to_full_pixels_torch(kp, crop_params)
        x = uv[:, :, 0] * 2.0 / W - 1.0
        y = uv[:, :, 1] * 2.0 / H - 1.0
        xy = torch.stack((x,y), dim=-1)


    uv_int = uv.type(torch.LongTensor)


    return {'uv': uv,
            'uv_int': uv_int,
            'xy': xy,
            }

def adjust_brightness(img, brightness_factor):
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(brightness_factor)
    return img


def adjust_contrast(img, contrast_factor):
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast_factor)
    return img


def adjust_saturation(img, saturation_factor):
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(saturation_factor)
    return img


def adjust_hue(img, hue_factor):

    if not(-0.5 <= hue_factor <= 0.5):
        raise ValueError('hue_factor is not in [-0.5, 0.5].'.format(hue_factor))

    input_mode = img.mode
    if input_mode in {'L', '1', 'I', 'F'}:
        return img

    h, s, v = img.convert('HSV').split()

    np_h = np.array(h, dtype=np.uint8)
    # uint8 addition take cares of rotation across boundaries
    with np.errstate(over='ignore'):
        np_h += np.uint8(hue_factor * 255)
    h = Image.fromarray(np_h, 'L')

    img = Image.merge('HSV', (h, s, v)).convert(input_mode)
    return img


def adjust_gamma(img, gamma, gain=1):

    if gamma < 0:
        raise ValueError('Gamma should be a non-negative real number')

    input_mode = img.mode
    img = img.convert('RGB')

    gamma_map = [255 * gain * pow(ele / 255., gamma) for ele in range(256)] * 3
    img = img.point(gamma_map)  # use PIL's point-function to accelerate this part

    img = img.convert(input_mode)
    return img



'''
record utils
'''

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



"""
Make debug images
"""

def visualize_transporter_output(des, # [B, 3, H, W],
                                 des_pred, # [B, 3, H, W]
                                 heatmap, # [B, -1, H, W]
                                 kp, # [B, -1] # keypoints
                                 ):

    nrows=1
    ncols=4
    scale=5

    n_split, d_split = 5, 4

    height_demo = 256
    width_demo = 256

    c = [(255, 105, 65), (0, 69, 255), (50, 205, 50), (0, 165, 255), (238, 130, 238),
         (128, 128, 128), (30, 105, 210), (147, 20, 255), (205, 90, 106), (255, 215, 0)]

    des_pred = to_np(torch.clamp(des_pred, -1., 1.)).transpose(0, 2, 3, 1)[..., ::-1]
    des_pred = (des_pred * 0.5 + 0.5) * 255.

    # the real des
    des = to_np(torch.clamp(des, -1., 1.)).transpose(0, 2, 3, 1)[..., ::-1]
    des = (des * 0.5 + 0.5) * 255.

    # predicted keypoints
    kp = to_np(kp) + 1.
    kp = kp * height_demo / 2.
    kp = np.round(kp).astype(np.int)

    # corresponding heatmap
    heatmap = to_np(heatmap).transpose(0, 2, 3, 1)
    heatmap = np.sum(heatmap, 3, keepdims=True)
    heatmap = np.clip(heatmap * 255., 0., 255.)

    images = []
    for j in range(des_pred.shape[0]):
        des_pred_cur = des_pred[j]
        des_pred_cur = cv2.resize(des_pred_cur, (width_demo, height_demo))

        des_cur = des[j]
        des_cur = cv2.resize(des_cur, (width_demo, height_demo))

        kp_cur = kp[j]

        heatmap_cur = heatmap[j]
        heatmap_cur = cv2.resize(heatmap_cur, (width_demo, height_demo),
                                 interpolation=cv2.INTER_NEAREST)

        # visualization
        kp_map = np.zeros((des_cur.shape[0], des_cur.shape[1], 3))

        for k in range(kp_cur.shape[0]):
            cv2.circle(kp_map, (kp_cur[k, 0], kp_cur[k, 1]), 6, c[k], -1)

        overlay_gt = des_cur * 0.5 + kp_map * 0.5

        h, w = des_cur.shape[:2]
        merge = np.zeros((h, w * n_split + d_split * (n_split - 1), 3))

        merge[:, :w] = des_cur
        merge[:, (w + d_split) * 1: (w + d_split) * 1 + w] = kp_map
        merge[:, (w + d_split) * 2: (w + d_split) * 2 + w] = overlay_gt
        merge[:, (w + d_split) * 3: (w + d_split) * 3 + w] = heatmap_cur[..., None]
        merge[:, (w + d_split) * 4:] = des_pred_cur
        merge = merge.astype(np.uint8)
        images.append(merge)


    return images


