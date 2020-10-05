import os
import time
import numpy as np
import cv2

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from key_dynam.utils.utils import load_yaml




# dense correspondence
from dense_correspondence_manipulation.utils import utils as pdc_utils
from dense_correspondence_manipulation.utils import torch_utils as pdc_torch_utils

# key_dynam
from key_dynam.utils import torch_utils
from key_dynam.transporter.dataset import find_crop_param, crop_and_resize, process_image, ImageTupleDataset
from key_dynam.transporter.utils import map_cropped_pixels_to_full_pixels_torch, map_transporter_keypoints_to_full_image

class SpatialSoftmax(torch.nn.Module):
    def __init__(self, height, width, channel, lim=[-1., 1., -1., 1.], temperature=None, data_format='NCHW'):
        super(SpatialSoftmax, self).__init__()
        self.data_format = data_format
        self.height = height
        self.width = width
        self.channel = channel

        if temperature:
            self.temperature = Parameter(torch.ones(1) * temperature)
        else:
            self.temperature = 1.

        pos_x, pos_y = np.meshgrid(
            np.linspace(lim[0], lim[1], self.width),
            np.linspace(lim[2], lim[3], self.height))

        pos_x = torch.from_numpy(pos_x.reshape(self.height * self.width)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self.height * self.width)).float()
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)

    def forward(self, feature):
        """
        It seems like the output is in [x,y] (u,v) coordinates
        x is width
        y is height
        """
        # Output:
        #   (N, C*2) x_0 y_0 ...
        if self.data_format == 'NHWC':
            feature = feature.transpose(1, 3).tranpose(2, 3).view(-1, self.height * self.width)
        else:
            feature = feature.view(-1, self.height * self.width)

        softmax_attention = F.softmax(feature / self.temperature, dim=-1)
        expected_x = torch.sum(Variable(self.pos_x) * softmax_attention, dim=1, keepdim=True)
        expected_y = torch.sum(Variable(self.pos_y) * softmax_attention, dim=1, keepdim=True)
        expected_xy = torch.cat([expected_x, expected_y], 1)
        feature_keypoints = expected_xy.view(-1, self.channel, 2)

        return feature_keypoints


class KeyPointPredictor(nn.Module):
    def __init__(self, config, lim=[-1., 1., -1., 1.]):
        super(KeyPointPredictor, self).__init__()

        nf = config['train']['nf_hidden']
        k = config['perception']['n_kp']
        norm_layer = config['train']['norm_layer']

        sequence = [
            # input is (ni) x 64 x 64
            nn.Conv2d(3, nf, 7, 1, 3),
            nn.BatchNorm2d(nf) if norm_layer == 'Batch' else nn.InstanceNorm2d(nf),
            nn.LeakyReLU(0.2, inplace=True),
            # feat size (nf) x 64 x 64
            nn.Conv2d(nf, nf, 5, 1, 2),
            nn.BatchNorm2d(nf) if norm_layer == 'Batch' else nn.InstanceNorm2d(nf),
            nn.LeakyReLU(0.2, inplace=True),
            # fesrcat size (nf) x 64 x 64
            nn.Conv2d(nf, nf * 2, 4, 2, 1),
            nn.BatchNorm2d(nf * 2) if norm_layer == 'Batch' else nn.InstanceNorm2d(nf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # feat size (nf * 2) x 32 x 32
            nn.Conv2d(nf * 2, nf * 2, 3, 1, 1),
            nn.BatchNorm2d(nf * 2) if norm_layer == 'Batch' else nn.InstanceNorm2d(nf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # feat size (nf * 2) x 32 x 32
            nn.Conv2d(nf * 2, nf * 4, 4, 2, 1),
            nn.BatchNorm2d(nf * 4) if norm_layer == 'Batch' else nn.InstanceNorm2d(nf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # feat size (nf * 4) x 16 x 16
            nn.Conv2d(nf * 4, k, 1, 1)
            # feat size (n_kp) x 16 x 16
        ]

        self.model = nn.Sequential(*sequence)
        self.integrater = SpatialSoftmax(
            height=config['perception']['height']//4,
            width=config['perception']['width']//4, channel=k, lim=lim)

    def integrate(self, heatmap):
        return self.integrater(heatmap)

    def forward(self, img):
        """
        Returns keypoint locations [B, n_kp, 2] in [v,u] ordering
        The values are in the range [-1,1] for each dimension
        :param img:
        :type img:
        :return:
        :rtype:
        """
        heatmap = self.model(img)
        return self.integrate(heatmap)


class FeatureExtractor(nn.Module):
    def __init__(self, config):
        super(FeatureExtractor, self).__init__()

        nf = config['train']['nf_hidden']
        norm_layer = config['train']['norm_layer']

        sequence = [
            # input is (ni) x 64 x 64
            nn.Conv2d(3, nf, 7, 1, 3),
            nn.BatchNorm2d(nf) if norm_layer == 'Batch' else nn.InstanceNorm2d(nf),
            nn.LeakyReLU(0.2, inplace=True),
            # feat size (nf) x 64 x 64
            nn.Conv2d(nf, nf, 5, 1, 2),
            nn.BatchNorm2d(nf) if norm_layer == 'Batch' else nn.InstanceNorm2d(nf),
            nn.LeakyReLU(0.2, inplace=True),
            # feat size (nf) x 64 x 64
            nn.Conv2d(nf, nf * 2, 4, 2, 1),
            nn.BatchNorm2d(nf * 2) if norm_layer == 'Batch' else nn.InstanceNorm2d(nf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # feat size (nf * 2) x 32 x 32
            nn.Conv2d(nf * 2, nf * 2, 3, 1, 1),
            nn.BatchNorm2d(nf * 2) if norm_layer == 'Batch' else nn.InstanceNorm2d(nf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # feat size (nf * 2) x 32 x 32
            nn.Conv2d(nf * 2, nf * 4, 4, 2, 1),
            nn.BatchNorm2d(nf * 4) if norm_layer == 'Batch' else nn.InstanceNorm2d(nf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # feat size (nf * 4) x 16 x 16
        ]

        self.model = nn.Sequential(*sequence)

    def forward(self,
                img, # B x 3 x H x W
                ): # B x 3 x H/4 x W/4 (4x lower resolution)
        return self.model(img)


class Refiner(nn.Module):
    def __init__(self, config):
        super(Refiner, self).__init__()

        nf = config['train']['nf_hidden']
        k = config['perception']['n_kp']
        norm_layer = config['train']['norm_layer']

        sequence = [
            # input is (nf * 4) x 16 x 16
            nn.ConvTranspose2d(nf * 4, nf * 4, 4, 2, 1),
            nn.BatchNorm2d(nf * 4) if norm_layer == 'Batch' else nn.InstanceNorm2d(nf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # input is (nf * 4) x 32 x 32
            nn.Conv2d(nf * 4, nf * 2, 3, 1, 1),
            nn.BatchNorm2d(nf * 2) if norm_layer == 'Batch' else nn.InstanceNorm2d(nf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # input is (nf * 4) x 32 x 32
            nn.ConvTranspose2d(nf * 2, nf * 2, 4, 2, 1),
            nn.BatchNorm2d(nf * 2) if norm_layer == 'Batch' else nn.InstanceNorm2d(nf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # input is (nf * 2) x 64 x 64
            nn.Conv2d(nf * 2, nf, 5, 1, 2),
            nn.BatchNorm2d(nf) if norm_layer == 'Batch' else nn.InstanceNorm2d(nf),
            nn.LeakyReLU(0.2, inplace=True),
            # input is (nf * 2) x 64 x 64
            nn.Conv2d(nf, 3, 7, 1, 3)
        ]

        self.model = nn.Sequential(*sequence)

    def forward(self, feat):
        return self.model(feat)


class Transporter(nn.Module):
    def __init__(self, config, use_gpu=True):
        super(Transporter, self).__init__()

        self.config = config
        self.use_gpu = use_gpu
        self.lim = [-1., 1., -1., 1.]

        # visual feature extractor
        self.feature_extractor = FeatureExtractor(config)

        # key point predictor
        self.keypoint_predictor = KeyPointPredictor(config, lim=self.lim)

        # map the feature back to the image
        self.refiner = Refiner(config)

        lim = self.lim
        self.width = config['perception']['width']
        self.height = config['perception']['height']
        self.n_kp = config['perception']['n_kp']
        self.inv_std = config['perception']['inv_std']

        x = np.linspace(lim[0], lim[1], self.width // 4)
        y = np.linspace(lim[2], lim[3], self.height // 4)
        z = np.linspace(-1., 1., config['perception']['n_kp'])

        if use_gpu:
            self.x = Variable(torch.FloatTensor(x)).cuda()
            self.y = Variable(torch.FloatTensor(y)).cuda()
            self.z = Variable(torch.FloatTensor(z)).cuda()
        else:
            self.x = Variable(torch.FloatTensor(x))
            self.y = Variable(torch.FloatTensor(y))
            self.z = Variable(torch.FloatTensor(z))

    def extract_feature(self, img):
        # img: B x 3 x H x W
        # ret: B x (nf * 4) x (H / 4) x (W / 4)
        return self.feature_extractor(img)

    def predict_keypoint(self, img, kp_gt=None):
        # img: B x 3 x H x W
        # kp_gt: B x n_kp x 2
        # ret: B x n_kp x 2
        if kp_gt is not None:
            return kp_gt
        return self.keypoint_predictor(img)

    def keypoint_to_heatmap(self, keypoint, inv_std=10.):
        # keypoint: B x n_kp x 2
        # heatpmap: B x n_kp x (H / 4) x (W / 4)
        # ret: B x n_kp x (H / 4) x (W / 4)
        height = self.height // 4
        width = self.width // 4

        mu_x, mu_y = keypoint[:, :, :1].unsqueeze(-1), keypoint[:, :, 1:].unsqueeze(-1)
        y = self.y.view(1, 1, height, 1)
        x = self.x.view(1, 1, 1, width)

        g_y = (y - mu_y)**2
        g_x = (x - mu_x)**2
        dist = (g_y + g_x) * inv_std**2

        hmap = torch.exp(-dist)

        return hmap

    def transport(self, src_feat, des_feat, src_hmap, des_hmap, des_feat_hmap=None):
        # src_feat: B x (nf * 4) x (H / 4) x (W / 4)
        # des_feat: B x (nf * 4) x (H / 4) x (W / 4)
        # src_hmap: B x n_kp x (H / 4) x (W / 4)
        # des_hmap: B x n_kp x (H / 4) x (W / 4)
        # des_feat_hmap = des_hmap * des_feat: B x (nf * 4) x (H / 4) * (W / 4)
        # mixed_feat: B x (nf * 4) x (H / 4) x (W / 4)
        src_hmap = torch.sum(src_hmap, 1, keepdim=True)
        des_hmap = torch.sum(des_hmap, 1, keepdim=True)
        src_digged = src_feat * (1. - src_hmap) * (1. - des_hmap)

        # print(src_digged.size())
        # print(des_hmap.size())
        # print(des_feat.size())
        if des_feat_hmap is None:
            mixed_feat = src_digged + des_hmap * des_feat
        else:
            mixed_feat = src_digged + des_feat_hmap

        return mixed_feat

    def refine(self, mixed_feat):
        # mixed_feat: B x (nf * 4) x (H / 4) x (W / 4)
        # ret: B x 3 x H x W
        return self.refiner(mixed_feat)

    def kp_feat(self, feat, hmap):
        # feat: B x (nf * 4) x (H / 4) x (W / 4)
        # hmap: B x n_kp x (H / 4) x (W / 4)
        # ret: B x n_kp x (nf * 4)
        B, nf, H, W = feat.size()
        n_kp = hmap.size(1)

        p = feat.view(B, 1, nf, H, W) * hmap.view(B, n_kp, 1, H, W)
        kp_feat = torch.sum(p, (3, 4))
        return kp_feat

    def forward(self, src, des, kp_gt=None):
        # src: B x 3 x H x W
        # des: B x 3 x H x W
        # kp_gt: B x n_kp x 2
        # des_pred: B x 3 x H x W
        cat = torch.cat([src, des], 0)
        feat = self.extract_feature(cat)
        kp = self.predict_keypoint(cat, kp_gt)
        B = kp.size(0)

        src_feat, des_feat = feat[:B//2], feat[B//2:]
        src_kp, des_kp = kp[:B//2], kp[B//2:]

        # stop gradients on src_feat and src_kp
        # https://github.com/deepmind/deepmind-research/blob/master/transporter/transporter.py#L97
        src_feat = src_feat.detach()
        src_kp = src_kp.detach()

        src_hmap = self.keypoint_to_heatmap(src_kp, self.inv_std)
        des_hmap = self.keypoint_to_heatmap(des_kp, self.inv_std)

        # src_kp_feat = self.kp_feat(src_feat, src_hmap)
        # des_kp_feat = self.kp_feat(des_feat, des_hmap)

        mixed_feat = self.transport(src_feat, des_feat, src_hmap, des_hmap)

        des_pred = self.refine(mixed_feat)

        return des_pred

    @staticmethod
    def load_model_from_checkpoint(model_chkpt_file):
        """
        Assumes the config is stored 2 levels up
        :param model_ckpt_file:
        :type model_ckpt_file:
        :return:
        :rtype:
        """

        train_dir = os.path.dirname(os.path.dirname(model_chkpt_file))
        model_name = os.path.split(train_dir)[-1]

        config = load_yaml(os.path.join(train_dir, 'config.yaml'))

        model_kp = Transporter(config, use_gpu=True)
        model_kp.load_state_dict(torch.load(model_chkpt_file))
        model_kp = model_kp.cuda()
        model_kp = model_kp.eval()

        return {'model': model_kp,
                'model_file': model_chkpt_file,
                'model_name': None,
                'train_dir': train_dir,
                'config': config,
                }



def localize_transporter_keypoints(model_kp,
                                   rgb=None,  # np.array [H, W, 3]
                                   mask=None,  # np.array [H, W]
                                   depth=None,  # np.array [H, W] in meters
                                   K=None,  # camera intrinsics matrix [3,3]
                                   T_world_camera=None,
                                   ):


    processed_image_dict = process_image(rgb=rgb,
                                         config=model_kp.config,
                                         mask=mask)

    rgb_input = processed_image_dict['image']
    crop_param = processed_image_dict['crop_param']

    H, W, _ = rgb.shape

    # cast them to torch so we can use them later
    crop_param_torch = None

    if crop_param is not None:
        crop_param_torch = dict()
        for key, val in crop_param.items():
            # cast to torch and add batch dim
            crop_param_torch[key] = torch.Tensor([val]).unsqueeze(0)

    rgb_to_tensor = ImageTupleDataset.make_rgb_image_to_tensor_transform()
    rgb_input_tensor = rgb_to_tensor(rgb_input).unsqueeze(0).cuda() # make it batch

    # [1, n_kp, 2]
    kp_pred = model_kp.predict_keypoint(rgb_input_tensor).cpu()

    # if it was cropped, then an extra step is needed
    # kp_pred_full_pixels = None
    # if crop_param is not None:
    #     kp_pred_full_pixels = map_cropped_pixels_to_full_pixels_torch(kp_pred,
    #                                                                  crop_param_torch)
    # else:
    #     kp_pred_full_pixels = kp_pred
    #
    # xy = kp_pred_full_pixels.clone()
    # xy[:, :, 0] = (xy[:, :, 0]) * 2.0 / W - 1.0
    # xy[:, :, 1] = (xy[:, :, 1]) * 2.0 / H - 1.0
    #
    # # get depth values
    # kp_pred_full_pixels_int = kp_pred_full_pixels.type(torch.LongTensor)

    full_image_coords = map_transporter_keypoints_to_full_image(kp_pred,
                                                                crop_params=crop_param_torch,
                                                                full_image_size=(H, W))

    uv = full_image_coords['uv']
    uv_int = full_image_coords['uv_int']
    xy = full_image_coords['xy']

    depth_batch = torch.from_numpy(depth).unsqueeze(0)


    z = None
    pts_world_frame = None
    pts_camera_frame = None
    if depth is not None:
        z = pdc_utils.index_into_batch_image_tensor(depth_batch.unsqueeze(1),
                                                    uv_int.transpose(1, 2))

        z = z.squeeze(1)

        K_inv = np.linalg.inv(K)
        K_inv_torch = torch.Tensor(K_inv).unsqueeze(0) # add batch dim
        pts_camera_frame = pdc_torch_utils.pinhole_unprojection(uv,
                                                                z,
                                                                K_inv_torch)

        # print("pts_camera_frame.shape", pts_camera_frame.shape)

        pts_world_frame = pdc_torch_utils.transform_points_3D(torch.from_numpy(T_world_camera),
                                                              pts_camera_frame)

        pts_world_frame_np = torch_utils.cast_to_numpy(pts_world_frame.squeeze())
        pts_camera_frame_np = torch_utils.cast_to_numpy(pts_camera_frame.squeeze())

    uv_np = torch_utils.cast_to_numpy(uv.squeeze())
    uv_int_np = torch_utils.cast_to_numpy(uv_int.squeeze())

    return {'uv': uv_np,
            'uv_int': uv_int_np,
            'xy': torch_utils.cast_to_numpy(xy.squeeze()),
            'z': torch_utils.cast_to_numpy(z.squeeze()),
            'pts_world_frame': pts_world_frame_np,
            'pts_camera_frame': pts_camera_frame_np,
            'pts_W': pts_world_frame_np, # just for backwards compatibility
            'kp_pred': kp_pred,
            'rgb_input': rgb_input,
            }
