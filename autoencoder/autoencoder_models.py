import copy
from collections import OrderedDict

import torch
import torch.nn as nn
import torchvision

# cfm
from cfm.models import Encoder, Decoder
from key_dynam.utils.torch_utils import SpatialSoftmax



class AutoEncoder(nn.Module):

    def __init__(self):
        super(AutoEncoder, self).__init__()
        pass

    def encode(self, x):
        # should return a dict
        # {'z': the latent state [B, z_dim]}
        raise NotImplementedError()

    def decode(self, z):
        # should return dict, at minimum has the reconstructed image
        # {'output': [B, -1]}

        raise NotImplementedError()


    def forward(self, x):
        """
        Do full encode and decode step
        :param x:
        :type x:
        :return:
        :rtype:
        """
        encode_output = self.encode(x)
        z = encode_output['z']
        decode_output = self.decode(z)
        encode_output.update(decode_output)
        return encode_output


# implementation of spatial autoencder from this paper
# https://arxiv.org/abs/1509.06113
class SpatialAutoencoder(AutoEncoder):

    def __init__(self, H, W, imagenet_pretrained=True):
        super(AutoEncoder, self).__init__()

        self.H = H
        self.W = W
        assert self.H == 240 # only support this size

        self._input_image_shape = [self.H, self.W]


        if self.W == 240:
            self.final_width = 109
            self.softmax_image_shape = [109, 109]
            self.decode_linear = nn.Linear(32, 60 * 60)
            self._output_image_shape = [60, 60]
        if self.W == 320:
            self.final_width = 149
            self.softmax_image_shape = [109, 149]
            self.decode_linear = nn.Linear(32, 80 * 60)
            self._output_image_shape = [60, 80]



        # seq = [nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2),
        #        nn.BatchNorm2d(64),
        #        nn.ReLU(inplace=True),
        #        nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, stride=1),
        #        nn.BatchNorm2d(32),
        #        nn.ReLU(inplace=True),
        #        nn.Conv2d(in_channels=32, out_channels=16, kernel_size=5, stride=1),
        #        nn.BatchNorm2d(16),
        #        nn.ReLU(inplace=True),
        #        ]

        seq = OrderedDict([
            ('conv1', nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2)),
            ('batch_norm1', nn.BatchNorm2d(64)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, stride=1)),
            ('batch_norm2', nn.BatchNorm2d(32)),
            ('relu2', nn.ReLU(inplace=True)),
            ('conv3', nn.Conv2d(in_channels=32, out_channels=16, kernel_size=5, stride=1)),
            ('batch_norm3', nn.BatchNorm2d(16)),
            ('relu3', nn.ReLU(inplace=True)),
               ])

        self.encoder = nn.Sequential(seq)


        if imagenet_pretrained:
            # set first layer weights to those from ImageNet pretrained model
            googlenet = torchvision.models.googlenet(pretrained=True)
            update_state_dict = {'encoder.conv1.weight': googlenet.state_dict()['conv1.conv.weight']}
            self.state_dict().update(update_state_dict)


        self.decode_linear = nn.Linear(32, self._output_image_shape[0] * self._output_image_shape[1])
        self.decoder = nn.Sequential(self.decode_linear)
        self.spatial_softmax = SpatialSoftmax(self.softmax_image_shape[0], self.softmax_image_shape[1])

    @property
    def input_image_shape(self):
        """
        Shape of the image that the network expects to receive as input
        :return:
        :rtype:
        """
        return copy.deepcopy(self._input_image_shape)


    @property
    def output_image_shape(self):
        """
        Shape of reconstructed image that the network outputs
        :return:
        :rtype:
        """
        return copy.deepcopy(self._output_image_shape)


    def encode(self,
               x, # B x 3 x H x W image
               ): # dict

        # B x self.softmax_image_shape
        feature = self.encoder(x)


        # [B, num_keypoints, 2]
        expected_xy = self.spatial_softmax(feature)


        # [B, num_keypoints * 2]
        z = expected_xy.flatten(start_dim=1)

        return {'z': z, # B x 32
                'expected_xy': expected_xy, # B x 16 x 2
                }


    def decode(self,
               z):

        B = z.shape[0]
        y = self.decoder(z)
        img = y.reshape([B, 1, self._output_image_shape[0], self._output_image_shape[1]])

        return {'output': img} # [B, 1, output_shape[0], output_shape[1]]

    @staticmethod
    def from_global_config(config):
        H = config['perception']['height']
        W = config['perception']['width']

        return SpatialAutoencoder(H=H, W=W, imagenet_pretrained=config['perception']['imagenet_pretrained'])



class ConvolutionalAutoencoder(AutoEncoder):

    def __init__(self, z_dim, channel_dim):
        super().__init__()

        self.encoder = Encoder(z_dim, channel_dim=channel_dim)
        self.decoder = Decoder(z_dim, channel_dim=channel_dim)

    def encode(self,
               x, # [B, 3, 64, 64]
               ):

        z = self.encoder(x)
        return {'z': z}

    def decode(self,
               z, # [B, z_dim]
               ):

        y = self.decoder(z)
        return {'output': y,
                'recon': y}


    @staticmethod
    def from_global_config(config):
        return ConvolutionalAutoencoder(z_dim=config['perception']['z_dim'],
                                        channel_dim=config['perception']['channel_dim'])

# # implementation of 'autoencoder' baseline from this paper
# # https://arxiv.org/abs/2003.05436
# class ConvolutionalAutoencoder(AutoEncoder):
#
#     def __init__(self):
#         super(ConvolutionalAutoencoder, self).__init__()
#
#
#         kernel_sizes = [3,4,3,4,4,4]
#         strides = [1,2,1,2,2,2]
#         filter_sizes = [64,64,64,128,256,256]
#
#
#
#         seq = [('conv1', nn.Conv2d(in_channels=3, out_channels=filter_sizes[0], kernel_size=kernel_sizes[0], stride=strides[0])),
#             ('relu1', nn.LeakyReLU(0.2, inplace=True))]
#
#         for i in range(1, len(kernel_sizes)):
#             in_channels = filter_sizes[i-1]
#             out_channels = filter_sizes[i]
#             kernel_size = kernel_sizes[i]
#             stride = strides[i]
#
#             conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
#                              kernel_size=kernel_size, stride=stride)
#
#
#             seq.append(('conv%s' %(i+1), conv))
#             seq.append(('relu%s' % (i + 1), nn.LeakyReLU(0.2, inplace=True)))
#
#
#
#         odict = OrderedDict(seq)
#         self.encoder_backbone = nn.Sequential(odict)
#         self.encoder_fc = nn.Linear()
#
#     def encode(self,
#                x, # [B, 3, 64, 64]
#                ):
#         z = self.encoder(x) # [B, 256, 1, 1]
#
#         return {'z': z}



