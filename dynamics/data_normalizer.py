from __future__ import print_function

"""
Helper class to normalize the input for the network.
Contains only non-trainable parameters
"""
import torch
import torch.nn as nn


class DataNormalizer(nn.Module):

    def __init__(self, mean, std_dev):
        """
        The dimensions of mean and std_dev should be such that they can
        be broadcast with the data. Usually this means that they should
        have matching trailing dimensions
        :param mean:
        :type mean:
        :param std_dev:
        :type std_dev:
        """
        super(DataNormalizer, self).__init__()

        # copy data and create parameters
        self._mean = nn.Parameter(mean.clone().detach().requires_grad_(False))
        self._mean.requires_grad = False

        self._std_dev = nn.Parameter(std_dev.clone().detach().requires_grad_(False))
        self._std_dev.requires_grad = False

    def normalize(self, data):
        """
        Subtracts mean and divides by std_dev
        :param data: input data in batch B x data.shape
        :type data: tensor with same shape as data
        :return:
        :rtype:
        """

        # use broadcasting to compute mean and std_dev
        data_norm = (data - self._mean)/(self._std_dev)
        return data_norm

    def denormalize(self, data):
        """
        inverse operation of normalize
        :param data:
        :type data:
        :return:
        :rtype:
        """
        data_denorm = data*self._std_dev + self._mean
        return data_denorm

