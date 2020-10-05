import torch
import torch.nn as nn

# pdc
from dense_correspondence.network.predict import get_argmax_l2
import dense_correspondence_manipulation.utils.utils as pdc_utils

# key_dynam
from key_dynam.utils.torch_utils import random_sample_from_masked_image_torch, get_model_device
# pdc
from dense_correspondence.network.predict import get_spatial_expectation, get_integral_preds_3d


def sample_descriptors(descriptor_img,  # torch.Tensor [D, H, W]
                       img_mask,  # binary mask [H, W]
                       num_samples,  # int
                       ): # return: dict
    """
    Samples reference descriptors from a mask
    """
    # [num_samples, 2]
    uv_tensor = random_sample_from_masked_image_torch(img_mask, num_samples)

    # [1, 2, num_samples]
    uv_batch = uv_tensor.permute([1, 0]).unsqueeze(0)

    # [num_samples, D]
    des = pdc_utils.index_into_batch_image_tensor(descriptor_img.unsqueeze(0), uv_batch).squeeze().permute([1, 0])


    return {'indices': uv_tensor, # [N, 2]
            'descriptors': des, # [N, D]
            }


class PrecomputedDescriptorNet(nn.Module):
    """
    Localizes reference descriptors given a descriptor image
    Note that this doesn't contain any trainable parameters.
    Currently this is deprecated since it is using the argmax L2
    correspondence strategy instead of the spatial expectation . . .
    """

    def __init__(self,
                 config,
                 ref_descriptors=None,
                 ):
        super(PrecomputedDescriptorNet, self).__init__()
        K = config['vision_net']['num_ref_descriptors']
        D = config['vision_net']['descriptor_dim']
        self._camera_name = config['vision_net']['camera_name']
        self._config = config

        # [K, D]
        # currently we don't support gradients
        self._ref_descriptors = torch.nn.Parameter(torch.zeros([K, D]))
        self._ref_descriptors.requires_grad = False

    @property
    def num_ref_descriptors(self):
        return self._ref_descriptors.shape[0]

    @property
    def descriptor_dim(self):
        return self._ref_descriptors.shape[1]

    @property
    def device(self):
        """
        Get the device that this model is on
        :return:
        :rtype:
        """

        return self._ref_descriptors.device


    def set_reference_descriptors(self,
                                  ref_descriptors):

        K, D = ref_descriptors.shape
        assert K == self.num_ref_descriptors, "K: %d, num_ref_descriptors: %d" %(K, self.num_ref_descriptors)
        assert D == self.descriptor_dim, "D: %d, descriptor_dim: %d" %(D, self.descriptor_dim)

        self._ref_descriptors.data = ref_descriptors



    def initialize_weights(self):
        """
        Initializes the weights of this network
        :return:
        :rtype:
        """
        torch.nn.init.normal_(self._ref_descriptors)

    def forward_descriptor_image(self,
                                 descriptor_image,  # [B, D, H, W]
                                 ):

        raise ValueError("Using deprecated argmax l2 keypoint localization")

        B, D, H, W = descriptor_image.shape
        assert D == self.descriptor_dim

        descriptor_image = descriptor_image.to(self.device)

        # localize descriptors
        best_match_dict = get_argmax_l2(self._ref_descriptors, descriptor_image)

        # K is num descriptors
        # [B, K, 2] in (u,v) ordering
        dynamics_net_input = best_match_dict['indices']

        # flatten last two dimensions
        # [B, 2*K]
        dynamics_net_input = dynamics_net_input.flatten(start_dim=1)

        out = {'best_match_dict': best_match_dict,
               'dynamics_net_input': dynamics_net_input}

        return out

    def forward(self,
                visual_observation,  # dict storing tensors of shape [B, D, H, W]
                ):
        descriptor_image = visual_observation[self._camera_name]['descriptor']
        return self.forward_descriptor_image(descriptor_image)



class DescriptorKeypointNet(nn.Module):
    """
    Container for both a DenseDescriptor network and a
    network that computes model keypoints . . .
    """

    def __init__(self,
                 config,
                 model_dd, # DenseDescriptorModel
                 model_keypoints, # PrecomputeDescriptorNet
                 ):
        super(DescriptorKeypointNet, self).__init__()
        self._config = config
        self._model_dd = model_dd
        self._model_keypoints = model_keypoints
        self._camera_name = config['vision_net']['camera_name']

    def forward_visual_obs(self,
                           visual_observation, # dict
                           ):
        

        # [3, H, W] or [B, 3, H, W]
        device = get_model_device(self)
        rgb_tensor = visual_observation[self._camera_name]['rgb_tensor'].to(device)

        # unsqueeze if necessary
        if len(rgb_tensor.shape) == 3:
            # unsqueeze it if necessary
            rgb_tensor = rgb_tensor.unsqueeze(0)

        dd_out = self._model_dd.forward(rgb_tensor)

        des_img = dd_out['descriptor_image']
        keypoints_out = self._model_keypoints.forward_descriptor_image(des_img)

        out = {'dd_out': dd_out,
               'keypoints_out': keypoints_out,
               'dynamics_net_input': keypoints_out['dynamics_net_input']}

        return out