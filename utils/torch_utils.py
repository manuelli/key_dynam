from torchvision import transforms
import torch
import numpy as np
import os
import torch.nn as nn

def make_default_image_to_tensor_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    return transforms.Compose([transforms.ToTensor(), normalize])


def random_sample_from_masked_image_torch(img_mask, num_samples):
    """
    :param img_mask: Numpy array [H,W] or torch.Tensor with shape [H,W]
    :type img_mask:
    :param num_samples: an integer
    :type num_samples:
    :return: torch.LongTensor shape [num_samples, 2] in u,v ordering
    :rtype:
    """

    nonzero_idx = torch.nonzero(img_mask, as_tuple=True)
    num_nonzero = nonzero_idx[0].numel()

    if num_nonzero < num_samples:
        raise ValueError("insufficient number of non-zero values to sample without replacement")

    sample_idx = torch.randperm(num_nonzero)[:num_samples]
    u_tensor = nonzero_idx[1][sample_idx]
    v_tensor = nonzero_idx[0][sample_idx]
    uv_tensor = torch.stack((u_tensor, v_tensor), dim=1)

    return uv_tensor


def get_model_device(model):
    return next(model.parameters()).device

def cast_to_numpy(x):
    """
    Cast to numpy.ndarray if it's a torch Tenso
    else return x itself
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    else:
        return x

def cast_to_torch(x):
    """
    Cast to torch Tensor if it's a numpy.ndarray,
    else return x itself
    """
    if isinstance(x, np.ndarray):
        return torch.Tensor(x)
    else:
        return x


def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)



def convert_torch_image_to_numpy(img, # [N, C, H, W] in range [0,1]
                                 ): # np.array [N, H, W, C] in range [0,255] np.uint8

    img = img.permute(0,2,3,1)
    img = cast_to_numpy(img) # clip to [0,1]
    img = np.clip(img, 0, 1)
    img = (cast_to_numpy(img)*255).astype(np.uint8)
    return img


def set_cuda_visible_devices(gpu_list):
    """
    Sets CUDA_VISIBLE_DEVICES environment variable to only show certain gpus
    If gpu_list is empty does nothing
    :param gpu_list: list of gpus to set as visible
    :return: None
    """

    if len(gpu_list) == 0:
        print("using all CUDA gpus")
        return

    cuda_visible_devices = ""
    for gpu in gpu_list:
        cuda_visible_devices += str(gpu) + ","

    print("setting CUDA_VISIBLE_DEVICES = ", cuda_visible_devices)
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

def convert_xy_to_uv_coordinates(xy, # tensor of shape [-1, 2]
                                 H=None,
                                 W=None):
    uv = (xy + 1)/2.0 * torch.Tensor([W, H]).to(xy.device).type_as(xy)
    return uv


class SpatialSoftmax(torch.nn.Module):
    def __init__(self, height, width, temperature=None, data_format='NCHW'):
        super(SpatialSoftmax, self).__init__()
        self.data_format = data_format
        self.height = height
        self.width = width


        if temperature:
            self.temperature = nn.Parameter(torch.ones(1)*temperature)
        else:
            self.temperature = 1.

        pos_x, pos_y = np.meshgrid(
                np.linspace(-1., 1., self.height),
                np.linspace(-1., 1., self.width)
                )
        pos_x = torch.from_numpy(pos_x.reshape(self.height*self.width)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self.height*self.width)).float()
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)

    def forward(self, feature):
        # Output:
        #   (N, C*2) x_0 y_0 ...

        assert feature.dim() == 4

        if self.data_format == 'NHWC':
            channel = feature.shape[-1]
            feature = feature.permute(0, 3, 1, 2)

        N, C, H, W = feature.shape
        assert self.height == H
        assert self.width == W

        # [N, C, H*W]
        feature = feature.view(N, C, self.height*self.width)

        # [N, C, H*W]
        softmax_attention = nn.functional.softmax(feature/self.temperature, dim=-1)


        # [N, C, 1]
        expected_x = torch.sum(self.pos_x*softmax_attention, dim=-1, keepdim=True)
        expected_y = torch.sum(self.pos_y*softmax_attention, dim=-1, keepdim=True)

        # [N, C, 2]
        expected_xy = torch.cat([expected_x, expected_y], dim=-1)

        return expected_xy

