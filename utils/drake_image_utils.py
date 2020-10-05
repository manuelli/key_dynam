import pydrake
import PIL
from PIL import Image
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# pydrake
import pydrake
import pydrake.systems.sensors
from pydrake.geometry.render import (
    DepthCameraProperties,
    RenderLabel,
    MakeRenderEngineVtk,
    RenderEngineVtkParams,
)

RESERVED_LABELS = [
    RenderLabel.kDoNotRender, RenderLabel.kDontCare, RenderLabel.kEmpty, RenderLabel.kUnspecified]


def get_color_image(sensor: pydrake.systems.sensors.RgbdSensor,
                    context: pydrake.systems.framework.Context,
                    )-> np.array:

    # pydrake.systems.sensors.Image[PixelType.kRgba8U]
    # uint8 format
    # https://github.com/RobotLocomotion/drake/blob/master/systems/sensors/pixel_types.h
    rgb_img_drake = sensor.color_image_output_port().Eval(context)

    # numpy array [W, H, 4] with rgba encoding
    # dtype is uint8. Cut off the a channel
    rgb_img_np = np.copy(rgb_img_drake.data[:, :, :3])
    # rgb_img_PIL = Image.fromarray(rgb_img_np, 'RGB')
    return rgb_img_np


def get_depth_image_32F(sensor: pydrake.systems.sensors.RgbdSensor,
                        context: pydrake.systems.framework.Context,
                        )-> np.array: # shape [W,H,1], dtype=float

    # pydrake.systems.sensors.Image[PixelType.kDepth32F]
    # https://github.com/RobotLocomotion/drake/blob/master/systems/sensors/pixel_types.h
    depth_img_drake = sensor.depth_image_32F_output_port().Eval(context)

    # np.array
    depth_img_np = np.copy(depth_img_drake.data)
    return depth_img_np.squeeze()

def get_depth_image_16U(sensor: pydrake.systems.sensors.RgbdSensor,
                        context: pydrake.systems.framework.Context,
                        )-> np.array: # shape [W,H,1], dtype=uint16

    # pydrake.systems.sensors.Image[PixelType.kDepth16U]
    depth_img_drake = sensor.depth_image_16U_output_port().Eval(context)
    depth_img_np = np.copy(depth_img_drake.data)
    return depth_img_np.squeeze()

def get_label_image(sensor: pydrake.systems.sensors.RgbdSensor,
                    context: pydrake.systems.framework.Context,
                    )-> np.array: # shape [W,H,1], dtype=uint16

    # pydrake.systems.sensors.Image[PixelType.ImageLabel16I]
    label_img_drake = sensor.label_image_output_port().Eval(context)
    label_img_np = np.copy(label_img_drake.data)
    return label_img_np.squeeze()


def remove_depth_16U_out_of_range_and_cast(depth_raw, # np.array [H, W], dtype=np.uint16
                                           dtype, # dtype you wish to cast to, must be integer type
                                           ):
    """
    Converts a depth image that's expressed in uint16 coming from a
    Drake RGBDSensor to the standard int16 format with max range values set to zero

    See https://drake.mit.edu/doxygen_cxx/classdrake_1_1systems_1_1sensors_1_1_rgbd_sensor.html
    for documentation on Drake's RGBD sensor

    np.iinfo(dtype).max values indicate beyond max sensor range
    np.iinfo(dtype).min mean below min sensor range.

    For simplicity we set them both to zero. In our (simplified) setting
    we just regard all zero depth values as invalid, not making a distinction
    between those that are beyond the max range and those that are less than the
    min range
    """
    assert depth_raw.dtype == np.uint16, "Expected depth_raw.dtype = np.uint16, received %s" %(depth_raw.dtype)
    depth = np.copy(depth_raw)
    max_val = np.iinfo(np.uint16).max # max value of uint16
    depth[depth == max_val] = 0
    if np.max(depth) > np.iinfo(dtype).max:
        raise OverflowError("np.max(depth)=%.1f > np.iinfo(dtype).max=%.1f" % (np.max(depth), np.iinfo(dtype).max))

    return depth.astype(dtype)

def remove_depth_32F_out_of_range_and_cast(depth_raw, # np.array [H, W], dtype=np.uint16 or np.float32 typically.
                                           dtype, # dtype you wish to cast to, must be float type
                                           ):
    """
    Converts a depth image that's expressed in float32 coming from a
    Drake RGBDSensor to the standard float32 but with max range values set
    to zero.

    See https://drake.mit.edu/doxygen_cxx/classdrake_1_1systems_1_1sensors_1_1_rgbd_sensor.html
    for documentation on Drake's RGBD sensor

    np.iinfo(dtype).max values indicate beyond max sensor range
    np.iinfo(dtype).min mean below min sensor range.

    For simplicity we set them both to zero. In our (simplified) setting
    we just regard all zero depth values as invalid, not making a distinction
    between those that are beyond the max range and those that are less than the
    min range
    """
    depth = np.copy(depth_raw)
    depth[depth == np.inf] = 0
    if np.max(depth) > np.finfo(dtype).max:
        raise OverflowError("np.max(depth)=%.1f > np.iinfo(dtype).max=%.1f" % (np.max(depth), np.iinfo(dtype).max))

    return depth.astype(dtype)


def remove_out_of_range_and_cast(depth_raw,  # np.array [H, W], dtype=np.uint16 or np.float32 typically.
                                 out_of_range_val,
                                 dtype,  # the dtype that you wish to cast to
                                 ):


    """
    Converts a depth image that's expressed in uint16 coming from a
    Drake RGBDSensor to the standard int16 format.

    See https://drake.mit.edu/doxygen_cxx/classdrake_1_1systems_1_1sensors_1_1_rgbd_sensor.html
    for documentation on Drake's RGBD sensor

    np.iinfo(dtype).max values indicate beyond max sensor range
    np.iinfo(dtype).min mean below min sensor range.

    For simplicity we set them both to zero. In our (simplified) setting
    we just regard all zero depth values as invalid, not making a distinction
    between those that are beyond the max range and those that are less than the
    min range
    """

    depth = np.copy(depth_raw)
    max_val = np.iinfo(depth_raw.dtype)
    depth[depth_raw == np.iinfo(depth_raw.dtyp).max] = 0
    if np.max(depth) > np.iinfo(dtype).max:
        raise OverflowError("np.max(depth)=%.1f > np.iinfo(dtype).max=%.1f" %(np.max(depth), np.iinfo(dtype).max))

    return depth.astype(dtype)


def colorize_labels(image):
    """

    Colorizes labels.

    Copied from https://nbviewer.jupyter.org/github/EricCousineau-TRI/drake/blob/feature-2019-11-21-label-example-minimal/tutorials/rendering_multibody_plant.ipynb

    """
    # TODO(eric.cousineau): Revive and use Kuni's palette.
    cc = mpl.colors.ColorConverter()
    color_cycle = plt.rcParams["axes.prop_cycle"]
    colors = np.array([cc.to_rgb(c["color"]) for c in color_cycle])
    bg_color = [0, 0, 0]
    image = np.squeeze(image)
    background = np.zeros(image.shape[:2], dtype=bool)
    # for label in RESERVED_LABELS:
    #     background |= image == int(label)
    color_image = colors[image % len(colors)]
    color_image[background] = bg_color
    return color_image