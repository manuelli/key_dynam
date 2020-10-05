from __future__ import print_function

import numpy as np
from PIL import Image, ImageColor, ImageDraw

def convert_pygame_coordinates_to_PIL_coordinates(width, # image width, from config
                                                    height, # image height, from config,
                                                    xy_pygame, # np.array shape [2,], coordinates in pygame frame
                                                    ):
    """
    Note, PIL uses a "Cartesian pixel coordinate system" with (0,0) in the upper left
    hand corner. This means x goes across the screen left to right, and y goes down the
    screen top to bottom.

    The image on this (https://inventwithpython.com/invent4thed/chapter12.html) page
    in the "Coordinate System of a Computer Screen" section shows a picture.

    This is the "right-down" image coordinate convention, commonly referred to as
    (u,v)
    """

    xy_PIL = np.zeros(xy_pygame.shape)
    xy_PIL[0] = xy_pygame[0]
    xy_PIL[1] = height - xy_pygame[1]
    return xy_PIL

def get_circle_bounding_box_for_PIL_draw(center, # np.array shape [2,]
                                         radius, # float/int
                                         ):

   x0y0 = center - radius
   x1y1 = center + radius
   xy = [x0y0[0], x0y0[1], x1y1[0], x1y1[1]]
   return xy

def pusher_slider_keypoints_image(config, # global config
                                  pusher_position, # np.array shape [2,] expressed in pygame frame
                                  keypoint_positions, # np.array shape [num_keypoints, 2], expressed in pygame frame
                                  img=None, # optionally pass in an image to draw onto
                                  verbose=False
                                  ): # PIL Image

    """

    Draws pusher and keypoint positions onto a blank image
    """

    # raise ValueError("need to flip coordinate system . . . ")

    width = config['env']['display_size'][0]
    height = config['env']['display_size'][1]

    if img is None:
        size = (width, height)
        mode = 'RGB'
        color = ImageColor.getrgb("white")
        img = Image.new('RGB', size, color)

    draw = ImageDraw.Draw(img)

    # add red circle at pusher location
    pusher_radius = 5
    pusher_color = ImageColor.getrgb("red")
    pusher_position_uv = convert_pygame_coordinates_to_PIL_coordinates(width=width,
                                                                       height=height,
                                                                       xy_pygame=pusher_position)


    xy = get_circle_bounding_box_for_PIL_draw(pusher_position_uv, pusher_radius)
    if verbose:
        print("xy \n", xy)
    draw.ellipse(xy, fill=None, outline=pusher_color, width=2)


    # add keypoint locations
    keypoint_radius = 5
    keypoint_color = ImageColor.getrgb("green")
    num_keypoints = keypoint_positions.shape[0]
    for i in range(num_keypoints):
        keypoint_xy = keypoint_positions[i,:]
        keypoint_uv = convert_pygame_coordinates_to_PIL_coordinates(width=width,
                                                                    height=height,
                                                                    xy_pygame=keypoint_xy)
        xy = get_circle_bounding_box_for_PIL_draw(keypoint_uv, keypoint_radius)
        draw.ellipse(xy, fill=None, outline=keypoint_color, width=2)


    return img
