import h5py
import numpy as np
import os

from key_dynam.dataset.episode_container import EpisodeContainer

W = 640
H = 480
rgb_img = np.zeros([H, W], np.int16)
depth_img = np.zeros([H,W], dtype=np.float32)

filename = 'data.h5'
if os.path.isfile(filename):
    os.remove(filename)

hf = h5py.File(filename, 'w')


# hf.create_dataset("rgb_img", data=rgb_img)
# hf.create_dataset("depth_img", data=depth_img)
# hf.close()
#
# # read the file
# hf_read = h5py.File(filename, 'r')
# print(hf_read.keys())
# rgb_read = hf_read.get("rgb_img")
# print(rgb_read.dtype)
#
# depth_read = hf_read.get("depth_img")
# print(depth_read.dtype)

tmp = ['camera_0', 'rgb']
s = "/".join(tmp)
print("s", s)


data = dict()
data['camera_0'] = dict()
data['camera_0']['rgb'] = rgb_img
data['camera_0']['depth'] = depth_img
key_tree = []
EpisodeContainer.recursive_image_save(hf, data, key_tree, verbose=True)
hf.close()

# read the file
hf_read = h5py.File(filename, 'r')
print(hf_read.keys())


rgb_read = hf_read.get("camera_0/rgb")
print(rgb_read.dtype)

depth_read = hf_read.get("camera_0/depth")
print(depth_read.dtype)
