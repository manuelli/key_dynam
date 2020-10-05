import datetime
from PIL import Image
import os
import yaml
import json
import numpy as np
from future.utils import iteritems
import h5py
import copy
from six.moves import cPickle as pickle


def get_current_YYYY_MM_DD_hh_mm_ss_ms():
    """
    Returns a string identifying the current:
    - year, month, day, hour, minute, second

    Using this format:

    YYYY-MM-DD-hh-mm-ss

    For example:

    2018-04-07-19-02-50

    Note: this function will always return strings of the same length.

    :return: current time formatted as a string
    :rtype: string

    """

    now = datetime.datetime.now()
    string = "%0.4d-%0.2d-%0.2d-%0.2d-%0.2d-%0.2d-%0.6d" % (
    now.year, now.month, now.day, now.hour, now.minute, now.second, now.microsecond)
    return string


def numpy_to_PIL(img_np):
    """
    Converts numpy image of shape H x W x 3 to PIL image
    Assumes numpy image is in RGB ordering
    :param img_np:
    :type img_np:
    :return:
    :rtype:
    """

    return Image.fromarray(img_np, 'RGB')


def convert_float_image_to_uint8(img):
    img = np.clip(img, 0, 1)*255
    img = img.astype(np.uint8)
    return img


def get_project_root():
    """
    Returns the root of the project
    :return:
    :rtype:
    """
    import key_dynam
    return os.path.dirname(key_dynam.__file__)


def get_data_root():
    # return os.path.join(get_project_root(), 'data')
    return os.path.join(os.getenv("DATA_ROOT"), 'key_dynam')

def get_data_ssd_root():
    # return os.path.join(get_project_root(), 'data')
    return os.path.join(os.getenv("DATA_SSD_ROOT"), 'key_dynam')

def load_yaml(filename):
    # load YAML file
    return yaml.safe_load(open(filename, 'r'))


def load_json(filename):
    data = None
    with open(filename) as json_file:
        data = json.load(json_file)

    return data


def save_yaml(data, filename):
    with open(filename, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)


def load_pickle(filename, encoding='ascii'):
    file = open(filename, 'rb')

    # attempts to deal with Python 2/3 incompatibility issues
    data = pickle.load(file, fix_imports=True, encoding=encoding)
    file.close()
    return data


def save_pickle(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def get_image_diagonal_from_config(config,  # dict
                                   ):  # double
    camera_name = next(iter(config['env']['rgbd_sensors']['sensor_list']))
    W = config['env']['rgbd_sensors']['sensor_list'][camera_name]['width']
    H = config['env']['rgbd_sensors']['sensor_list'][camera_name]['height']

    return np.sqrt(W ** 2 + H ** 2)


def random_sample_in_range(low, high):
    alpha = np.random.random()
    return alpha * low + (1 - alpha) * high


def number_to_base(n,  # int
                   b,  # int
                   ):  # list[int]

    """
    Converts a number n into it's base b representation.

    Copied from https://stackoverflow.com/questions/2267362/how-to-convert-an-integer-to-a-string-in-any-base
    """

    if n == 0:
        return [0]
    digits = []
    while n:
        digits.append(int(n % b))
        n //= b
    return digits[::-1]


def save_dictionary_to_hdf5(hf,  # h5py file object
                            data,  # dict
                            key_tree,  # list of str
                            replace_value_with_key=False,
                            verbose=False):  # None

    """
    Recursively traverses `data`.
    If it encounters a np.ndarray it will save it to the h5py object.

    Optionally replace the value in `data` with the key that was used to store
    it in the h5py object.
    """

    for key, value in iteritems(data):
        new_key_tree = copy.deepcopy(key_tree)
        new_key_tree.append(key)

        if isinstance(value, np.ndarray):
            hf_key = "/".join(new_key_tree)
            if verbose:
                print("(SAVING): Found numpy array, saving:\n", hf_key)

            hf.create_dataset(hf_key, data=value)
            data[key] = hf_key
        elif isinstance(value, dict):
            if verbose:
                print("(RECURSING): Found dict array with key_tree:\n", key_tree)
            save_dictionary_to_hdf5(hf, value, new_key_tree, verbose=verbose)


def return_value_or_default(data,
                            key,
                            default,
                            ):
    try:
        return data[key]
    except KeyError:
        return default

def get_spartan_camera_info(camera_name=""):
    """
    Returns intrinsics + extrinsics
    :param camera_name:
    :type camera_name:
    :return:
    :rtype:
    """

    from key_dynam.utils.transform_utils import transform_from_pose_dict



    # intrinsics
    rgb_camera_info = load_yaml(os.path.join(get_project_root(), 'camera_config', camera_name, 'camera_info.yaml'))

    assert rgb_camera_info['image_width'] == 640
    assert rgb_camera_info['image_height'] == 480
    K = np.array(rgb_camera_info['camera_matrix']['data'])
    K = K.reshape([3,3])

    print("K\n", K)

    spartan_root_dir = os.path.join(os.path.dirname(get_project_root()), 'spartan')
    spartan_camera_data_dir = os.path.join(spartan_root_dir, "src/catkin_projects/camera_config/data", camera_name, 'master')
    extrinsics = load_yaml(os.path.join(spartan_camera_data_dir, 'rgb_extrinsics.yaml'))
    T_world_camera = transform_from_pose_dict(extrinsics['transform_to_reference_link'])

    print("T_world_camera\n", T_world_camera)

    return {'K': K,
            'T_world_camera': T_world_camera,
            'intrinsics_dict': rgb_camera_info,
            'extrinsics_dict': extrinsics}
