from __future__ import print_function

# system
import os
import time
import glob
import cv2

# key_dynam
from key_dynam.dataset.episode_dataset import MultiEpisodeDataset
from key_dynam.dataset.episode_reader import PyMunkEpisodeReader
from key_dynam.dataset.drake_sim_episode_reader import DrakeSimEpisodeReader
from key_dynam.dataset.function_factory import ObservationFunctionFactory, ActionFunctionFactory
from key_dynam.utils.utils import get_project_root, load_pickle, get_data_root


def load_episodes_from_config(config):
    """
    Loads episodes using the path specified in the config
    :param config:
    :type config:
    :return:
    :rtype:
    """
    data_path = config["dataset"]["data_path"]
    if not os.path.isabs(data_path):
        data_path = os.path.join(get_project_root(), data_path)

    # load the data
    print("loading data from disk . . . ")
    raw_data = load_pickle(data_path)
    print("finished loading data")
    episodes = PyMunkEpisodeReader.load_pymunk_episodes_from_raw_data(raw_data)

    return episodes


def load_drake_sim_episodes_from_config(config,
                                        load_descriptor_images=True,
                                        load_descriptor_keypoints=True):
    dataset_root = config["dataset"]["dataset_dir"]
    if not os.path.isabs(dataset_root):
        dataset_root = os.path.join(get_data_root(), dataset_root)

    descriptor_images_root = None
    if 'descriptor_images_dir' in config['dataset'] and load_descriptor_images:
        descriptor_images_root = os.path.join(get_data_root(), config['dataset']['descriptor_images_dir'])

    descriptor_keypoints_root = None
    if 'descriptor_keypoints_dir' in config['dataset'] and load_descriptor_keypoints:
        descriptor_keypoints_root = os.path.join(get_data_root(), config['dataset']['descriptor_keypoints_dir'])

    max_num_episodes = None
    if 'max_num_episodes' in config['dataset']:
        max_num_episodes = config['dataset']['max_num_episodes']

    multi_episode_dict = \
        DrakeSimEpisodeReader.load_dataset(dataset_root,
                                           descriptor_images_root=descriptor_images_root,
                                           descriptor_keypoints_root=descriptor_keypoints_root,
                                           max_num_episodes=max_num_episodes)

    return multi_episode_dict


def construct_dataset_from_config(config,  # dict for global config
                                  phase="train",  # str: either "train" or "valid"
                                  episodes=None,  # optional dict of episodes to use instead of loading data
                                  ):
    """
    Construct dataset based on global config
    :param config:
    :type config:
    :return:
    :rtype:
    """

    assert phase in ["train", "valid"]

    data_path = config["dataset"]["data_path"]
    if not os.path.isabs(data_path):
        data_path = os.path.join(get_project_root(), data_path)

    if episodes is None:
        # load the data if not passed in
        episodes = load_episodes_from_config(config)

    action_function = ActionFunctionFactory.function_from_config(config)
    observation_function = ObservationFunctionFactory.function_from_config(config)

    dataset = MultiEpisodeDataset(config,
                                  action_function=action_function,
                                  observation_function=observation_function,
                                  episodes=episodes,
                                  phase=phase)

    return_data = {
        "dataset": dataset,
        "action_function": action_function,
        "observation_function": observation_function,
        "episodes": episodes,
    }

    return return_data


def drake_sim_dataset_loader(precomputed_vision_data_dir,  # precomputed_vision_data_dir
                             dataset_root,  # location of original dataset
                             max_num_episodes=None,
                             ):
    spatial_descriptor_data = load_pickle(os.path.join(precomputed_vision_data_dir, 'spatial_descriptors.p'))
    metadata = load_pickle(os.path.join(precomputed_vision_data_dir, 'metadata.p'))
    descriptor_keypoints_root = os.path.join(precomputed_vision_data_dir, 'descriptor_keypoints')

    multi_episode_dict = DrakeSimEpisodeReader.load_dataset(dataset_root,
                                                            max_num_episodes=max_num_episodes,
                                                            descriptor_keypoints_root=descriptor_keypoints_root)

    return {'spatial_descriptors_data': spatial_descriptor_data,
            'metadata': metadata,
            'multi_episode_dict': multi_episode_dict,
            }


def make_video(image_folder,
               fps=5,
               video_file=None,):
    if video_file is None:
        video_file = os.path.join(image_folder, "video.mp4")



    images = [img for img in os.listdir(image_folder) if img.endswith("rgb.png")]

    images = glob.glob("%s/*rgb.png" %(image_folder))
    images.sort()
    # print("images", images)

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    video = cv2.VideoWriter(video_file, fourcc, fps, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()

def concatenate_videos(video_files, # list[str] absolute paths
                       save_file):

    # following https://stackoverflow.com/questions/36967947/how-to-merge-two-videos
    # filename = "/tmp/video_file_list.txt"
    filename = "video_file_list.txt"
    if os.path.exists(filename):
        os.remove(filename)


    with open(filename, 'w') as f:
        for video_file in video_files:
            line = "file '%s'\n" %(video_file)
            f.write(line)


    cmd = "ffmpeg -f concat -safe 0 -i %s -c copy %s" %(filename, save_file)
    print("cmd", cmd)
    os.system(cmd)

