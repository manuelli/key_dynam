import os
from key_dynam.dataset.utils import make_video

# image_folder = "/home/manuelli/data/key_dynam/sandbox/2020-06-29-21-04-16/processed/images_camera_0"
# make_video(image_folder)

def make_videos_for_all_episodes():
    episodes_root = "/home/manuelli/data_ssd/key_dynam/dataset/push_box_hardware"
    episode_names = os.listdir(episodes_root)
    episode_names.sort()

    for episode_name in episode_names:
        print("processing episode", episode_name)
        image_folder = os.path.join(episodes_root, episode_name, 'processed/images_camera_0')
        make_video(image_folder)

        image_folder = os.path.join(episodes_root, episode_name, 'processed/images_camera_1')
        make_video(image_folder)



if __name__ == "__main__":
    make_videos_for_all_episodes()