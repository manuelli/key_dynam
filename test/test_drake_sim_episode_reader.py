import os


from key_dynam.utils.utils import load_pickle
from key_dynam.dataset.drake_sim_episode_reader import DrakeSimEpisodeReader

data_file = "/home/manuelli/data/key_dynam/dev/experiments/18/data/dps_box_on_side_600/2020-05-13-21-53-45-823302_idx_0.p"

non_image_data = load_pickle(data_file)
episode = DrakeSimEpisodeReader(non_image_data)

data = episode.get_data(0)

print("data.keys()", data.keys())
print("data['observation']", data['observation'].keys())