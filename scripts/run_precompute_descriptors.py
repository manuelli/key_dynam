import os
import torch

from key_dynam.dense_correspondence.image_dataset import ImageDataset
from key_dynam.utils.utils import get_project_root, load_yaml
from key_dynam.dataset.drake_sim_episode_reader import DrakeSimEpisodeReader
from key_dynam.dense_correspondence.precompute_descriptors import precompute_descriptors

from dense_correspondence_manipulation.utils.utils import set_cuda_visible_devices

CUDA_VISIBLE_DEVICES = [1]
set_cuda_visible_devices(CUDA_VISIBLE_DEVICES)

config = dict()
config['train'] = dict()
config['train']['train_valid_ratio'] = 0.9

# DATASET_NAME = "top_down_rotated"
DATASET_NAME = "2019-12-05-15-58-48-462834_top_down_rotated_250"

# MODEL_NAME = "2019-12-04-01-32-12-010393_top_down_rotated_sigma_5"
# MODEL_NAME = "2019-12-26-20-49-35-944397_dataset_2019-12-05-15-58-48-462834_top_down_rotated_250"
MODEL_NAME = "2019-12-27-22-29-23-359200_dataset_2019-12-05-15-58-48-462834_top_down_rotated_250"


dataset_root = os.path.join(get_project_root(), "data/dev/experiments/05/data", DATASET_NAME)
multi_episode_dict = DrakeSimEpisodeReader.load_dataset(dataset_root)


network_folder = os.path.join(get_project_root(), "data/dev/experiments/05/trained_models", MODEL_NAME)

epoch = 55
iter = 0
model_file = os.path.join(network_folder, "net_dy_epoch_%d_iter_%d_model.pth" %(epoch, iter))
model = torch.load(model_file)
model.cuda()
model = model.eval()

output_dir = os.path.join(get_project_root(), "data/dev/experiments/05/precomputed_descriptors", DATASET_NAME, MODEL_NAME)

precompute_descriptors(multi_episode_dict, model, output_dir, model_file=model_file)

