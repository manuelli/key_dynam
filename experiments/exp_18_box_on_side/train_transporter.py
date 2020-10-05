import os
import pydrake
import time



from dense_correspondence_manipulation.utils.utils import set_cuda_visible_devices

set_cuda_visible_devices([1])


from key_dynam.utils.utils import get_project_root, save_yaml, load_yaml, get_current_YYYY_MM_DD_hh_mm_ss_ms, get_data_root
from key_dynam.transporter.train_transporter import train_transporter
from key_dynam.dense_correspondence.dc_drake_sim_episode_reader import DCDrakeSimEpisodeReader


# project root
PROJECT_ROOT = get_project_root()

# load config
config = load_yaml(os.path.join(get_project_root(), "experiments/exp_18_box_on_side/config.yaml"))

# specify the dataset
dataset_name = "dps_box_on_side_600"
config['perception']['dataset_name'] = dataset_name
dataset_name = config['perception']['dataset_name']

config['dataset']['set_epoch_size_to_num_images'] = False
config['dataset']['epoch_size'] = {'train': 6000, 'valid': 300}


# camera_name = "camera_angled"
camera_name = "camera_1_top_down"

config['perception']['camera_name'] = camera_name

config['train'] = config['train_transporter']
config['train']['train_valid_ratio'] = 0.83

print(config)

# prepare folders
data_dir = os.path.join(get_data_root(), 'dev/experiments/18/data', dataset_name)

train_dir = os.path.join(get_data_root(), 'dev/experiments/18/trained_models/perception', "transporter_%s_%s" %(camera_name, get_current_YYYY_MM_DD_hh_mm_ss_ms()))
ckp_dir = os.path.join(train_dir, 'train_nKp%d_invStd%.1f' % (
    config['perception']['n_kp'], config['perception']['inv_std']))


start_time = time.time()

multi_episode_dict = DCDrakeSimEpisodeReader.load_dataset(dataset_root=data_dir)
print("loading dataset took %d seconds", time.time() - start_time)

train_transporter(config=config,
                  train_dir=train_dir,
                  ckp_dir=ckp_dir,
                  multi_episode_dict=multi_episode_dict)