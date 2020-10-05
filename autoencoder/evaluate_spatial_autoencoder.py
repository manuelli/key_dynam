import os
import matplotlib.pyplot as plt
import numpy as np

import torch


from dense_correspondence_manipulation.utils.visualization import draw_reticles

from key_dynam.utils.utils import load_yaml, convert_float_image_to_uint8
from key_dynam.experiments.drake_pusher_slider import utils as exp_dps_utils
from key_dynam.dense_correspondence.dc_drake_sim_episode_reader import DCDrakeSimEpisodeReader
from key_dynam.autoencoder.autoencoder_models import SpatialAutoencoder
from key_dynam.autoencoder.dataset import AutoencoderImagePreprocessFunctionFactory, AutoencoderImageDataset
from key_dynam.utils import torch_utils


def load_model():

    # dataset_name
    # model_file
    sae_train_dir = "/home/manuelli/data/key_dynam/dev/experiments/drake_pusher_slider_v2/dataset_2020-04-20-14-58-21-418302_T_aug_random_velocity_1000/trained_models/perception/spatial_autoencoder"

    # model_name = "2020-06-05-20-57-10-394927"
    ckp_file = 'net_best.pth'
    model_name = "2020-06-06-01-57-53-187767" # lr 1e-3


    model_name = "2020-06-06-17-31-05-356659" # with masked loss
    ckp_file = "net_kp_epoch_38_iter_0.pth"


    dataset_name = "2020-04-20-14-58-21-418302_T_aug_random_velocity_1000"
    train_dir = os.path.join(sae_train_dir, model_name)
    dataset_paths = exp_dps_utils.get_dataset_paths(dataset_name)

    config = load_yaml(os.path.join(train_dir, 'config.yaml'))
    ckp_file = os.path.join(train_dir, 'checkpoints', ckp_file)

    camera_name = config['perception']['camera_name']

    model = SpatialAutoencoder.from_global_config(config)
    model.load_state_dict(torch.load(ckp_file))


    dataset_root = dataset_paths['dataset_root']
    dataset_name = dataset_paths['dataset_name']
    multi_episode_dict = DCDrakeSimEpisodeReader.load_dataset(dataset_root=dataset_root)
    image_preprocess_func = AutoencoderImagePreprocessFunctionFactory.spatial_autoencoder(config)

    dataset = AutoencoderImageDataset(config=config,
                                      episodes=multi_episode_dict,
                                      phase="train",
                                      camera_names=[camera_name],
                                      image_preprocess_func=image_preprocess_func,
                                      )



    return {'dataset_name': dataset_name,
            'dataset': dataset,
            'model': model}


def visualize_autoencoder_result():

    d = load_model()
    dataset = d['dataset']
    model = d['model']

    model = model.train()
    model = model.cuda()


    for i in range(10):
        idx = np.random.randint(0, len(dataset))
        data = dataset[idx]
        input = data['input_tensor'].unsqueeze(0).cuda()
        out = model(input)



        target_tensor = data['target_tensor'].squeeze()
        target_pred = out['output'].squeeze()
        keypoints_xy = out['expected_xy'].squeeze()



        print("keypoints_xy[0]", keypoints_xy[0])

        print("target_tensor.shape", target_tensor.shape)
        print("target_pred.shape", target_pred.shape)

        print("target_pred.dtype", target_pred.dtype)
        print("target_pred.max()", target_pred.max())
        print("target_pred.min()", target_pred.min())

        target_tensor_np = convert_float_image_to_uint8(torch_utils.cast_to_numpy(target_tensor))
        target_pred_np = convert_float_image_to_uint8(torch_utils.cast_to_numpy(target_pred))

        print(target_pred_np.shape)
        print(target_tensor_np.shape)


        # draw reticles on input image
        H, W, _ = data['input'].shape
        keypoints_uv = torch_utils.convert_xy_to_uv_coordinates(keypoints_xy, H=H, W=W)


        input_wr = np.copy(data['input'])
        draw_reticles(input_wr,
                                 u_vec=keypoints_uv[:, 0],
                                 v_vec=keypoints_uv[:, 1],
                                 label_color=[0,255,0])

        print("type(input_wr)", type(input_wr))

        figsize = 2*np.array([4.8,6.4])
        fig = plt.figure(figsize=figsize)
        ax = fig.subplots(1,3)
        ax[0].imshow(input_wr)
        ax[1].imshow(target_tensor_np, cmap='gray', vmin=0, vmax=255)
        ax[2].imshow(target_pred_np, cmap='gray', vmin=0, vmax=255)
        plt.show()


if __name__ == "__main__":

    visualize_autoencoder_result()


