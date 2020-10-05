import os
import h5py
import time
import numpy as np
import copy

import torch
from torch.utils.data import DataLoader

# dense correspondence
from dense_correspondence_manipulation.utils.constants import DEPTH_IM_SCALE
from dense_correspondence_manipulation.utils import utils as pdc_utils
from dense_correspondence_manipulation.utils import torch_utils as pdc_torch_utils

# key_dynam
from key_dynam.transporter.dataset import ImageTupleDataset
import key_dynam.transporter.utils as transporter_utils
from key_dynam.utils import torch_utils
from key_dynam.utils.utils import save_pickle, save_dictionary_to_hdf5, save_yaml

"""
Utilities for precomputing keypoints that come from the transporter model
Try 
"""



def project_keypoints_3d(uv, # [n_kp, 2] or [B, n_kp,2]
                         depth,
                         T_W_C,
                         K,
                         ):


    uv_batch = None
    depth_batch = None
    has_batch_dim = False
    if uv.dim() == 2:
        # [H, W]
        assert depth.dim() == 2
        uv_batch = uv.unsqueeze(0)
        depth_batch = depth.unsqueeze(0)
    else:
        # [B, H, W]
        assert depth.dim() == 3
        uv_batch = uv
        depth_batch = depth
        has_batch_dim = True


    # [B, n_kp, 1]
    depth_uv = pdc_utils.index_into_batch_image_tensor(depth.unsqueeze(1), uv.permute(0,2,1))

    print("depth_uv.shape")



    return {'uv': uv,
            'uv_int': uv.astype(torch.int16)}


def precompute_transporter_keypoints(multi_episode_dict,
                                     model_kp,
                                     output_dir,  # str
                                     batch_size=10,
                                     num_workers=10,
                                     camera_names=None,
                                     model_file=None,
                                     ):

    assert model_file is not None
    metadata = dict()
    metadata['model_file'] = model_file

    save_yaml(metadata, os.path.join(output_dir, 'metadata.yaml'))
    start_time = time.time()

    log_freq = 10

    device = next(model_kp.parameters()).device
    model_kp = model_kp.eval()  # make sure model is in eval mode

    image_data_config = {'rgb': True,
                         'mask': True,
                         'depth_int16': True,
                         }

    # build all the dataset
    datasets = {}
    dataloaders = {}
    for episode_name, episode in multi_episode_dict.items():
        single_episode_dict = {episode_name: episode}
        config = model_kp.config

        # need to do this since transporter type data sampling only works
        # with tuple_size = 1
        dataset_config = copy.deepcopy(config)
        dataset_config['dataset']['use_transporter_type_data_sampling'] = False

        datasets[episode_name] = ImageTupleDataset(dataset_config,
                                                   single_episode_dict,
                                                   phase="all",
                                                   image_data_config=image_data_config,
                                                   tuple_size=1,
                                                   compute_K_inv=True,
                                                   camera_names=camera_names)

        dataloaders[episode_name] = DataLoader(datasets[episode_name],
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               shuffle=False)

    episode_counter = 0
    num_episodes = len(multi_episode_dict)


    for episode_name, dataset in datasets.items():
        episode_counter += 1
        print("\n\n")

        episode = multi_episode_dict[episode_name]
        hdf5_file = None
        try:
            hdf5_file = os.path.basename(episode.image_data_file)
        except AttributeError:
            hdf5_file = "%s.h5" % (episode.name)

        hdf5_file_fullpath = os.path.join(output_dir, hdf5_file)

        str_split = hdf5_file_fullpath.split(".")
        assert len(str_split) == 2
        pickle_file_fullpath = str_split[0] + ".p"


        # print("episode_name", episode_name)
        # print("hdf5_file_fullpath", hdf5_file_fullpath)
        # print("pickle_file_fullpath", pickle_file_fullpath)

        if os.path.isfile(hdf5_file_fullpath):
            os.remove(hdf5_file_fullpath)

        if os.path.isfile(pickle_file_fullpath):
            os.remove(pickle_file_fullpath)

        episode_keypoint_data = dict()


        episode_start_time = time.time()
        with h5py.File(hdf5_file_fullpath, 'w') as hf:
            for i, data in enumerate(dataloaders[episode_name]):
                data = data[0]
                rgb_crop_tensor = data['rgb_crop_tensor'].to(device)
                crop_params = data['crop_param']
                depth_int16 = data['depth_int16']
                key_tree_joined = data['key_tree_joined']

                # print("\n\n i = %d, idx = %d, camera_name = %s" %(i, data['idx'], data['camera_name']))

                depth = depth_int16.float()*1.0/DEPTH_IM_SCALE

                if (i % log_freq) == 0:
                    log_msg = "computing [%d/%d][%d/%d]" % (episode_counter, num_episodes, i + 1, len(dataloaders[episode_name]))
                    print(log_msg)

                B = rgb_crop_tensor.shape[0]

                _, H, W, _ = data['rgb'].shape


                kp_pred = None
                kp_pred_full_pixels = None
                with torch.no_grad():
                    kp_pred = model_kp.predict_keypoint(rgb_crop_tensor)



                    # [B, n_kp, 2]
                    kp_pred_full_pixels = transporter_utils.map_cropped_pixels_to_full_pixels_torch(kp_pred,
                                                                                                crop_params)

                    xy = kp_pred_full_pixels.clone()
                    xy[:,:, 0] = (xy[:,:, 0])*2.0/W - 1.0
                    xy[:,:, 1] = (xy[:,:, 1])*2.0/H - 1.0

                    # debug
                    # print("xy[0,0]", xy[0,0])


                    # get depth values
                    kp_pred_full_pixels_int = kp_pred_full_pixels.type(torch.LongTensor)

                    z = pdc_utils.index_into_batch_image_tensor(depth.unsqueeze(1),
                                                                kp_pred_full_pixels_int.transpose(1,2))

                    z = z.squeeze(1)
                    K_inv = data['K_inv']
                    pts_camera_frame = pdc_torch_utils.pinhole_unprojection(kp_pred_full_pixels,
                                                            z,
                                                            K_inv)

                    # print("pts_camera_frame.shape", pts_camera_frame.shape)

                    pts_world_frame = pdc_torch_utils.transform_points_3D(data['T_W_C'],
                                                                          pts_camera_frame)

                    # print("pts_world_frame.shape", pts_world_frame.shape)


                for j in range(B):

                    keypoint_data = {}

                    # this goes from [-1,1]
                    keypoint_data['xy'] = torch_utils.cast_to_numpy(xy[j])
                    keypoint_data['uv'] = torch_utils.cast_to_numpy(kp_pred_full_pixels[j])
                    keypoint_data['uv_int'] = torch_utils.cast_to_numpy(kp_pred_full_pixels_int[j])
                    keypoint_data['z'] = torch_utils.cast_to_numpy(z[j])
                    keypoint_data['pos_world_frame'] = torch_utils.cast_to_numpy(pts_world_frame[j])
                    keypoint_data['pos_camera_frame'] = torch_utils.cast_to_numpy(pts_camera_frame[j])



                    # save out some data in both hdf5 and pickle format
                    for key, val in keypoint_data.items():
                        save_key = key_tree_joined[j] + "/transporter_keypoints/%s" %(key)
                        hf.create_dataset(save_key, data=val)
                        episode_keypoint_data[save_key] = val


            save_pickle(episode_keypoint_data, pickle_file_fullpath)
            print("duration: %.3f seconds" % (time.time() - episode_start_time))











