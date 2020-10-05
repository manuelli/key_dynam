from future.utils import iteritems
import os
import h5py
import time
import matplotlib.pyplot as plt
import shutil

# torch
import torch
from torch.utils.data import DataLoader

# key_dynam
from key_dynam.dense_correspondence.image_dataset import ImageDataset
from key_dynam.utils.utils import save_yaml, save_pickle, load_pickle
from key_dynam.dynamics.utils import set_seed
from key_dynam.dense_correspondence.descriptor_net import sample_descriptors
from key_dynam.utils import transform_utils, torch_utils
from key_dynam.dense_correspondence.keypoint_selection import score_and_select_spatially_separated_keypoints


# pdc
from dense_correspondence.network.predict import get_argmax_l2, get_spatial_expectation, get_integral_preds_3d
from dense_correspondence_manipulation.utils import constants
import dense_correspondence_manipulation.utils.utils as pdc_utils
from dense_correspondence_manipulation.utils.visualization import draw_reticles



def sample_reference_descriptors(multi_episode_dict,
                                 model,
                                 model_file=None,
                                 episode_name_arg=None,
                                 episode_idx=None,
                                 camera_name=None,  # required
                                 num_ref_descriptors=None,
                                 ):  # dict containing information about the reference descriptors

    # can't be left blank
    assert camera_name is not None
    assert num_ref_descriptors is not None

    device = next(model.parameters()).device
    model = model.eval()  # make sure model is in eval mode

    # build all the dataset
    datasets = {}
    dataloaders = {}
    for episode_name, episode in iteritems(multi_episode_dict):
        single_episode_dict = {episode_name: episode}
        config = None
        datasets[episode_name] = ImageDataset(config, single_episode_dict, phase="all")
        dataloaders[episode_name] = DataLoader(datasets[episode_name],
                                               batch_size=1,
                                               num_workers=1,
                                               shuffle=False)
    # sample ref descriptors
    if episode_name_arg is None:
        episode_list = list(multi_episode_dict.keys())
        episode_list.sort()
        episode_name_arg = episode_list[0]

    if episode_idx is None:
        episode_idx = 0

    dataset = datasets[episode_name_arg]
    episode = multi_episode_dict[episode_name_arg]
    data = dataset._getitem(episode,
                            episode_idx,
                            camera_name)

    rgb_tensor = data['rgb_tensor'].unsqueeze(0).to(device)

    # compute descriptor image from model
    with torch.no_grad():
        out = model.forward(rgb_tensor)
        des_img = out['descriptor_image']

        # get image mask and cast it to a tensor on the appropriate
        # device
        img_mask = episode.get_image(camera_name,
                                     episode_idx,
                                     type="mask")
        img_mask = torch.tensor(img_mask).to(des_img.device)

        ref_descriptors_dict = sample_descriptors(des_img.squeeze(), img_mask, num_ref_descriptors)

        ref_descriptors = ref_descriptors_dict['descriptors']
        ref_indices = ref_descriptors_dict['indices']

        print("ref_descriptors_dict\n", ref_descriptors_dict)

    # save metadata in dict
    metadata = {'model_file': model_file,
                'ref_descriptors': ref_descriptors.cpu().numpy(),
                'indices': ref_indices.cpu().numpy(),
                'episode_name': episode_name_arg,
                'episode_idx': episode_idx,
                'camera_name': camera_name}

    return metadata


def precompute_descriptors(multi_episode_dict,
                           model,
                           output_dir,  # str
                           batch_size=10,
                           num_workers=10,
                           model_file=None,
                           ):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    metadata = {'model_file': model_file}
    metadata_file = os.path.join(output_dir, 'metadata.yaml')
    save_yaml(metadata, metadata_file)

    start_time = time.time()

    log_freq = 10

    device = next(model.parameters()).device
    model.eval()  # make sure model is in eval mode

    # build all the dataset
    datasets = {}
    dataloaders = {}
    for episode_name, episode in iteritems(multi_episode_dict):
        single_episode_dict = {episode_name: episode}
        config = None
        datasets[episode_name] = ImageDataset(config, single_episode_dict, phase="all")
        dataloaders[episode_name] = DataLoader(datasets[episode_name],
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               shuffle=False)

    episode_counter = 0
    num_episodes = len(multi_episode_dict)

    for episode_name, dataset in iteritems(datasets):
        episode_counter += 1
        print("\n\n")

        episode = multi_episode_dict[episode_name]
        hdf5_file = os.path.basename(episode.image_data_file)
        hdf5_file_fullpath = os.path.join(output_dir, hdf5_file)

        if os.path.isfile(hdf5_file_fullpath):
            os.remove(hdf5_file_fullpath)

        dataloader = dataloaders[episode_name]

        episode_start_time = time.time()
        with h5py.File(hdf5_file_fullpath, 'w') as hf:
            for i, data in enumerate(dataloaders[episode_name]):
                rgb_tensor = data['rgb_tensor'].to(device)
                key_tree_joined = data['key_tree_joined']

                if (i % log_freq) == 0:
                    log_msg = "computing [%d/%d][%d/%d]" % (episode_counter, num_episodes, i + 1, len(dataloader))
                    print(log_msg)

                # don't use gradients
                with torch.no_grad():
                    start_time = time.time()
                    out = model.forward(rgb_tensor)
                    print("forward took", time.time() - start_time)
                    B, _, H, W = rgb_tensor.shape

                    # iterate over elements in the batch
                    start_time = time.time()
                    for j in range(B):
                        # [D, H, W]
                        des_image = out['descriptor_image'][j].cpu().numpy()
                        key = key_tree_joined[j] + "/descriptor_image"
                        hf.create_dataset(key, data=des_image)

                    print("saving images took", time.time() - start_time)

        print("duration: %.3f seconds" % (time.time() - episode_start_time))

    print("total time to compute descriptors: %.3f seconds" % (time.time() - start_time))


def precompute_descriptor_keypoints(multi_episode_dict,
                                    model,
                                    output_dir,  # str
                                    ref_descriptors_metadata,
                                    batch_size=10,
                                    num_workers=10,
                                    localization_type="spatial_expectation",  # ['spatial_expectation', 'argmax']
                                    compute_3D=True,  # in world frame
                                    camera_names=None,
                                    ):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    start_time = time.time()

    log_freq = 10

    device = next(model.parameters()).device
    model = model.eval()  # make sure model is in eval mode

    # build all the dataset
    datasets = {}
    dataloaders = {}
    for episode_name, episode in iteritems(multi_episode_dict):
        single_episode_dict = {episode_name: episode}
        config = None
        datasets[episode_name] = ImageDataset(config,
                                              single_episode_dict,
                                              phase="all",
                                              camera_names=camera_names)
        dataloaders[episode_name] = DataLoader(datasets[episode_name],
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               shuffle=False)

    # K = num_ref_descriptors
    metadata = ref_descriptors_metadata
    ref_descriptors = torch.Tensor(metadata['ref_descriptors'])
    ref_descriptors = ref_descriptors.cuda()
    K, _ = ref_descriptors.shape

    metadata_file = os.path.join(output_dir, 'metadata.p')
    save_pickle(metadata, metadata_file)

    episode_counter = 0
    num_episodes = len(multi_episode_dict)

    for episode_name, dataset in iteritems(datasets):
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

        # print("hdf5_file_fullpath", hdf5_file_fullpath)
        # print("pickle_file_fullpath", pickle_file_fullpath)

        if os.path.isfile(hdf5_file_fullpath):
            os.remove(hdf5_file_fullpath)

        if os.path.isfile(pickle_file_fullpath):
            os.remove(pickle_file_fullpath)

        dataloader = dataloaders[episode_name]


        episode_keypoint_data = dict()

        episode_start_time = time.time()
        with h5py.File(hdf5_file_fullpath, 'w') as hf:
            for i, data in enumerate(dataloaders[episode_name]):
                rgb_tensor = data['rgb_tensor'].to(device)
                key_tree_joined = data['key_tree_joined']

                if (i % log_freq) == 0:
                    log_msg = "computing [%d/%d][%d/%d]" % (episode_counter, num_episodes, i + 1, len(dataloader))
                    print(log_msg)

                # don't use gradients
                tmp_time = time.time()
                with torch.no_grad():
                    out = model.forward(rgb_tensor)

                    # [B, D, H, W]
                    des_img = out['descriptor_image']

                    B, _, H, W = rgb_tensor.shape

                    # [B, N, 2]
                    batch_indices = None
                    preds_3d = None
                    if localization_type == "spatial_expectation":
                        sigma_descriptor_heatmap = 5  # default
                        try:
                            sigma_descriptor_heatmap = model.config['network']['sigma_descriptor_heatmap']
                        except:
                            pass

                        # print("ref_descriptors.shape", ref_descriptors.shape)
                        # print("des_img.shape", des_img.shape)
                        d = get_spatial_expectation(ref_descriptors,
                                                    des_img,
                                                    sigma=sigma_descriptor_heatmap,
                                                    type='exp',
                                                    return_heatmap=True,
                                                    )

                        batch_indices = d['uv']

                        # [B, K, H, W]
                        if compute_3D:
                            # [B*K, H, W]
                            heatmaps_no_batch = d['heatmap_no_batch']

                            # [B, H, W]
                            depth = data['depth_int16'].to(device)

                            # expand depth images and convert to meters, instead of mm
                            # [B, K, H, W]
                            depth_expand = depth.unsqueeze(1).expand([B, K, H, W]).reshape([B * K, H, W])
                            depth_expand = depth_expand.type(torch.FloatTensor) / constants.DEPTH_IM_SCALE
                            depth_expand = depth_expand.to(heatmaps_no_batch.device)

                            pred_3d = get_integral_preds_3d(heatmaps_no_batch,
                                                            depth_images=depth_expand,
                                                            compute_uv=True)

                            pred_3d['uv'] = pred_3d['uv'].reshape([B, K, 2])
                            pred_3d['xy'] = pred_3d['xy'].reshape([B, K, 2])
                            pred_3d['z'] = pred_3d['z'].reshape([B, K])


                    elif localization_type == "argmax":
                        # localize descriptors
                        best_match_dict = get_argmax_l2(ref_descriptors,
                                                        des_img)

                        # [B, N, 2]
                        # where N is num_ref_descriptors
                        batch_indices = best_match_dict['indices']
                    else:
                        raise ValueError("unknown localization type: %s" % (localization_type))

                    print("computing keypoints took", time.time() - tmp_time)

                    tmp_time = time.time()
                    # iterate over elements in the batch
                    for j in range(B):
                        keypoint_data = {} # dict that stores information to save out

                        # [N,2]
                        # indices = batch_indices[j].cpu().numpy()
                        # key = key_tree_joined[j] + "/descriptor_keypoints"


                        # hf.create_dataset(key, data=indices)
                        # keypoint_indices_dict[key] = indices

                        # stored 3D keypoint locations (in both camera and world frame)
                        if pred_3d is not None:


                            # key_3d_W = key_tree_joined[j] + "/descriptor_keypoints_3d_world_frame"
                            # key_3d_C = key_tree_joined[j] + "/descriptor_keypoints_3d_camera_frame"

                            # T_W_C = data['T_world_camera'][j].cpu().numpy()
                            # K_matrix = data['K'][j].cpu().numpy()

                            T_W_C = torch_utils.cast_to_numpy(data['T_world_camera'][j])
                            K_matrix = torch_utils.cast_to_numpy(data['K'][j])

                            uv = torch_utils.cast_to_numpy(pred_3d['uv'][j])
                            xy = torch_utils.cast_to_numpy(pred_3d['xy'][j])
                            z = torch_utils.cast_to_numpy(pred_3d['z'][j])

                            # [K, 3]
                            # this is in camera frame
                            pts_3d_C = pdc_utils.pinhole_unprojection(uv, z, K_matrix)
                            # hf.create_dataset(key_3d_C, data=pts_3d_C)
                            # keypoint_indices_dict[key_3d_C] = pts_3d_C

                            # project into world frame
                            pts_3d_W = transform_utils.transform_points_3D(transform=T_W_C,
                                                                           points=pts_3d_C)

                            # hf.create_dataset(key_3d_W, data=pts_3d_W)
                            # keypoint_indices_dict[key_3d_W] = pts_3d_W

                            keypoint_data['xy'] = torch_utils.cast_to_numpy(xy)
                            keypoint_data['uv'] = torch_utils.cast_to_numpy(uv)
                            keypoint_data['z'] = torch_utils.cast_to_numpy(z)
                            keypoint_data['pos_world_frame'] = torch_utils.cast_to_numpy(pts_3d_W)
                            keypoint_data['pos_camera_frame'] = torch_utils.cast_to_numpy(pts_3d_C)

                        # save out some data in both hdf5 and pickle format
                        for key, val in keypoint_data.items():
                            save_key = key_tree_joined[j] + "/descriptor_keypoints/%s" % (key)
                            hf.create_dataset(save_key, data=val)
                            episode_keypoint_data[save_key] = val


                    print("saving to disk took", time.time() - tmp_time)

        # save_pickle(keypoint_indices_dict, pickle_file_fullpath)
        save_pickle(episode_keypoint_data, pickle_file_fullpath)
        print("duration: %.3f seconds" % (time.time() - episode_start_time))

    print("total time to compute descriptors: %.3f seconds" % (time.time() - start_time))


def compute_descriptor_confidences(multi_episode_dict,
                                   model,
                                   output_dir,  # str
                                   batch_size=10,
                                   num_workers=10,
                                   model_file=None,
                                   ref_descriptors=None,
                                   episode_name_arg=None,
                                   episode_idx=None,
                                   camera_name=None,
                                   num_ref_descriptors=None,
                                   localization_type="spatial_expectation",  # ['spatial_expectation', 'argmax']
                                   num_batches=None,
                                   ):
    """
    Computes confidence scores for different reference descriptors.
    Saves two files

    metadata.p: has information about reference descriptors, etc.
    data.p: descriptor confidence scores
    """

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    start_time = time.time()

    log_freq = 10

    device = next(model.parameters()).device
    model.eval()  # make sure model is in eval mode

    # build all the dataset
    config = None
    dataset = ImageDataset(config, multi_episode_dict, phase="all")
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            shuffle=True)

    # sample ref descriptors
    if ref_descriptors is None:
        if episode_name_arg is None:
            episode_list = list(multi_episode_dict.keys())
            episode_list.sort()
            episode_name_arg = episode_list[0]

        if episode_idx is None:
            episode_idx = 0

        # can't be left blank
        assert camera_name is not None
        episode = multi_episode_dict[episode_name_arg]
        data = dataset._getitem(episode,
                                episode_idx,
                                camera_name)

        rgb_tensor = data['rgb_tensor'].unsqueeze(0).to(device)

        # compute descriptor image from model
        with torch.no_grad():
            out = model.forward(rgb_tensor)
            des_img = out['descriptor_image']

            # get image mask and cast it to a tensor on the appropriate
            # device
            img_mask = episode.get_image(camera_name,
                                         episode_idx,
                                         type="mask")
            img_mask = torch.tensor(img_mask).to(des_img.device)

            ref_descriptors_dict = sample_descriptors(des_img.squeeze(), img_mask, num_ref_descriptors)

            ref_descriptors = ref_descriptors_dict['descriptors']
            ref_indices = ref_descriptors_dict['indices']

            print("ref_descriptors_dict\n", ref_descriptors_dict)

    # save metadata in dict
    metadata = {'model_file': model_file,
                'ref_descriptors': ref_descriptors.cpu().numpy(),  # [N, D]
                'indices': ref_indices.cpu().numpy(),
                'episode_name': episode_name_arg,
                'episode_idx': episode_idx,
                'camera_name': camera_name}

    metadata_file = os.path.join(output_dir, 'metadata.p')
    save_pickle(metadata, metadata_file)

    scores = dict()
    heatmap_value_list = []

    for i, data in enumerate(dataloader):
        if num_batches is not None and i > num_batches:
            break
        rgb_tensor = data['rgb_tensor'].to(device)
        key_tree_joined = data['key_tree_joined']

        if (i % log_freq) == 0:
            log_msg = "computing %d" % (i)
            print(log_msg)

        # don't use gradients
        with torch.no_grad():
            out = model.forward(rgb_tensor)

            # [B, D, H, W]
            des_img = out['descriptor_image']

            B, _, H, W = rgb_tensor.shape

            heatmap_values = None
            if localization_type == "spatial_expectation":
                sigma_descriptor_heatmap = 5  # default
                try:
                    sigma_descriptor_heatmap = model.config['network']['sigma_descriptor_heatmap']
                except:
                    pass

                # print("ref_descriptors.shape", ref_descriptors.shape)
                # print("des_img.shape", des_img.shape)
                d = get_spatial_expectation(ref_descriptors,
                                            des_img,
                                            sigma=sigma_descriptor_heatmap,
                                            type='exp',
                                            compute_heatmap_values=True,
                                            return_heatmap=True,
                                            )

                # [B, K]
                heatmap_values = d['heatmap_values']
            else:
                raise ValueError("unknown localization type: %s" % (localization_type))

            heatmap_value_list.append(heatmap_values)

    heatmap_values_tensor = torch.cat(heatmap_value_list)
    heatmap_values_np = heatmap_values_tensor.cpu().numpy()
    save_data = {'heatmap_values': heatmap_values_np}

    data_file = os.path.join(output_dir, 'data.p')
    save_pickle(save_data, data_file)
    print("total time to compute descriptors: %.3f seconds" % (time.time() - start_time))


def select_spatially_separated_descriptors(K=5,  # number of reference descriptors
                                          output_dir=None,
                                          visualize=False):
    raise ValueError("deprecated")
    multi_episode_dict = exp_utils.load_episodes()['multi_episode_dict']
    model_file = exp_utils.get_DD_model_file()

    confidence_scores_folder = os.path.join(get_data_root(),
                                            "dev/experiments/09/descriptor_confidence_scores/2020-03-25-19-57-26-556093_constant_velocity_500/2020-03-30-14-21-13-371713")

    # folder = "dev/experiments/07/descriptor_confidence_scores/real_push_box/2020-03-10-15-57-43-867147"
    folder = confidence_scores_folder
    folder = os.path.join(get_data_root(), folder)
    data_file = os.path.join(folder, 'data.p')
    data = load_pickle(data_file)

    heatmap_values = data['heatmap_values']
    scoring_func = keypoint_selection.create_scoring_function(gamma=3)
    score_data = keypoint_selection.score_heatmap_values(heatmap_values,
                                                         scoring_func=scoring_func)
    sorted_idx = score_data['sorted_idx']

    metadata_file = os.path.join(folder, 'metadata.p')
    metadata = load_pickle(metadata_file)
    camera_name = metadata['camera_name']

    keypoint_idx = keypoint_selection.select_spatially_separated_keypoints(sorted_idx,
                                                                           metadata['indices'],
                                                                           position_diff_threshold=30,
                                                                           K=K,
                                                                           verbose=False)

    ref_descriptors = metadata['ref_descriptors'][keypoint_idx]  # [K, D]
    spatial_descriptors_data = score_data
    spatial_descriptors_data['spatial_descriptors'] = ref_descriptors
    spatial_descriptors_data['spatial_descriptors_idx'] = keypoint_idx
    save_pickle(spatial_descriptors_data, os.path.join(folder, 'spatial_descriptors.p'))


def run_precompute_descriptors_pipeline(multi_episode_dict, # dict
                                        model, # dense descriptor model file
                                        model_file=None,
                                        output_dir=None, # str where to save data
                                        episode_name=None, # optional for descriptor sampling
                                        camera_name=None, # which camera to compute descriptors for
                                        episode_idx=None, # optional for descriptor sampling
                                        visualize=True,
                                        K=5,
                                        position_diff_threshold=20,
                                        seed=0,
                                        ):

    assert model_file is not None
    assert camera_name is not None
    #
    # ## Load Model
    # model_train_dir = os.path.dirname(model_file)
    #
    # print("model_train_dir", model_train_dir)
    # print("model_file", model_file)
    # model = torch.load(model_file)
    # model = model.cuda()
    # model = model.eval()
    #
    # output_dir = os.path.join(model_train_dir,
    #                           'precomputed_vision_data/descriptor_keypoints/dataset_%s/' % (dataset_name))

    # compute descriptor confidence scores
    set_seed(seed)
    if True:
        print("\n\n---------Computing Descriptor Confidence Scores-----------")
        metadata_file = os.path.join(output_dir, 'metadata.p')
        if os.path.isfile(metadata_file):
            answer = input("metadata.p file already exists, do you want to overwrite it? y/n\n")

            if answer == "y":
                shutil.rmtree(output_dir)
                print("removing existing file and continuing")

            else:
                print("aborting")
                quit()



        compute_descriptor_confidences(multi_episode_dict,
                                       model,
                                       output_dir,
                                       batch_size=10,
                                       num_workers=20,
                                       model_file=model_file,
                                       camera_name=camera_name,
                                       num_ref_descriptors=50,
                                       num_batches=10,
                                       episode_name_arg=episode_name,
                                       episode_idx=episode_idx,
                                       )

    if True:
        confidence_score_data_file = os.path.join(output_dir, 'data.p')
        confidence_score_data = load_pickle(confidence_score_data_file)

        metadata_file = os.path.join(output_dir, 'metadata.p')
        metadata = load_pickle(metadata_file)

        print("\n\n---------Selecting Spatially Separated Keypoints-----------")
        score_and_select_spatially_separated_keypoints(metadata,
                                                       confidence_score_data=confidence_score_data,
                                                       K=K,
                                                       position_diff_threshold=position_diff_threshold,
                                                       output_dir=output_dir,
                                                       )

    # visualize descriptors
    if True:
        metadata_file = os.path.join(output_dir, 'metadata.p')
        metadata = load_pickle(metadata_file)

        episode_name = metadata['episode_name']
        episode_idx = metadata['episode_idx']
        camera_name = metadata['camera_name']

        episode = multi_episode_dict[episode_name]
        data = episode.get_image_data(camera_name, episode_idx)
        rgb = data['rgb']

        uv = metadata['indices']

        print("uv.shape", uv.shape)

        color = [0, 255, 0]
        draw_reticles(rgb, uv[:, 0], uv[:, 1], label_color=color)

        save_file = os.path.join(output_dir, 'sampled_descriptors.png')


        plt.figure()
        plt.imshow(rgb)
        plt.savefig(save_file)
        if visualize:
            plt.show()

    # visualize spatially separated descriptors
    if True:
        metadata_file = os.path.join(output_dir, 'metadata.p')
        metadata = load_pickle(metadata_file)

        spatial_descriptor_file = os.path.join(output_dir, 'spatial_descriptors.p')
        spatial_descriptors_data = load_pickle(spatial_descriptor_file)
        des_idx = spatial_descriptors_data['spatial_descriptors_idx']

        episode_name = metadata['episode_name']
        episode_idx = metadata['episode_idx']
        camera_name = metadata['camera_name']

        episode = multi_episode_dict[episode_name]
        data = episode.get_image_data(camera_name, episode_idx)
        rgb = data['rgb']

        uv = metadata['indices']

        print("uv.shape", uv.shape)

        color = [0, 255, 0]
        draw_reticles(rgb, uv[des_idx, 0], uv[des_idx, 1], label_color=color)

        save_file = os.path.join(output_dir, 'spatially_separated_descriptors.png')

        plt.figure()
        plt.imshow(rgb)
        plt.savefig(save_file)
        if visualize:
            plt.show()


    if True:
        metadata_file = os.path.join(output_dir, 'metadata.p')
        metadata = load_pickle(metadata_file)

        print("\n\n---------Precomputing Descriptor Keypoints-----------")
        descriptor_keypoints_output_dir = os.path.join(output_dir, "descriptor_keypoints")
        precompute_descriptor_keypoints(multi_episode_dict,
                                        model,
                                        descriptor_keypoints_output_dir,
                                        ref_descriptors_metadata=metadata,
                                        batch_size=8,
                                        num_workers=20,
                                        camera_names=[camera_name]
                                        )

    print("Data saved at: ", output_dir)
    print("Finished Normally")