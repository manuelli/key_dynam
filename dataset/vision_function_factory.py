# system
import torch
import numpy as np
import cv2

# key_dynam
from key_dynam.utils import drake_image_utils
from key_dynam.utils import torch_utils, transform_utils
from key_dynam.dataset.drake_sim_episode_reader import DrakeSimEpisodeReader
import key_dynam.envs.utils as env_utils
from key_dynam.transporter.models_kp import localize_transporter_keypoints
from key_dynam.autoencoder.dataset import AutoencoderImagePreprocessFunctionFactory

# pdc
from dense_correspondence_manipulation.utils import constants
from dense_correspondence.network import predict
from dense_correspondence_manipulation.utils import utils as pdc_utils
import dense_correspondence_manipulation.utils.visualization as vis_utils


class VisualObservationFunctionFactory:
    """
    Helper functions for constructing the visual observation from a raw
    environment observation
    """

    @staticmethod
    def function_from_config(config, **kwargs):
        """
        Helper function that dispatches to other functions depending on the value in
        the config
        :param config:
        :type config:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """


        obs_type = config["dataset"]["visual_observation_function"]["type"]


        if obs_type in ["precomputed_descriptor_keypoints_3D"]:
            return VisualObservationFunctionFactory.descriptor_keypoints_3D(config, **kwargs)

        elif obs_type == "precomputed_descriptor_keyoints_2D":
            VisualObservationFunctionFactory.descriptor_keypoints_3D(config, **kwargs)

        elif obs_type in ["transporter_keypoints_3D_world_frame"]:
            return VisualObservationFunctionFactory.transporter_keypoints_3D_world_frame(config, **kwargs)

        elif obs_type in ["transporter_keypoints_2D"]:
            return VisualObservationFunctionFactory.transporter_keypoints_2D(config, **kwargs)

        else:
            raise ValueError("unknown type %s" %(obs_type))


    @staticmethod
    def localize_descriptor_keypoints(env_observation,
                                      camera_name,
                                      model_dd,
                                      ref_descriptors,  # [N, D]
                                      K_matrix,  # camera K matrix
                                      T_world_camera=None,
                                      debug=False,
                                      ): # dict with 'pts_W', 'pts_C', etc.

        """
        Localizes dense descriptor keypoints
        """
        # print("env_observation keys", env_observation["images"][camera_name].keys())

        rgb = env_observation["images"][camera_name]['rgb']

        depth_16U = env_observation['images'][camera_name]['depth_16U']
        depth_int16 = drake_image_utils.remove_depth_16U_out_of_range_and_cast(depth_16U, np.int16)

        rgb_image_to_tensor = torch_utils.make_default_image_to_tensor_transform()
        rgb_tensor = rgb_image_to_tensor(rgb).cuda()

        # print("rgb_tensor.shape", rgb_tensor.shape)

        depth_torch = torch.Tensor(depth_int16) * 1.0 / constants.DEPTH_IM_SCALE
        depth_torch = depth_torch.cuda()

        # print("torch.min(depth_torch)", torch.min(depth_torch))
        # print("torch.max(depth_torch)", torch.max(depth_torch))

        # compute descriptor images
        with torch.no_grad():
            dd_out = model_dd.forward(rgb_tensor.unsqueeze(0))
            descriptor_image = dd_out['descriptor_image'].squeeze()  # [D, H, W]

            # localize the keypoints
            sigma = model_dd.config['network']['sigma_descriptor_heatmap']

            spatial_pred = predict.get_spatial_expectation(des=ref_descriptors,  # [N, D]
                                                           des_img=descriptor_image,
                                                           sigma=sigma,
                                                           type='exp',
                                                           return_heatmap=True,
                                                           compute_heatmap_values=True,
                                                           depth_images=depth_torch)

        # need camera K matrix
        uv = spatial_pred['uv'].cpu().numpy()
        z = spatial_pred['z'].cpu().numpy()
        pts_C = pdc_utils.pinhole_unprojection(uv=uv,
                                               z=z,
                                               K=K_matrix)

        #
        pts_W = None
        if T_world_camera is not None:
            pts_W = transform_utils.transform_points_3D(T_world_camera, pts_C)

        if debug:
            import matplotlib.pyplot as plt
            print("pts_W\n", pts_W)
            print("uv\n", uv)
            print("z\n")

            rgb_wr = np.copy(rgb)
            label_color = [255, 0, 0]
            vis_utils.draw_reticles(rgb_wr,
                                    uv[:, 0],
                                    uv[:, 1],
                                    label_color)
            plt.figure()
            plt.imshow(rgb_wr)
            plt.show()

        return {'pts_W': pts_W,  # [N, 3]
                'pts_C': pts_C,  # [N, 3]
                'uv': uv,
                'z': z,
                }


    @staticmethod
    def descriptor_keypoints_3D(config,
                                camera_name,
                                model_dd,
                                ref_descriptors,  # [N, D]
                                K_matrix,  # camera K matrix
                                T_world_camera,
                                debug=False,
                                **kwargs):
        """
        Localizes descriptor keypoints in 3D
        """

        def func(env_observation):
            d = VisualObservationFunctionFactory.localize_descriptor_keypoints(env_observation,
                                                                               camera_name,
                                                                               model_dd,
                                                                               ref_descriptors,
                                                                               K_matrix,
                                                                               T_world_camera)

            d['flattened'] = torch.Tensor(d['pts_W']).flatten()
            d['tensor'] = d['flattened']

            return d

        return func

    # @staticmethod
    # def descriptor_keypoints_3D(config,
    #                             camera_name,
    #                             model_dd,
    #                             ref_descriptors,  # [N, D]
    #                             debug=False,
    #                             **kwargs):
    #
    #     raise ValueError("unused")
    #     """
    #     Localizes descriptor keypoints in 3D
    #     """
    #
    #     def func(env_observation):
    #         rgb = env_observation["images"][camera_name]['rgb']
    #
    #         rgb_image_to_tensor = torch_utils.make_default_image_to_tensor_transform()
    #         rgb_tensor = rgb_image_to_tensor(rgb).cuda()
    #
    #         # compute descriptor images
    #         with torch.no_grad():
    #             dd_out = model_dd.forward(rgb_tensor.unsqueeze(0))
    #
    #         descriptor_image = dd_out['descriptor_image'].squeeze()  # [D, H, W]
    #
    #         # localize the keypoints
    #         sigma = model_dd.config['network']['sigma_descriptor_heatmap']
    #
    #         spatial_pred = predict.get_spatial_expectation(des=ref_descriptors,  # [N, D]
    #                                                        des_img=descriptor_image,
    #                                                        sigma=sigma,
    #                                                        type='exp',
    #                                                        return_heatmap=True,
    #                                                        compute_heatmap_values=True,
    #                                                        )
    #
    #         d = dict()
    #         d['xy'] = spatial_pred['xy']
    #         d['flattened'] = torch.Tensor(d['xy']).flatten()
    #         d['tensor'] = d['flattened']
    #         return d
    #
    #     return func


    @staticmethod
    def transporter_keypoints_3D_world_frame(config,
                                             camera_name,
                                             model_kp,
                                             K_matrix,  # camera K matrix
                                             T_world_camera,
                                             mask_labels=None,
                                             debug=False,
                                             **kwargs):

        """
        Returns a function that produces transporter_3D keypoints
        :param config:
        :type config:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """

        assert mask_labels is not None

        def func(env_observation):
            rgb = env_observation["images"][camera_name]['rgb']
            depth_16U = env_observation['images'][camera_name]['depth_16U']
            depth_int16 = drake_image_utils.remove_depth_16U_out_of_range_and_cast(depth_16U, np.int16)
            depth = depth_int16 * 1.0 / constants.DEPTH_IM_SCALE

            label = env_observation['images'][camera_name]['label']
            mask = env_utils.binary_mask_from_label_image(label, mask_labels)

            d = localize_transporter_keypoints(model_kp,
                                               rgb=rgb,
                                               mask=mask,
                                               depth=depth,
                                               K=K_matrix,
                                               T_world_camera=T_world_camera
                                               )

            d['flattened'] = torch.Tensor(d['pts_world_frame']).flatten()
            d['tensor'] = d['flattened']

            return d

        return func

    @staticmethod
    def transporter_keypoints_2D(config,
                                 camera_name,
                                 model_kp,
                                 K_matrix,  # camera K matrix
                                 T_world_camera,
                                 mask_labels=None,
                                 debug=False,
                                 **kwargs):

        """
        Returns a function that produces transporter_3D keypoints
        :param config:
        :type config:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """

        assert mask_labels is not None

        func_3D = VisualObservationFunctionFactory.transporter_keypoints_3D_world_frame(config,
                                                                                        camera_name,
                                                                                        model_kp,
                                                                                        K_matrix,  # camera K matrix
                                                                                        T_world_camera,
                                                                                        mask_labels=mask_labels,
                                                                                        debug=False,
                                                                                        )

        def func(env_observation):
            d = func_3D(env_observation)
            d['flattened'] = torch.Tensor(d['xy']).flatten()
            d['tensor'] = d['flattened']

            return d

        return func


    @staticmethod
    def autoencoder_latent_state(config,
                                 model_ae, # autoencoder model
                                 **kwargs):
        """
        Returns a function.

        This function consumes an environment observation and produces
        the autoencoder latent state

        """

        camera_name = config['perception']['camera_name']
        image_preprocess_func = AutoencoderImagePreprocessFunctionFactory.convolutional_autoencoder(config)

        def func(env_observation):

            rgb = env_observation['images'][camera_name]['rgb']
            img_proc = image_preprocess_func(rgb)

            input = img_proc['input_tensor'].unsqueeze(0).cuda()
            z = model_ae.encode(input)['z'].squeeze(0) # [z_dim]

            return {'flattened': z}

        return func




class PrecomputedVisualObservationFunctionFactory(object):

    @staticmethod
    def function_from_config(config, **kwargs):
        """
        Helper function that dispatches to other functions depending on the value in
        the config
        :param config:
        :type config:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """

        obs_type = config["dataset"]["visual_observation_function"]["type"]
        func = getattr(PrecomputedVisualObservationFunctionFactory, obs_type)(config, **kwargs)
        return func

    @staticmethod
    def precomputed_descriptor_keypoints_3D(config,
                                            keypoint_idx):

        camera_name = config['dataset']['visual_observation_function']['camera_name']

        def func(episode,  # key_dynam EpisodeReader
                 episode_idx,
                 T=None,
                 data_augmentation=False,
                 **kwargs):
            # only support it for these types right now
            # assert isinstance(episode, DrakeSimEpisodeReader)

            image_episode_idx = episode_idx
            try:
                image_episode_idx = episode.image_episode_idx_from_query_idx(episode_idx)
            except NotImplementedError:
                pass

            key = "descriptor_keypoints/pos_world_frame"
            keypoints_3d = episode.image_episode.get_precomputed_data(camera_name=camera_name,
                                                                      idx=image_episode_idx,
                                                                      type=key)[keypoint_idx]

            # keypoints_3d = \
            #     image_episode.get_image(camera_name, image_episode_idx, "descriptor_keypoints_3d_world_frame")[
            #         keypoint_idx]

            if T is not None:
                keypoints_3d = transform_utils.transform_points_3D(T,
                                                                   keypoints_3d)

            keypoints_tensor = torch.Tensor(keypoints_3d).flatten()
            return {'keypoints_3d': keypoints_3d,
                    'tensor': keypoints_tensor}

        return func

    @staticmethod
    def precomputed_descriptor_keypoints_2D(config,
                                            keypoint_idx):

        camera_name = config['dataset']['visual_observation_function']['camera_name']

        def func(episode,  # key_dynam EpisodeReader
                 episode_idx,
                 T=None,
                 data_augmentation=False,
                 **kwargs):

            # only support it for these types right now
            assert isinstance(episode, DrakeSimEpisodeReader)
            assert T is None, "don't support T_aug data augmentation"

            image_episode_idx = episode_idx
            image_episode = episode.image_episode

            key = "descriptor_keypoints/xy"
            keypoints_2d = episode.image_episode.get_precomputed_data(camera_name=camera_name,
                                                                      idx=image_episode_idx,
                                                                      type=key)[keypoint_idx]


            keypoints_tensor = torch.Tensor(keypoints_2d).flatten()
            return {'keypoints_2d': keypoints_2d,
                    'tensor': keypoints_tensor}

        return func

    @staticmethod
    def transporter_keypoints_3D_world_frame(config,
                                             **kwargs):

        key = "transporter_keypoints/pos_world_frame"
        camera_name = config["dataset"]["visual_observation_function"]['camera_name']

        def func(episode,
                 episode_idx,
                 T=None,
                 data_augmentation=False,
                 **kwargs):
            # only support it for these types right now
            assert isinstance(episode, DrakeSimEpisodeReader)

            image_episode_idx = episode_idx  # this might be different for SpartanEpisodes w/ downsampling
            image_episode = episode.image_episode
            keypoints_3d = image_episode.get_precomputed_data(camera_name=camera_name,
                                                              idx=image_episode_idx,
                                                              type=key)

            if T is not None:
                keypoints_3d = transform_utils.transform_points_3D(T,
                                                                   keypoints_3d)

            keypoints_tensor = torch.Tensor(keypoints_3d).flatten()
            return {'keypoints_3d': keypoints_3d,
                    'tensor': keypoints_tensor,
                    'flattened': keypoints_tensor}

        return func

    @staticmethod
    def transporter_keypoints_2D(config,
                                 **kwargs):

        key = "transporter_keypoints/xy"
        key_uv = "transporter_keypoints/uv"
        camera_name = config["dataset"]["visual_observation_function"]['camera_name']

        def func(episode,
                 episode_idx,
                 T=None,
                 data_augmentation=False,
                 **kwargs):
            # only support it for these types right now
            assert isinstance(episode, DrakeSimEpisodeReader)

            if T is not None:
                raise ValueError("T augmentation not supported")

            image_episode_idx = episode_idx  # this might be different for SpartanEpisodes w/ downsampling
            image_episode = episode.image_episode
            keypoints_xy = image_episode.get_precomputed_data(camera_name=camera_name,
                                                              idx=image_episode_idx,
                                                              type=key)

            keypoints_uv = image_episode.get_precomputed_data(camera_name=camera_name,
                                                              idx=image_episode_idx,
                                                              type=key_uv)

            keypoints_tensor = torch.Tensor(keypoints_xy).flatten()
            return {'keypoints_xy': keypoints_xy,
                    'keypoints_uv': keypoints_uv,
                    'tensor': keypoints_tensor,
                    'flattened': keypoints_tensor,
                    }

        return func
