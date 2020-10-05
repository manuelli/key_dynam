from __future__ import print_function

# system
import numpy as np
import torch

# key_dynam
import key_dynam.utils.transform_utils as transform_utils
from key_dynam.dataset.dynamic_spartan_episode_reader import DynamicSpartanEpisodeReader


class ObservationFunctionFactory(object):
    """
    These methods are (or return) functions which consume
    raw observations and return processed observations
    as torch tensors
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

        obs_type = config["dataset"]["observation_function"]["type"]
        func = getattr(ObservationFunctionFactory, obs_type)(config, **kwargs)
        return func

    @staticmethod
    def pusher_slider_pose(config,
                           **kwargs):
        """
        Wrapper for pusher_slider_pose_function
        :param config:
        :type config:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        return ObservationFunctionFactory.pusher_slider_pose_function

    @staticmethod
    def pusher_slider_pose_function(obs_raw,  # dict containing the observations
                                    **kwargs):  # return -> torch.FloatTensor shape (5,)

        slider_pose = np.zeros(3)
        slider_pose[:2] = obs_raw["slider"]["position"]
        slider_pose[2] = obs_raw["slider"]["angle"]

        # don't worry about pusher angle since it's always constant
        # and the pusher is symmetric so it's irrelevant
        pusher_pose = np.array(obs_raw["pusher"]["position"])

        pusher_slider_np = np.concatenate((slider_pose, pusher_pose))
        obs_tensor = torch.FloatTensor(pusher_slider_np)
        return obs_tensor

    @staticmethod
    def pusher_pose_slider_keypoints(config,
                                     **kwargs):  # return -> a function (the observation function)

        from key_dynam.envs.pusher_slider import PusherSlider
        DEBUG_PRINTS = False
        width, height = config["env"]["slider"]["size"]

        # keypoints in body frame
        keypoints_B = PusherSlider.box_keypoints(width, height)

        config_data_aug = config["dataset"]["observation_function"]["data_augmentation"]

        def func(obs_raw, data_augmentation=False, **kwargs):

            # don't worry about pusher angle since it's always constant
            # and the pusher is symmetric so it's irrelevant
            pusher_pose = np.array(obs_raw["pusher"]["position"])

            T_W_B = transform_utils.translation_rotation_to_2D_homogeneous_transform(obs_raw["slider"]["position"],
                                                                                     obs_raw["slider"]["angle"])

            if DEBUG_PRINTS:
                print("\n\n")
                print("T_W_B.shape", T_W_B.shape)
                print("T_W_B\n", T_W_B)
                print("keypoints_B.shape", keypoints_B.shape)

            # transform slider keypoints to world frame
            keypoints_W = transform_utils.transform_points_2D(T_W_B, keypoints_B)

            # add noise to the keypoints
            # do this via keyword args

            if DEBUG_PRINTS:
                print("slider position\n", obs_raw["slider"]["position"])
                print("slider angle\n", obs_raw["slider"]["angle"])
                print("keypoints_B\n", keypoints_B)
                print("keypoints_W\n", keypoints_W)
                print("\n\n")

            # noise augmentation
            if data_augmentation and config_data_aug["keypoints"]["augment"]:
                # print("ADDING NOISE")
                noise = np.random.normal(scale=config_data_aug["keypoints"]["std_dev"],
                                         size=keypoints_W.shape)
                keypoints_W += noise

            obs_tensor_np = np.concatenate((keypoints_W.flatten(), pusher_pose))
            obs_tensor = torch.FloatTensor(obs_tensor_np)
            return obs_tensor

        return func

    @staticmethod
    def drake_pusher_pose(config,
                          **kwargs):

        raise NotImplementedError("DEPRECATED")

        def func(obs_raw,
                 data_augmentation=False,
                 T=None,
                 **kwargs):
            # don't worry about pusher angle since it's always constant
            # and the pusher is symmetric so it's irrelevant
            pusher_pose = np.array(obs_raw["pusher"]["position"])

            obs_tensor = torch.FloatTensor(pusher_pose)
            return obs_tensor

        return func

    @staticmethod
    def drake_pusher_position_2D(config,
                                 **kwargs):

        func_3D = ObservationFunctionFactory.drake_pusher_position_3D(config, **kwargs)

        def func(obs_raw,
                 T=None,  # transform to apply to all data
                 **kwargs):
            action_tensor = func_3D(obs_raw, T=T, **kwargs)
            return action_tensor[:2]  # only first 2 elements to make it 2D instead of 3D

    @staticmethod
    def drake_pusher_position_3D(config,
                                 **kwargs):

        config_data_aug = None
        try:
            config_data_aug = config["dataset"]["observation_function"]["data_augmentation"]
        except:
            pass

        def func(obs_raw,
                 T=None,  # transform to apply to all data
                 data_augmentation=False,
                 **kwargs):
            # print("obs_raw['pusher']'", obs_raw['pusher'])
            pos = np.array(obs_raw['pusher']['T_W_B']['position'])

            # transform it by T if that was passed in
            if T is not None:
                pos = np.matmul(T, np.append(pos, 1))[:3]

            # noise augmentation
            if data_augmentation and (config_data_aug is not None) and config_data_aug["pusher_position"]["augment"]:
                # print("pusher: ADDING NOISE")
                noise = np.random.normal(scale=config_data_aug["pusher_position"]["std_dev"],
                                         size=pos.shape)
                pos += noise

            return torch.Tensor(pos)

        return func

    @staticmethod
    def spartan_ee_position_2D(config,
                               **kwargs):

        def func(obs_raw, **kwargs):
            """
            Returns the x-y position of the gripper
            :param obs_raw:
            :type obs_raw:
            :return:
            :rtype:
            """
            pos = obs_raw['ee_to_world']['translation']
            obs_tensor = torch.FloatTensor([pos['x'], pos['y']])
            return obs_tensor

        return func

    @staticmethod
    def spartan_ee_points(config,
                          **kwargs):

        # [N, 3]
        pts = np.array(config['dataset']['observation_function']['ee_points'])

        def func(obs_raw,
                 data_raw=None,
                 T=None,  # transform for data augmentation
                 **kwargs,
                 ):  # don't support T_aug for now

            # print("data_raw.keys()", data_raw.keys())
            # print("data_raw['observations']['ee_to_world']", data_raw['observations']['ee_to_world'])
            # ee to world transform
            T_W_E = DynamicSpartanEpisodeReader.ee_to_world(data_raw)

            if T is not None:
                T_W_E = T @ T_W_E

            pts_E = transform_utils.transform_points_3D(T_W_E, pts)

            obs_tensor = torch.FloatTensor(pts_E).flatten()
            return obs_tensor

        return func

    @staticmethod
    def spartan_ee_points_and_gripper_width(config,
                                            **kwargs):
        func_ee_points = ObservationFunctionFactory.spartan_ee_points(config, **kwargs)

        def func(obs_raw,
                 **kwargs_loc):
            obs_tensor = func_ee_points(obs_raw=obs_raw, **kwargs_loc)
            gripper_width = obs_raw['gripper_state']['width']
            return torch.cat((obs_tensor, torch.Tensor([gripper_width])))

        return func


def drake_pusher_position_3D_slider_GT_3D(config,
                                          **kwargs):
    pusher_pose_func = ObservationFunctionFactory.drake_pusher_position_3D(config, **kwargs)
    GT_3D_func = ObservationFunctionFactory.drake_pusher_slider_GT_3D(config, **kwargs)

    def func(obs_raw,
             T=None,  # 3 x 3 homogeneous transform to apply to data
             **kwargs):
        # [N, 3]
        pts_W = GT_3D_func(obs_raw, T=T, **kwargs)

        # [1, 3]
        pusher_pose = pusher_pose_func(obs_raw, T=T, **kwargs).unsqueeze(0)

        # [M, 3] = [N+1, 3] object pts and pusher_pose
        obs_tensor = torch.cat((pts_W, pusher_pose), dim=0)

        # flatten it
        return obs_tensor.flatten()

    return func


@staticmethod
def drake_pusher_position_3D_slider_GT_3D_noisy_keypoint(config,
                                                         **kwargs):
    pusher_pose_func = ObservationFunctionFactory.drake_pusher_position_3D(config, **kwargs)
    GT_3D_func = ObservationFunctionFactory.drake_pusher_slider_GT_3D(config, **kwargs)

    sigma = 1.0
    try:
        sigma = config['dataset']['observation_function']['noise_std_dev']
    except:
        pass

    normal_distn = torch.distributions.normal.Normal(torch.Tensor([0.]), torch.Tensor([sigma]))

    def func(obs_raw,
             T=None,  # 3 x 3 homogeneous transform to apply to data
             **kwargs):

        # [N, 3]
        pts_W = GT_3D_func(obs_raw, T=T, **kwargs)

        # add noise to the last one
        noise = normal_distn.sample(sample_shape=pts_W[-1].shape).squeeze()
        pts_W[-1] += noise

        # [1, 3]
        pusher_pose = pusher_pose_func(obs_raw, T=T, **kwargs).unsqueeze(0)

        # [M, 3] = [N+1, 3] object pts and pusher_pose
        obs_tensor = torch.cat((pts_W, pusher_pose), dim=0)

        # flatten it
        return obs_tensor.flatten()

    return func


@staticmethod
def drake_pusher_slider_GT_3D(config,
                              **kwargs):
    # [N, 3]
    pts_obj = np.array(config['dataset']['observation_function']['GT_3D_object_points'])

    config_data_aug = None
    if "data_augmentation" in config["dataset"]["observation_function"]:
        config_data_aug = config["dataset"]["observation_function"]["data_augmentation"]

    def func(obs_raw,
             T=None,  # 3 x 3 homogeneous transform to apply to data, used for data augmentation
             data_augmentation=False,
             **kwargs,  # for compatibility
             ):
        pos = obs_raw['slider']['position']['translation']
        quat = obs_raw['slider']['position']['quaternion']
        T_W_obj = transform_utils.transform_from_pose(pos, quat)

        # transform points to world frame
        pts_W = transform_utils.transform_points_3D(T_W_obj, pts_obj)

        # apply additional transform it was passed in
        # T is typically
        if T is not None:
            pts_W = transform_utils.transform_points_3D(T, pts_W)

        # noise augmentation
        if data_augmentation and (config_data_aug is not None) and config_data_aug["keypoints"]["augment"]:
            # print("slider: ADDING NOISE")
            noise = np.random.normal(scale=config_data_aug["keypoints"]["std_dev"],
                                     size=pts_W.shape)
            pts_W += noise

        return torch.Tensor(pts_W)

    return func


class ActionFunctionFactory(object):

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

        action_type = config["dataset"]["action_function"]["type"]
        func = getattr(ActionFunctionFactory, action_type)(config, **kwargs)
        return func

    @staticmethod
    def pusher_velocity(config, **kwargs):
        return ActionFunctionFactory.pusher_velocity_function

    @staticmethod
    def pusher_velocity_function(action_raw,  # raw action vector which contains pusher velocity
                                 ):  # return -> torch.FloatTensor shape (2,)
        """
        Returns the pusher velocity as torch FloatTensor
        :param action_raw:
        :type action_raw:
        :return:
        :rtype:
        """

        action_tensor = torch.FloatTensor(action_raw)
        return action_tensor

    @staticmethod
    def pusher_velocity_3D(config,
                           **kwargs):
        raise NotImplementedError

    @staticmethod
    def drake_pusher_velocity(config,
                              **kwargs):  # torch.Tensor [2,]

        """
        Velocity of the pusher in the drake pusher slider environment.
        The returned function supports accepting a transform T and will
        apply this transform appropriately before returning the data



        :param config:
        :type config:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """

        def func(action_raw,
                 T=None,  # 4 x 4 homogeneous transform applied to the velocity
                 data_augmentation=False,
                 **kwargs):
            # action_raw is 2D velocity of the pusher
            vel_2D = action_raw

            # apply the transform T if it was passed in
            if T is not None:
                vel_3D = np.append(vel_2D, 0)
                R = T[:3, :3]  # rotation matrix part
                vel_3D = np.matmul(R, vel_3D)
                vel_2D = vel_3D[:2]

            action_tensor = torch.FloatTensor(vel_2D)
            return action_tensor

        return func

    @staticmethod
    def spartan_ee_setpoint_2D(config,
                               **kwargs):
        def func(action_raw,
                 data_raw=None,
                 **kwargs):
            pos = action_raw['ee_setpoint']['position']
            pos_setpoint_tensor = torch.FloatTensor([pos['x'], pos['y']])
            pos_ee = data_raw['observations']['ee_to_world']['translation']
            pos_ee_tensor = torch.FloatTensor([pos_ee['x'], pos_ee['y']])

            delta = pos_setpoint_tensor - pos_ee_tensor
            action_tensor = torch.cat((pos_setpoint_tensor, delta))
            return action_tensor

        return func

    @staticmethod
    def spartan_ee_points(config,
                          **kwargs):
        # [N, 3]

        pts = np.array(config['dataset']['action_function']['ee_points'])
        idx_offset = config['dataset']['action_function']['idx_offset']

        def func(obs_raw,
                 T=None,  # 4 x 4 homogeneous transform for data augmentation
                 episode=None,
                 episode_idx=None,
                 **kwargs,
                 ):  # don't support T_aug for now

            data = episode.get_data(episode_idx + idx_offset)

            # ee to world transform
            T_W_E = DynamicSpartanEpisodeReader.ee_to_world(data)
            if T is not None:
                T_W_E = T @ T_W_E
            pts_E = transform_utils.transform_points_3D(T_W_E, pts)
            obs_tensor = torch.FloatTensor(pts_E).flatten()

            return obs_tensor

        return func

    @staticmethod
    def spartan_ee_points_and_gripper_width(config,
                                            **kwargs):

        idx_offset = config['dataset']['action_function']['idx_offset']
        func_ee_points = ActionFunctionFactory.spartan_ee_points(config, **kwargs)

        def func(obs_raw,
                 T=None,  # 4 x 4 homogeneous transform for data augmentation
                 episode=None,
                 episode_idx=None,
                 **kwargs,
                 ):  # don't support T_aug for now

            obs_tensor = func_ee_points(obs_raw,
                                        T=T,  # 4 x 4 homogeneous transform for data augmentation
                                        episode=episode,
                                        episode_idx=episode_idx,
                                        **kwargs,
                                        )

            data = episode.get_data(episode_idx + idx_offset)
            gripper_width = data['observations']['gripper_state']['width']
            return torch.cat((obs_tensor, torch.Tensor([gripper_width])))

        return func

    @staticmethod
    def spartan_ee_setpoint_linear_velocity_2D(config,
                                               **kwargs):

        def func(action_raw,
                 T=None,
                 **kwargs,
                 ):
            vel_dict = action_raw['ee_setpoint']['setpoint_linear_velocity']
            vel = np.array([vel_dict['x'], vel_dict['y'], vel_dict['z']])

            if T is not None:
                vel = T[:3, :3] @ vel

            vel_2D = vel[:2]
            return torch.FloatTensor(vel_2D)

        return func


def slider_pose_from_observation(obs,  # the raw observation
                                 ):  # 4 x 4 homogeneous transform
    pos = obs['slider']['position']['translation']
    quat = obs['slider']['position']['quaternion']
    T_W_obj = transform_utils.transform_from_pose(pos, quat)
    return T_W_obj


def pusher_slider_pose_from_tensor(output,  # np.array shape [5,]
                                   ):
    """
    Extracts pusher-slider pose from the raw output
    :param output:
    :type output:
    :return:
    :rtype:
    """
    assert len(output) == 5

    d = dict()
    slider = dict()
    slider['position'] = np.array([output[0], output[1]])
    slider['angle'] = output[2]

    pusher = dict()
    pusher['position'] = np.array([output[3], output[4]])

    d['slider'] = slider
    d['pusher'] = pusher

    return d


def pusher_pose_slider_keypoints_from_tensor(output,  # np.array shape [2*num_keypoints + 2]
                                             ):
    DEBUG_PRINTS = False

    pusher = dict()
    pusher['position'] = np.array([output[-2], output[-1]])

    keypoints_tensor = output[:-2]

    num_keypoints = keypoints_tensor.size / 2

    if DEBUG_PRINTS:
        print("keypoints_tensor.size", keypoints_tensor.size)
        print("num_keypoints", num_keypoints)

    keypoint_positions = np.reshape(keypoints_tensor, [num_keypoints, 2])

    if DEBUG_PRINTS:
        print("keypoint_positions", keypoint_positions)

    d = dict()
    d['keypoint_positions'] = keypoint_positions
    d['pusher'] = pusher
    return d
