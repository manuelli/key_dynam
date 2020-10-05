import numpy as np
import transforms3d

from key_dynam.utils import transform_utils

def compute_pose_error(obs, obs_goal):
    """
    Returns a dict with position and angle errors
    for the slider
    :param obs:
    :type obs:
    :param obs_goal:
    :type obs_goal:
    :return:
    :rtype:
    """

    T_W_goal = None
    T_W_s = None
    try:
        T_W_goal = transform_utils.transform_from_pose_dict(obs_goal['slider']['position'])
        T_W_S = transform_utils.transform_from_pose_dict(obs['slider']['position'])
    except KeyError:
        T_W_goal = transform_utils.transform_from_pose_dict(obs_goal['object']['position'])
        T_W_S = transform_utils.transform_from_pose_dict(obs['object']['position'])

    T_goal_S = np.linalg.inv(T_W_goal) @ T_W_S

    pos_err = np.linalg.norm(T_goal_S[:3, 3])
    axis, angle = transforms3d.axangles.mat2axangle(T_goal_S[:3, :3])
    angle_error = abs(angle)
    angle_error_deg = np.rad2deg(angle_error)

    data = {'position_error': pos_err,
            'angle_error': angle_error,
            'angle_error_degrees': angle_error_deg,
            }

    return data