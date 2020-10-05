import numpy as np
import transforms3d
import math
import scipy.linalg

def rotation_matrix_2D(theta):
    R = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    return R

def translation_rotation_to_2D_homogeneous_transform(translation, # np.array shape [2,]
                                                     theta, # float
                                                     ): # return -> np.array shape [3,3]
    T = np.eye(3)
    T[:2, :2] = rotation_matrix_2D(theta)
    T[:2, 2] = translation
    return T

def transform_point_2D(T, # 3 x 3 homogeneous transform matrix in 2D
                       pt, # np.array shape [2,]
                       ): # np.array shape [2,]

    pt_homog = np.ones(3)
    pt_homog[:2] = pt
    pt_transformed = np.matmul(T, pt_homog)[:2]
    return pt_transformed

def transform_points_2D(T, # 3 x 3 homogeneous transform matrix in 2D
                       pts, # np.array shape [N, 2]
                       ): # np.array shape [N, 2]

    N = pts.shape[0]
    pts_homog = np.ones([3, N])
    pts_homog[:2,:] = pts.transpose()

    # shape [3,N]
    pts_transform_homog = np.matmul(T, pts_homog)

    pts_transform = pts_transform_homog.transpose()[:,:2]
    return pts_transform

def transform_points_3D(transform, points):
    """
    :param transform: homogeneous transform
    :type transform:  4 x 4 numpy array
    :param points:
    :type points: numpy array, [N,3]
    :return: numpy array [N,3]
    :rtype:
    """

    N = points.shape[0]
    points_homog = np.append(np.transpose(points), np.ones([1,N]), axis=0) # shape [4,N]
    transformed_points_homog = transform.dot(points_homog) # [4, N]

    transformed_points = np.transpose(transformed_points_homog[0:3, :]) # shape [N, 3]
    return transformed_points


def matrix_to_dict(T, # 4 x 4 homogeneous transform matric
                   ): # dict with position/quaternion

    pos = T[0:3, 3]
    quat = transforms3d.quaternions.mat2quat(T[:3, :3])
    return {'position': pos, 'quaternion': quat}


def transform_from_pose(pos, # xyz
                        quat, # quat [w,x,y,z]
                        ): # 4 x 4 homogeneous transform matrix

    T = np.eye(4)
    T[:3, 3] = pos
    T[:3, :3] = transforms3d.quaternions.quat2mat(quat)

    return T


def transform_from_pose_dict(d, # dict with keys ['position', 'quaternion']
                   ): # 4 x 4 homogeneous transform matrix

    pos = get_position_from_dict(d)
    if isinstance(pos, dict):
        pos = np.array([pos['x'], pos['y'], pos['z']])
    else:
        pos = np.array(pos)

    quat = get_quaterion_from_dict(d)
    if isinstance(quat, dict):
        quat = np.array([quat['w'], quat['x'], quat['y'], quat['z']])
    else:
        quat = np.array(get_quaterion_from_dict(d))

    return transform_from_pose(pos, quat)


def get_position_from_dict(d):
    """

    Extracts position value from dict
    :param d:
    :type d:
    :return:
    :rtype:
    """
    key_options = ['pos', 'position', 'translation']
    for key in key_options:
        if key in d:
            return d[key]

    raise ValueError("position key not found")


def get_quaterion_from_dict(d):
    """
    Extracts quaternion value from dict
    :param d:
    :type d:
    :return:
    :rtype:
    """
    key_options = ['quat', 'quaternion']
    for key in key_options:
        if key in d:
            return d[key]

    raise ValueError("quaternion key not found")

def transform_norm(T):
    """
    Computes the pos/angle error on the transform
    """
    translation = float(np.linalg.norm(T[:3, 3]))
    quat = transforms3d.quaternions.mat2quat(T[:3, :3])
    axis, angle = transforms3d.quaternions.quat2axangle(quat)

    angle = float(abs(angle))


    return {'translation': translation,
            'angle': angle,
            'angle_degrees': float(np.rad2deg(angle)),
            }

def camera_K_matrix_from_FOV(width, # pixels
                             height, # pixels
                             fov_y, # radians
                             ):
    """
    width, height, fov_y are as in
    https://drake.mit.edu/doxygen_cxx/structdrake_1_1geometry_1_1render_1_1_depth_camera_properties.html
    :param d:
    :type d:
    :return: np.ndarray [3,3]
    :rtype:
    """

    # camera center
    cx = width/2.
    cy = height/2.

    # then following https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
    # width/2. = f_y * tan(fov_y/2.)
    # see documentation here https://drake.mit.edu/doxygen_cxx/classdrake_1_1systems_1_1sensors_1_1_camera_info.html
    fy = height/2. * (1 / math.tan(fov_y/2.))

    # assuming that focal length is the same in x and y
    fx = fy

    K = np.array([[fx, 0, cx], [0, fy, cy], [0,0,1]])
    return K

def pinhole_projection(K, # camera intrinsics
                       X_CW, # world to camera transform, 4 x 4 homogeneous transform
                       p_WQ, # point in world frame
                       ):
    """
    Follows notation from https://drake.mit.edu/doxygen_cxx/classdrake_1_1systems_1_1sensors_1_1_camera_info.html
    :return:
    :rtype:
    """
    raise Warning("NOT TESTED")

    # in camera frame
    p_WQ_homog = np.append(p_WQ, 1)
    xyz = np.matmul(K, np.matmul(X_CW[:3,:], p_WQ_homog))

    u = xyz[0]/xyz[2]
    v = xyz[1]/xyz[2]

    return np.array([u,v])

def pinhole_unprojection(K, # camera intrinsics matrix
                         X_WC, # 4 x 4 homogeneous transform, camera to world
                         uv, # uv pixel location
                         depth, # depth in meters
                         ):
    """
    Unproject to world frame
    :param K:
    :type K:
    :param X_WC:
    :type X_WC:
    :param uv:
    :type uv:
    :param depth:
    :type depth:
    :return:
    :rtype:
    """

    raise Warning("NOT TESTED")

    K_inv = np.linalg.inv(K)
    xyz = depth * np.array([uv[0], uv[1], 1])
    p_WQ = np.matmul(X_WC[:3, :], np.matmul(K_inv, xyz))
    return p_WQ


def procrustes_alignment(A, # [M, N] numpy array
                         B, # [M, N] numpy array
                         ):
    """
    Compute the homogeneous transform T that aligns
    p to q

    http://nghiaho.com/?page_id=671
    """

    D = A.shape[1]

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    A_bar = A - centroid_A
    B_bar = B - centroid_B

    # note that this solves for R_sp which minimizes
    # (A@R_sp) - B, this is the transpose of the R that we are
    # used to
    R_sp = scipy.linalg.orthogonal_procrustes(A_bar, B_bar)[0]
    R = R_sp.transpose()

    t = centroid_B - R @ centroid_A

    T = np.eye(D+1)
    T[:D,:D] = R
    T[:D, -1] = t

    return T






