import time
import numpy as np

# drake
from pydrake.math import RigidTransform
from pydrake.multibody.tree import SpatialInertia, UnitInertia
from pydrake.multibody.plant import (
    AddMultibodyPlantSceneGraph, CoulombFriction)
from pydrake.geometry import Box, Cylinder

# key_dynam
from key_dynam.dataset.online_episode_reader import OnlineEpisodeReader
from key_dynam.dataset.episode_container import EpisodeContainer
from key_dynam.utils.utils import get_current_YYYY_MM_DD_hh_mm_ss_ms
from key_dynam.utils import transform_utils

def get_body_state_as_dict(body):
    """
    Gets the state (position, velocity) of a pymunk body,
    packs results into a dict
    :param body:
    :type body:
    :return:
    :rtype:
    """
    d = dict()
    d["position"] = np.array(body.position)
    d["velocity"] = np.array(body.velocity)

    d["angle"] = np.array(body.angle)
    d["angular_velocity"] = np.array(body.angular_velocity)

    return d


def set_body_state_from_dict(body, state_dict):
    """
    Sets body properties based on keys in dict
    :param body:
    :type body:
    :param state_dict:
    :type state_dict:
    :return:
    :rtype:
    """
    keys = ["position", 'velocity', 'angle', 'angular_velocity']

    for key in keys:
        if key in state_dict:
            setattr(body, key, state_dict[key])


def zero_body_velocity(body):
    body.velocity = [0, 0]
    body.angular_velocity = 0


def drake_position_to_dict(q,  # np.array shape [7,]
                           ):
    assert q.size == 7

    quat = q[0:4]
    pos = q[4:]

    d = {'translation': pos,
         'quaternion': quat,
         'raw': q,
         }

    return d


def drake_velocity_to_dict(v,  # np.array shape [6,]
                           ):
    assert v.size == 6
    angular = v[0:3]
    linear = v[3:]

    d = {'linear': linear,
         'angular': angular,
         'raw': v,
         }

    return d


def drake_position_velocity_to_dict(q, v):
    d = dict()
    d['position'] = drake_position_to_dict(q)
    d['velocity'] = drake_velocity_to_dict(v)

    return d

def drake_position_to_pose(q):
    assert len(q) == 7 # [quat, pos]

    quat = q[0:4]
    pos = q[4:]

    return transform_utils.transform_from_pose(pos, quat)


def rollout_action_sequence(env, # gym env
                            action_seq,  # [N, action_dim]
                            ):
    """
    Rollout the action sequence using the simulator
    Record the observations
    """

    episode_reader = OnlineEpisodeReader()

    obs_list = []
    N = action_seq.shape[0]
    for i in range(N):
        action = action_seq[i]
        obs, reward, done, info = env.step(action)
        episode_reader.add_observation_action(obs, action)
        obs_list.append(obs)

    return {'observations': obs_list,
            'episode_reader': episode_reader,
            }

def binary_mask_from_label_image(label_img, # np.array [H, W]
                                 mask_labels, # list[int]
                                 ): # np.array [H, W] binary
    """
    Given a binary mask image, set all the pixels matching a value in mask_labels
    to 1
    """

    mask_img = np.zeros_like(label_img)

    for label_val in mask_labels:
        mask_img[label_img == label_val] = 1

    return mask_img


def add_procedurally_generated_table(mbp, # multi-body plant
                                     table_config, # dict
                                     ):
    """
    Adds a procedurally generated table to the scene

    param table_config: dict with keys ['size', 'color']

    """
    world_body = mbp.world_body()
    dims = table_config['size']

    # box_shape = Box(1., 2., 3.)
    box_shape = Box(*dims)
    translation = np.zeros(3)
    translation[2] = -dims[2] / 2.0
    T_W_B = RigidTransform(p=translation)

    # This rigid body will be added to the world model instance since
    # the model instance is not specified.
    box_body = mbp.AddRigidBody("table", SpatialInertia(
        mass=1.0, p_PScm_E=np.array([0., 0., 0.]),
        G_SP_E=UnitInertia(1.0, 1.0, 1.0)))
    mbp.WeldFrames(world_body.body_frame(), box_body.body_frame(),
                   T_W_B)

    # this is a grey color
    color = np.array(table_config['color'])
    mbp.RegisterVisualGeometry(
        box_body, RigidTransform.Identity(), box_shape, "table_vis",
        color)

    # friction
    friction_params = table_config['coulomb_friction']
    mbp.RegisterCollisionGeometry(
        box_body, RigidTransform.Identity(), box_shape, "table_collision",
        CoulombFriction(*friction_params))


def set_model_position(diagram, # drake diagram (top level one)
                       context, # mutable context
                       mbp, # multibody plant
                       model_name,
                       q, # the position to set
                       ):

    """
    Sets the model positions appropriately
    """

    mbp_context = diagram.GetMutableSubsystemContext(mbp, context)
    model_instance = mbp.GetModelInstanceByName(model_name)
    mbp.SetPositions(mbp_context, model_instance, q)


def set_model_velocity(diagram, # drake diagram (top level one)
                       context, # mutable context
                       mbp, # multibody plant
                       model_name,
                       v, # the velocity to set
                       ):

    """
    Sets the model velocities appropriately
    """

    mbp_context = diagram.GetMutableSubsystemContext(mbp, context)
    model_instance = mbp.GetModelInstanceByName(model_name)
    mbp.SetVelocities(mbp_context, model_instance, v)