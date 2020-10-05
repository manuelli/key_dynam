import numpy as np

# meshcat
import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf

from key_dynam.utils import transform_utils


def make_default_visualizer_object():
    return meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")


def default_zmq_url():
    return "tcp://127.0.0.1:6000"


def visualize_points(vis,
                     name,
                     pts,  # [N,3]
                     color=None,  # [3,] array of float in [0,1]
                     size=0.001,  # size of the points
                     T=None,  # T_world_pts transform
                     ):

    if color is not None:
        N, _ = pts.shape
        color = 1.0 * np.ones([N, 3]) * np.array(color)
        color = color.transpose()

    geom = g.Points(
        g.PointsGeometry(pts.transpose(), color=color),
        g.PointsMaterial(size=size)
    )

    vis[name].set_object(geom)

    if T is not None:
        vis[name].set_transform(T)



