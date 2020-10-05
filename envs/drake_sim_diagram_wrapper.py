from future.utils import iteritems
from math import *
import numpy as np
import matplotlib.pyplot as plt
from gym import Env
import os
from tinydb import Query
from tinydb import TinyDB
from tinydb.storages import MemoryStorage



from pydrake.common.eigen_geometry import Quaternion
from pydrake.geometry import SceneGraph
from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.common.eigen_geometry import Quaternion
from pydrake.multibody.all import MultibodyPlant, Parser, SpatialForce
from pydrake.multibody.plant import ExternallyAppliedSpatialForce, VectorExternallyAppliedSpatialForced
from pydrake.systems.controllers import PidController
from pydrake.systems.framework import (AbstractValue, BasicVector, Diagram,
                                       DiagramBuilder, LeafSystem)
from pydrake.systems.meshcat_visualizer import MeshcatVisualizer
from pydrake.systems.sensors import RgbdSensor, Image, PixelType, PixelFormat
from pydrake.geometry.render import DepthCameraProperties, MakeRenderEngineVtk, RenderEngineVtkParams
from pydrake.systems.analysis import Simulator
from pydrake.systems.primitives import ConstantVectorSource
from pydrake.geometry import Box, Cylinder
from pydrake.multibody.tree import SpatialInertia, UnitInertia
from pydrake.multibody.plant import (
    AddMultibodyPlantSceneGraph, CoulombFriction)
from pydrake.geometry import ConnectDrakeVisualizer
from pydrake.multibody.tree import BodyIndex

from key_dynam.utils.utils import get_project_root
from key_dynam.utils import drake_image_utils as drake_image_utils
import key_dynam.utils.paths as paths
import key_dynam.envs.utils as env_utils
from key_dynam.utils import transform_utils
from key_dynam.utils import drake_utils



class DrakeSimDiagramWrapper:
    """
    Helper class that wraps a Drake Diagram.
    Has utilities for adding cameras, etc.
    """

    def __init__(self,
                 config):

        self._config = config
        builder = DiagramBuilder()
        dt = config['env']['mbp_dt']
        mbp, sg = AddMultibodyPlantSceneGraph(builder, MultibodyPlant(dt), SceneGraph())
        parser = Parser(mbp, sg)

        # renderer_params = RenderEngineVtkParams()
        # renderer_name = "vtk_renderer"
        # sg.AddRenderer(renderer_name, MakeRenderEngineVtk(renderer_params))

        self._mbp = mbp
        self._sg = sg
        self._parser = parser
        self._builder = builder
        self._rgbd_sensors = dict()
        self._finalized = False
        self._built = False
        self._diagram = None

        # add rendered
        # Add renderer to scene
        renderer_params = RenderEngineVtkParams()
        self._renderer_name = "vtk_renderer"
        sg.AddRenderer(self._renderer_name, MakeRenderEngineVtk(renderer_params))

    @property
    def mbp(self):
        return self._mbp

    @property
    def sg(self):
        return self._sg

    @property
    def builder(self):
        return self._builder

    @property
    def diagram(self):
        return self._diagram

    @property
    def parser(self):
        return self._parser

    @property
    def rgbd_sensors(self):
        return self._rgbd_sensors

    def add_sensor(self, camera_name, sensor_config):
        """
        Adds Rgbd sensor to the diagram
        :param camera_name:
        :type camera_name:
        :param sensor_config:
        :type sensor_config:
        :return:
        :rtype:
        """
        builder = self.builder
        sg = self.sg

        width = sensor_config['width']
        height = sensor_config['height']
        fov_y = sensor_config['fov_y']
        z_near = sensor_config['z_near']
        z_far = sensor_config['z_far']
        renderer_name = self._renderer_name

        depth_camera_properties = DepthCameraProperties(
            width=width, height=height, fov_y=fov_y, renderer_name=renderer_name, z_near=z_near, z_far=z_far)
        parent_frame_id = sg.world_frame_id()

        # This is in right-down-forward convention
        pos = np.array(sensor_config['pos'])
        quat_eigen = Quaternion(sensor_config['quat'])

        # camera to world transform
        T_W_camera = RigidTransform(quaternion=quat_eigen, p=pos)

        # add camera system
        camera = builder.AddSystem(
            RgbdSensor(parent_frame_id, T_W_camera, depth_camera_properties, show_window=False))
        builder.Connect(sg.get_query_output_port(),
                        camera.query_object_input_port())

        self._rgbd_sensors[camera_name] = camera

    def finalize(self):
        """
        Finalize the MBP and connect it to the
        scene graph
        :return:
        :rtype:
        """
        mbp = self.mbp
        mbp.Finalize()
        self._finalized = True

    def is_finalized(self):
        return self._finalized

    def is_built(self):
        return self._built

    def connect_to_meshcat(self):
        """
        Connects system to meshcat visualizer
        :return:
        :rtype:
        """
        assert self.is_finalized()
        builder = self.builder
        sg = self.sg

        zmq_url = "default"
        meshcat = builder.AddSystem(MeshcatVisualizer(sg, zmq_url=zmq_url))
        builder.Connect(sg.get_pose_bundle_output_port(),
                        meshcat.get_input_port(0))

    def connect_to_drake_visualizer(self):

        builder = self.builder
        sg = self.sg
        ConnectDrakeVisualizer(builder, sg, sg.get_pose_bundle_output_port())



    def build(self):
        assert self.is_finalized()
        self._built = True

        builder = self.builder
        self._diagram = builder.Build()

    def add_sensors_from_config(self,
                                config: dict # global config
                                ):

        if not config['env']['rgbd_sensors']["enabled"]:
            return

        for camera_name, sensor_config in iteritems(config['env']['rgbd_sensors']['sensor_list']):
            self.add_sensor(camera_name, sensor_config)

    def get_rgb_image(self,
                      sensor_name: str,
                      diagram, # diagram that context corresponds to
                      context,
                      ):

        sensor = self._rgbd_sensors[sensor_name]
        sensor_context = diagram.GetSubsystemContext(sensor, context)
        rgb_img_PIL = drake_image_utils.get_color_image(sensor, sensor_context)
        return rgb_img_PIL

    def get_image_observations_single_sensor(self,
                                             sensor_name,
                                             diagram,
                                             context,
                                             ):


        sensor = self.rgbd_sensors[sensor_name]
        sensor_context = diagram.GetSubsystemContext(sensor, context)
        rgb = drake_image_utils.get_color_image(sensor, sensor_context)
        depth_32F = drake_image_utils.get_depth_image_32F(sensor, sensor_context)
        depth_16U = drake_image_utils.get_depth_image_16U(sensor, sensor_context)
        label = drake_image_utils.get_label_image(sensor, sensor_context)

        return {'rgb': rgb,
                'depth_32F': depth_32F,
                'depth_16U': depth_16U,
                'label': label}


    def get_image_observations(self,
                               diagram,
                               context):

        d = dict()
        for sensor_name in self.rgbd_sensors:
            d[sensor_name] = self.get_image_observations_single_sensor(sensor_name, diagram, context)


        return d

    def get_label_db(self):
        """
        Builds database that associates bodies and labels
        :return: TinyDB database
        :rtype:
        """

        return drake_utils.get_label_db(self.mbp)

