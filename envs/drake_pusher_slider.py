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
from key_dynam.envs.drake_sim_diagram_wrapper import DrakeSimDiagramWrapper


class DrakePusherSliderDiagramWrapper(DrakeSimDiagramWrapper):

    def __init__(self,
                 config):
        DrakeSimDiagramWrapper.__init__(self, config)
        self._models = dict()
        self._rigid_bodies = dict()
        self._port_names = dict()
        self._export_port_functions = []

        # sets that keep track of what appears in the mask
        # for doing dense correspondence learning
        self._body_names_to_mask = list()
        self._model_names_to_mask = list()

    @property
    def models(self):
        return self._models


    def get_labels_to_mask(self):
        """
        Returns a set of labels that represent objects of interest in the scene
        Usually this is the object being pushed
        :return: TinyDB database
        :rtype:
        """

        # write sql query
        label_db = self.get_label_db()

        # return all objects that have matching body_name or model_name in the list
        Label = Query()
        labels_to_mask = label_db.search(Label.body_name.one_of(self._body_names_to_mask) |
                        Label.model_name.one_of(self._model_names_to_mask))

        # create new TinyDB with just those
        mask_db = TinyDB(storage=MemoryStorage)
        mask_db.insert_multiple(labels_to_mask)

        return mask_db

    def get_labels_to_mask_list(self):
        """
        Returns a list of labels that should be masked
        :return:
        :rtype:
        """

        mask_db = self.get_labels_to_mask()
        mask_labels = []
        for entry in mask_db:
            mask_labels.append(int(entry['label']))

        return mask_labels


    def add_procedurally_generated_table(self):
        """
        Adds table to the MBP
        :return:
        :rtype:
        """
        mbp = self._mbp

        world_body = mbp.world_body()
        dims = self._config['env']['table']['size']


        # box_shape = Box(1., 2., 3.)
        box_shape = Box(*dims)
        translation = np.zeros(3)
        translation[2] = -dims[2]/2.0
        T_W_B = RigidTransform(p=translation)

        # This rigid body will be added to the world model instance since
        # the model instance is not specified.
        box_body = mbp.AddRigidBody("table", SpatialInertia(
            mass=1.0, p_PScm_E=np.array([0., 0., 0.]),
            G_SP_E=UnitInertia(1.0, 1.0, 1.0)))
        mbp.WeldFrames(world_body.body_frame(), box_body.body_frame(),
                       T_W_B)

        # this is a grey color
        color = np.array(self._config['env']['table']['color'])
        mbp.RegisterVisualGeometry(
            box_body, RigidTransform.Identity(), box_shape, "table_vis",
            color)

        # friction
        friction_params = self._config['env']['table']['coulomb_friction']
        mbp.RegisterCollisionGeometry(
            box_body, RigidTransform.Identity(), box_shape, "table_collision",
            CoulombFriction(*friction_params))

    def add_table_from_sdf(self):
        """
        Adds extra heavy duty table from URDF
        :return:
        :rtype:
        """
        mbp = self._mbp
        parser = self.parser

        world_body = mbp.world_body()
        table_body_idx = parser.AddModelFromFile(paths.extra_heavy_duty_table, "heavy_duty_table")
        base_link_name = 'link'
        base_link = mbp.GetBodyByName(base_link_name, table_body_idx)
        print("type(base_link)", type(base_link))

        # don't weld as it has a <static> tag and thus is already welded
        # mbp.WeldFrames(world_body.body_frame(), base_link.body_frame())

    def add_ycb_model_from_sdf(self, model_name):
        mbp = self._mbp
        parser = self.parser

        ycb_model_idx = parser.AddModelFromFile(paths.ycb_model_paths[model_name], model_name)
        base_link_ycb_model = mbp.GetBodyByName(paths.ycb_model_baselink_names[model_name], ycb_model_idx)

        # DEBUGGING: weld to world so can visualize
        # mbp.WeldFrames(mbp.world_frame(), base_link_ycb_model.body_frame())

        self._models[model_name] = ycb_model_idx

        # export the output port with the object state
        def export_port_func():
            port_state_output_name = model_name + '_state_output'
            self._port_names[port_state_output_name] = port_state_output_name
            state_output_port = mbp.get_state_output_port(ycb_model_idx)
            self.builder.ExportOutput(state_output_port, self._port_names[port_state_output_name] )

        # this should be masked
        self._model_names_to_mask.append(model_name)
        self._export_port_functions.append(export_port_func)


    def add_model_from_sdf(self,
                           model_name,
                           sdf_path):

        mbp = self._mbp
        parser = self.parser

        model_idx = parser.AddModelFromFile(sdf_path, model_name)

        self._models[model_name] = model_idx

        # export the output port with the object state
        def export_port_func():
            port_state_output_name = model_name + '_state_output'
            self._port_names[port_state_output_name] = port_state_output_name
            state_output_port = mbp.get_state_output_port(model_idx)
            self.builder.ExportOutput(state_output_port, self._port_names[port_state_output_name])

        # this should be masked
        self._model_names_to_mask.append(model_name)
        self._export_port_functions.append(export_port_func)

    @property
    def port_names(self):
        return self._port_names

    def add_pusher(self):
        """
        Adds a cylindrical pusher object
        :return:
        :rtype:
        """
        mbp = self.mbp
        parser = self.parser
        pusher_model_idx = parser.AddModelFromFile(paths.xy_slide, "pusher")

        base_link = mbp.GetBodyByName("base", pusher_model_idx)
        # weld it to the world
        mbp.WeldFrames(mbp.world_frame(), base_link.body_frame())

        self.models['pusher'] = pusher_model_idx

        radius = 0.01
        length = 0.1
        pusher_shape = Cylinder(radius, length)

        # This rigid body will be added to the pusher model instance
        world_body = mbp.world_body()
        pusher_body = mbp.AddRigidBody("pusher_body", pusher_model_idx, SpatialInertia(
            mass=10.0, p_PScm_E=np.array([0., 0., 0.]),
            G_SP_E=UnitInertia(1.0, 1.0, 1.0)))

        self._rigid_bodies['pusher'] = pusher_body

        # weld it to EE frame at particular height
        translation = np.zeros(3)
        translation[2] = length / 2.0 + 0.001 # give it a little room for collision stuff
        T_EE_P = RigidTransform(p=translation)
        ee_link = mbp.GetBodyByName("end_effector", pusher_model_idx)
        mbp.WeldFrames(ee_link.body_frame(), pusher_body.body_frame(), T_EE_P)

        # color it green
        color = np.array([0., 1., 0., 1.])
        mbp.RegisterVisualGeometry(
            pusher_body, RigidTransform.Identity(), pusher_shape, "pusher_vis",
            color)
        mbp.RegisterCollisionGeometry(
            pusher_body, RigidTransform.Identity(), pusher_shape, "pusher_collision",
            CoulombFriction(0.9, 0.8))



        def export_port_func():
            # export relevant ports
            actuation_input_port = mbp.get_actuation_input_port(pusher_model_idx)
            state_output_port = mbp.get_state_output_port(pusher_model_idx)

            self._port_names["pusher_state_output"] = "pusher_state_output"
            self._port_names["pusher_actuation_input"] = "pusher_actuation_input"

            self.builder.ExportOutput(state_output_port, self._port_names["pusher_state_output"])
            self.builder.ExportInput(actuation_input_port, self._port_names["pusher_actuation_input"])

        self._export_port_functions.append(export_port_func)

    def get_model_state(self,
                        diagram,  # top level diagram
                        context,  # context gotten from that diagram
                        model_name="sugar_box",  # model name to be queried
                        ):
        # will be 7-DOF FYI

        model_idx = self._models[model_name]
        mbp = self.mbp
        mbp_context = diagram.GetMutableSubsystemContext(mbp, context)
        q = np.copy(mbp.GetPositions(mbp_context, model_idx))
        v = np.copy(mbp.GetVelocities(mbp_context, model_idx))

        return env_utils.drake_position_velocity_to_dict(q, v)

    def get_pusher_state(self,
                         diagram,  # top level diagram
                         context,  # context gotten from that diagram
                         ):


        model_idx = self.models['pusher']
        mbp = self.mbp
        mbp_context = diagram.GetMutableSubsystemContext(mbp, context)
        q = np.copy(mbp.GetPositions(mbp_context, model_idx))
        v = np.copy(mbp.GetVelocities(mbp_context, model_idx))

        # rigid body pose in world
        body = self._rigid_bodies["pusher"]
        T_W_B = mbp.EvalBodyPoseInWorld(mbp_context, body).GetAsMatrix4()



        d = dict()
        d['position'] = q
        d['velocity'] = v
        d['T_W_B'] = transform_utils.matrix_to_dict(T_W_B)


        # if False:
        #     print("type(T_W_B)", type(T_W_B))
        #     print("q.size", q.size)
        #     print("v.size", v.size)

        return d


    def export_ports(self):
        """
        Exports ports, you must call this after Finalize
        :return:
        :rtype:
        """
        assert self.is_finalized()

        for func in self._export_port_functions:
            func()

    def add_pid_controller(self,
                           builder, # DiagramBuilder
                           ):
        """
        Adds PID controller to system
        :return:
        :rtype:
        """
        mbp = self.mbp

        kp = [0, 0]
        kd_gain = 1000
        kd = [kd_gain] * 2
        ki = [0, 0]
        pid = builder.AddSystem(PidController(kp=kp, kd=kd,ki=ki))


        port_name = self.port_names["pusher_state_output"]
        pusher_state_output_port = self.diagram.GetOutputPort(port_name)

        port_name = self.port_names["pusher_actuation_input"]
        pusher_actuation_input_port = self.diagram.GetInputPort(port_name)

        builder.Connect(pusher_state_output_port, pid.get_input_port_estimated_state())
        builder.Connect(pid.get_output_port_control(), pusher_actuation_input_port)

        # make this visible externally
        # will need to be connected up to SliderVelocityController(LeafSystem)
        port_name = "pid_input_port_desired_state"
        port_idx = builder.ExportInput(pid.get_input_port_desired_state(), port_name)

        return {'pid_input_port_name': port_name,
                'pid_input_port_index': port_idx}


class DrakePusherSliderEnv(Env):

    def __init__(self, config, visualize=True):
        Env.__init__(self)

        self._config = config
        self._step_dt = config['env']['step_dt']
        self._model_name = config['env']['model_name']

        # setup DPS wrapper
        self._dps_wrapper = DrakePusherSliderDiagramWrapper(config)

        # setup the simulator
        self._dps_wrapper.add_procedurally_generated_table()
        self._dps_wrapper.add_ycb_model_from_sdf(config['env']['model_name'])
        self._dps_wrapper.add_pusher()
        self._dps_wrapper.finalize()
        self._dps_wrapper.export_ports()

        if visualize:
            self._dps_wrapper.connect_to_meshcat()
            self._dps_wrapper.connect_to_drake_visualizer()

        self._dps_wrapper.add_sensors_from_config(config)
        self._dps_wrapper.build()


        # records port indices
        self._port_idx = dict()

        # add controller and other stuff
        builder = DiagramBuilder()
        self._builder = builder
        # print("type(self._dps_wrapper.diagram)", type(self._dps_wrapper.diagram))
        builder.AddSystem(self._dps_wrapper.diagram)

        # need to connect actuator ports
        # set the controller gains
        self.add_pid_controller()

        diagram = builder.Build()
        self._diagram = diagram
        self._pid_input_port_desired_state = self.diagram.get_input_port(self._port_idx["pid_input_port_desired_state"])

        # setup simulator
        context = diagram.CreateDefaultContext()
        self._simulator = Simulator(diagram, context)
        self._sim_initialized = False
        self._context = context

        # reset env
        self.reset()

    def add_pid_controller(self):
        """
        Adds PID controller to system
        :return:
        :rtype:
        """
        mbp = self._dps_wrapper.mbp
        builder = self._builder

        kp = [0, 0]
        kd_gain = 1000
        kd = [kd_gain] * 2
        ki = [0, 0]
        pid = builder.AddSystem(PidController(kp=kp, kd=kd,ki=ki))


        # pusher state output port
        pusher_model_idx = self._dps_wrapper.models['pusher']
        pusher_state_output_port = mbp.get_state_output_port(pusher_model_idx)
        pusher_actuation_input_port = mbp.get_actuation_input_port(pusher_model_idx)

        port_name = self._dps_wrapper.port_names["pusher_state_output"]
        pusher_state_output_port = self._dps_wrapper.diagram.GetOutputPort(port_name)

        port_name = self._dps_wrapper.port_names["pusher_actuation_input"]
        pusher_actuation_input_port = self._dps_wrapper.diagram.GetInputPort(port_name)

        builder.Connect(pusher_state_output_port, pid.get_input_port_estimated_state())
        builder.Connect(pid.get_output_port_control(), pusher_actuation_input_port)

        # make this visible externally
        # will need to be connected up to SliderVelocityController(LeafSystem)
        port_name = "pid_input_port_desired_state"
        self._port_idx[port_name] = builder.ExportInput(pid.get_input_port_desired_state(), port_name)

    def reset(self):
        """
        Sets up the environment according to the config
        :return:
        :rtype:
        """

        context = self.get_mutable_context()
        # self.simulator.SetDefaultContext(context)
        # self.diagram.SetDefaultContext(context)

        # put the ycb model slightly above the table
        q_ycb_model = np.array([1, 0, 0, 0, 0, 0, 0.1])
        self.set_slider_position(context, q_ycb_model)
        self.set_slider_velocity(context, np.zeros(6))

        # set pusher initial position
        q_pusher = np.array([-0.1, 0.])
        self.set_pusher_position(context, q_pusher)
        self.set_pusher_velocity(context, np.zeros(2))


        # set PID setpoint to zero
        pid_setpoint = np.zeros(4)
        self._pid_input_port_desired_state.FixValue(context, pid_setpoint)

        self._sim_initialized = False


    def set_initial_condition(self, q_slider, q_pusher):
        """
        Sets the initial condition
        :param q_slider:
        :type q_slider:
        :param q_pusher:
        :type q_pusher:
        :return:
        :rtype:
        """
        context = self.get_mutable_context()
        self.reset()  # sets velocities to zero
        self.set_pusher_position(context, q_pusher)
        self.set_slider_position(context, q_slider)
        self.step(np.array([0, 0]), dt=2.0)  # allow box to drop down

    def set_initial_condition_from_dict(self, initial_cond):
        self.set_initial_condition(q_slider=initial_cond['q_slider'],
                                   q_pusher=initial_cond['q_pusher'],)


    def step(self, action, dt=None):

        assert action.size == 2

        if not self._sim_initialized:
            self._simulator.Initialize()
            self._sim_initialized = True

        if dt is None:
            dt = self._step_dt

        # the time to advance to
        t_advance = self.get_context().get_time() + dt
        context = self.get_mutable_context()

        # set the value of the PID controller input port
        pid_setpoint = np.concatenate((np.zeros(2), action))
        self._pid_input_port_desired_state.FixValue(context, pid_setpoint)

        # self._velocity_controller.set_velocity(action)
        self._simulator.AdvanceTo(t_advance)


        # set the sim time from the above context to avoid floating point errors
        obs = self.get_observation()

        # just placeholders for now
        reward = 0.
        done = False
        info = None

        return obs, reward, done, info

    def render(self, mode='rgb'):
        pass

    def get_observation(self,
                        context=None, # the specific context to use, otherwise none
                        ):
        """
        Gets obesrvations (optionally including images)
        :param context:
        :type context:
        :return:
        :rtype:
        """

        if context is None:
            context = self.get_context()

        obs = dict()
        obs["slider"] = self._dps_wrapper.get_model_state(self.diagram, context, self._model_name)
        obs['pusher'] = self._dps_wrapper.get_pusher_state(self.diagram, context)
        obs['sim_time'] = context.get_time()

        if self._config["env"]["observation"]["rgbd"]:
            obs["images"] = self._dps_wrapper.get_image_observations(self.diagram, context)


        return obs

    def get_mutable_context(self):
        return self._simulator.get_mutable_context()

    def get_context(self):
        return self._simulator.get_context()

    def get_rgb_image(self,
                      sensor_name: str,
                      context,
                      ):

        sensor = self._dps_wrapper._rgbd_sensors[sensor_name]
        print("type(sensor)", type(sensor))
        sensor_context = self.diagram.GetSubsystemContext(sensor, context)
        rgb_img_PIL = drake_image_utils.get_color_image(sensor, sensor_context)
        return rgb_img_PIL


    def set_pusher_position(self, context, q):
        """
        Sets the pusher position
        :param context:
        :type context:
        :return:
        :rtype:
        """
        assert q.size == 2

        mbp = self._dps_wrapper.mbp
        mbp_context = self._diagram.GetMutableSubsystemContext(mbp, context)
        mbp.SetPositions(mbp_context, self._dps_wrapper.models['pusher'], q)

    def set_pusher_velocity(self, context, v):
        assert v.size == 2

        mbp = self._dps_wrapper.mbp
        mbp_context = self._diagram.GetMutableSubsystemContext(mbp, context)
        mbp.SetVelocities(mbp_context, self._dps_wrapper.models['pusher'], v)


    def set_slider_position(self, context, q):
        """
        Sets the slider position
        :param context:
        :type context:
        :param q:
        :type q:
        :return:
        :rtype:
        """
        assert q.size == 7

        # check that first 4 are a quaternion
        if abs(np.linalg.norm(q[0:4]) - 1 ) > 1e-3:
            print("q[0:4]", q[0:4])
            print("np.linalg.norm(q[0:4])", np.linalg.norm(q[0:4]))

        assert (abs(np.linalg.norm(q[0:4]) - 1 ) < 1e-3), "q[0:4] is not a valid quaternion"

        # put the ycb model slightly above the table
        mbp = self._dps_wrapper.mbp
        mbp_context = self._diagram.GetMutableSubsystemContext(mbp, context)
        mbp.SetPositions(mbp_context, self._dps_wrapper.models[self._model_name], q)


    def set_slider_velocity(self, context, v):
        assert v.size == 6

        # put the ycb model slightly above the table
        mbp = self._dps_wrapper.mbp
        mbp_context = self._diagram.GetMutableSubsystemContext(mbp, context)
        mbp.SetVelocities(mbp_context, self._dps_wrapper.models[self._model_name], v)


    def set_simulator_state_from_observation_dict(self,
                                                  context,
                                                  d,  # dictionary returned by DrakeSimEpisodeReader
                                                  ):

        # set slider position
        q_slider = d['slider']['position']['raw']
        v_slider = d['slider']['velocity']['raw']
        self.set_slider_position(context, q_slider)
        self.set_slider_velocity(context, v_slider)

        # set pusher position
        q_pusher = d['pusher']['position']
        v_pusher = d['pusher']['velocity']
        self.set_pusher_position(context, q_pusher)
        self.set_pusher_velocity(context, v_pusher)


    def pusher_within_boundary(self, context=None):
        """
        Return true if pusher within boundary of table + tolerance
        :return:
        :rtype:
        """

        if context is None:
            context = self.get_context()

        tol = 0.03 # 3 cm

        pusher_state = self._dps_wrapper.get_pusher_state(self.diagram, context)
        pos = pusher_state['position'] # x,y

        dims = np.array(self.config['env']['table']['size'])
        high = dims[0:2]/2.0 - tol
        low = -high

        in_range = ((low < pos) & (pos < high)).all()
        return in_range



    def slider_within_boundary(self, context=None):
        """
        Return true if slider within boundary of table + tolerance
        :return:
        :rtype:
        """

        if context is None:
            context = self.get_context()

        tol = 0.03 # 3 cm

        slider_state = self._dps_wrapper.get_model_state(self.diagram, context, self._model_name)
        pos = slider_state['position']['translation']

        dims = np.array(self.config['env']['table']['size'])

        high = np.zeros(3)
        low = np.zeros(3)
        high[0:2] = dims[0:2]/2.0 - tol
        high[2] = 100 # just large number so that it's nonbinding

        low[0:2] = -high[0:2]
        low[2] = 0

        in_range = ((low < pos) & (pos < high)).all()
        return in_range

    def state_is_valid(self): # checks if state is valid
        return (self.slider_within_boundary() and self.pusher_within_boundary())

    def get_metadata(self):
        """
        Returns a bunch of data that is useful to be logged
        :return: dict()
        :rtype:
        """

        label_db = self._dps_wrapper.get_label_db()
        mask_db = self._dps_wrapper.get_labels_to_mask()

        # note saving TinyDB object with pickle causes error on load
        # this is a workaround, can easily re-create TinyDB object
        # when we load this
        return {'label_db': label_db.all(),
                'mask_db': mask_db.all()}

    def get_labels_to_mask_list(self):
        """
        Returns a list of labels that should be masked
        :return:
        :rtype:
        """

        return self._dps_wrapper.get_labels_to_mask_list()

    @property
    def simulator(self):
        return self._simulator

    @property
    def diagram(self):
        return self._diagram

    @property
    def config(self):
        return self._config

    @property
    def diagram_wrapper(self):
        return self._dps_wrapper


    def camera_pose(self,
                    camera_name): # [4,4] homogeneous transform

        pose_dict = self.config['env']['rgbd_sensors']['sensor_list'][camera_name]
        return transform_utils.transform_from_pose_dict(pose_dict)

    def get_zero_action(self):
        """
        Returns a zero action
        :return:
        :rtype:
        """
        return np.zeros(2)

    def camera_K_matrix(self,
                        camera_name): # np.array [4,4]
        return DrakePusherSliderEnv.camera_K_matrix_from_config(self.config, camera_name)

    @staticmethod
    def camera_K_matrix_from_config(config,
                                    camera_name): # np.array [4,4]

        sensor_dict = config['env']['rgbd_sensors']['sensor_list'][camera_name]
        width = sensor_dict['width']
        height = sensor_dict['height']
        fov_y = sensor_dict['fov_y']

        return transform_utils.camera_K_matrix_from_FOV(width, height, fov_y)




    @staticmethod
    def object_position_from_observation(obs):
        q_slider = obs['slider']['position']['raw']
        return env_utils.drake_position_to_pose(q_slider)