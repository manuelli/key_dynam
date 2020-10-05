import numpy as np
from gym import Env
import os

# drake
from pydrake.systems.framework import (AbstractValue, BasicVector, Diagram,
                                       DiagramBuilder, LeafSystem)
from pydrake.systems.analysis import Simulator

# key_dynam
from key_dynam.envs.drake_sim_diagram_wrapper import DrakeSimDiagramWrapper
from key_dynam.envs.drake_pusher_slider import DrakePusherSliderDiagramWrapper
from key_dynam.envs import utils as env_utils
from key_dynam.utils import drake_image_utils as drake_image_utils
from key_dynam.utils.paths import LARGE_SIM_ASSETS_ROOT
from key_dynam.utils.utils import get_current_YYYY_MM_DD_hh_mm_ss_ms, get_data_root
from key_dynam.dataset.episode_container import EpisodeContainer
from key_dynam.sim_assets.sdf_helper import SDFHelper
from key_dynam.utils import transform_utils


class DrakeMugsEnv(Env):

    def __init__(self, config, visualize=True):
        Env.__init__(self)

        self._config = config
        self._step_dt = config['env']['step_dt']
        self._model_name = "mug"

        # setup DPS wrapper
        self._diagram_wrapper = DrakePusherSliderDiagramWrapper(config)

        # setup the simulator
        # add procedurally generated table
        env_utils.add_procedurally_generated_table(self.diagram_wrapper.mbp,
                                                   config['env']['table']),

        self.diagram_wrapper.add_pusher()

        self.add_object_model()

        self.diagram_wrapper.finalize()
        self.diagram_wrapper.export_ports()

        if visualize:
            self.diagram_wrapper.connect_to_meshcat()
            self.diagram_wrapper.connect_to_drake_visualizer()

        self.diagram_wrapper.add_sensors_from_config(config)
        self.diagram_wrapper.build()

        # records port indices
        self._port_idx = dict()

        # add controller and other stuff
        builder = DiagramBuilder()
        self._builder = builder
        # print("type(self.diagram_wrapper.diagram)", type(self.diagram_wrapper.diagram))
        builder.AddSystem(self.diagram_wrapper.diagram)

        # need to connect actuator ports
        # set the controller gains
        pid_data = self.diagram_wrapper.add_pid_controller(builder=builder)
        self._port_idx["pid_input_port_desired_state"] = pid_data['pid_input_port_index']

        diagram = builder.Build()
        self._diagram = diagram
        self._pid_input_port_desired_state = self._diagram.get_input_port(
            self._port_idx["pid_input_port_desired_state"])

        # setup simulator
        context = diagram.CreateDefaultContext()
        self._simulator = Simulator(self._diagram, context)
        self._sim_initialized = False
        self._context = context

        # reset env
        self.reset()

    @property
    def diagram_wrapper(self):
        return self._diagram_wrapper

    def add_object_model(self):
        # add mug
        # sdf_path = "anzu_mugs/big_mug-corelle_mug-6.sdf"
        # sdf_path = "anzu_mugs/big_mug-small_mug-0.sdf"
        # sdf_path = "anzu_mugs/corelle_mug-small_mug-8.sdf"
        # sdf_path = "manual_babson_11oz_mug/manual_babson_11oz_mug.sdf"
        # sdf_path = os.path.join(LARGE_SIM_ASSETS_ROOT, sdf_path)
        #

        self._object_name = "mug"

        sdf_file_fullpath = os.path.join(get_data_root(), self.config['env']['model']['sdf'])
        model_color = self.config['env']['model']['color']

        # output_dir = "/home/manuelli/data/key_dynam/sandbox/sdf_helper"
        output_dir = None
        sdf_data = SDFHelper.create_sdf_specific_color(sdf_file_fullpath=sdf_file_fullpath,
                                                       color=model_color,
                                                       output_dir=output_dir)

        self.diagram_wrapper.add_model_from_sdf(self._object_name, sdf_data['sdf_file'])

    def reset(self):
        """
        Sets up the environment according to the config
        :return:
        :rtype:
        """

        context = self.get_mutable_context()
        # self.simulator.SetDefaultContext(context)
        # self.diagram.SetDefaultContext(context)

        # # put the ycb model slightly above the table
        q_object = np.array([1, 0, 0, 0, 0, 0, 0.1])
        self.set_object_position(context, q_object)
        # self.set_slider_position(context, q_ycb_model)
        # self.set_slider_velocity(context, np.zeros(6))

        # set pusher initial position
        q_pusher = np.array([-0.1, 0.])
        self.set_pusher_position(context, q_pusher)
        self.set_pusher_velocity(context, np.zeros(2))

        # set PID setpoint to zero
        pid_setpoint = np.zeros(4)
        self._pid_input_port_desired_state.FixValue(context, pid_setpoint)

        self._sim_initialized = False

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
                        context=None,  # the specific context to use, otherwise none
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
        obs['slider'] = self.diagram_wrapper.get_model_state(self.diagram, context, self._object_name)
        obs['object'] = obs['slider']
        obs['pusher'] = self.diagram_wrapper.get_pusher_state(self.diagram, context)
        obs['sim_time'] = context.get_time()

        if self._config["env"]["observation"]["rgbd"]:
            obs["images"] = self.diagram_wrapper.get_image_observations(self.diagram, context)

        return obs

    def get_mutable_context(self):
        return self._simulator.get_mutable_context()

    def get_context(self):
        return self._simulator.get_context()

    def get_rgb_image(self,
                      sensor_name: str,
                      context,
                      ):

        sensor = self.diagram_wrapper._rgbd_sensors[sensor_name]
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

        mbp = self.diagram_wrapper.mbp
        mbp_context = self._diagram.GetMutableSubsystemContext(mbp, context)
        mbp.SetPositions(mbp_context, self.diagram_wrapper.models['pusher'], q)

    def set_pusher_velocity(self, context, v):
        assert v.size == 2

        mbp = self.diagram_wrapper.mbp
        mbp_context = self._diagram.GetMutableSubsystemContext(mbp, context)
        mbp.SetVelocities(mbp_context, self.diagram_wrapper.models['pusher'], v)

    def set_object_position(self, context, q):

        assert q.size == 7

        # check that first 4 are a quaternion
        if abs(np.linalg.norm(q[0:4]) - 1) > 1e-2:
            print("q[0:4]", q[0:4])
            print("np.linalg.norm(q[0:4])", np.linalg.norm(q[0:4]))

        assert (abs(np.linalg.norm(q[0:4]) - 1) < 1e-2), "q[0:4] is not a valid quaternion"

        mbp = self.diagram_wrapper.mbp
        env_utils.set_model_position(diagram=self.diagram,
                                     context=context,
                                     mbp=mbp,
                                     model_name=self._object_name,
                                     q=q)

    def set_object_velocity(self, context, v):

        assert v.size == 6

        mbp = self.diagram_wrapper.mbp
        env_utils.set_model_velocity(diagram=self.diagram,
                                     context=context,
                                     mbp=mbp,
                                     model_name=self._object_name,
                                     v=v)

    def get_zero_action(self):
        """
        Returns a zero action
        :return:
        :rtype:
        """
        return np.zeros(2)

    @property
    def simulator(self):
        return self._simulator

    @property
    def diagram(self):
        return self._diagram

    def camera_pose(self,
                    camera_name): # [4,4] homogeneous transform

        pose_dict = self.config['env']['rgbd_sensors']['sensor_list'][camera_name]
        return transform_utils.transform_from_pose_dict(pose_dict)

    @property
    def config(self):
        return self._config

    def get_metadata(self):
        """
        Returns a bunch of data that is useful to be logged
        :return: dict()
        :rtype:
        """

        label_db = self.diagram_wrapper.get_label_db()
        mask_db = self.diagram_wrapper.get_labels_to_mask()

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

        return self.diagram_wrapper.get_labels_to_mask_list()

    def pusher_within_boundary(self, context=None):
        """
        Return true if pusher within boundary of table + tolerance
        :return:
        :rtype:
        """

        if context is None:
            context = self.get_context()

        tol = 0.03  # 3 cm

        pusher_state = self.diagram_wrapper.get_pusher_state(self.diagram, context)
        pos = pusher_state['position']  # x,y

        dims = np.array(self.config['env']['table']['size'])
        high = dims[0:2] / 2.0 - tol
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

        tol = 0.03  # 3 cm

        slider_state = self.diagram_wrapper.get_model_state(self.diagram, context, self._model_name)
        pos = slider_state['position']['translation']

        dims = np.array(self.config['env']['table']['size'])

        high = np.zeros(3)
        low = np.zeros(3)
        high[0:2] = dims[0:2] / 2.0 - tol
        high[2] = 100  # just large number so that it's nonbinding

        low[0:2] = -high[0:2]
        low[2] = 0

        in_range = ((low < pos) & (pos < high)).all()
        return in_range

    def camera_K_matrix(self,
                        camera_name): # np.array [4,4]
        return DrakeMugsEnv.camera_K_matrix_from_config(self.config, camera_name)

    @staticmethod
    def camera_K_matrix_from_config(config,
                                    camera_name): # np.array [4,4]

        sensor_dict = config['env']['rgbd_sensors']['sensor_list'][camera_name]
        width = sensor_dict['width']
        height = sensor_dict['height']
        fov_y = sensor_dict['fov_y']

        return transform_utils.camera_K_matrix_from_FOV(width, height, fov_y)

    def set_simulator_state_from_observation_dict(self,
                                                  context,
                                                  d,  # dictionary returned by DrakeSimEpisodeReader
                                                  ):

        # set slider position
        q_slider = d['slider']['position']['raw']
        v_slider = d['slider']['velocity']['raw']
        self.set_object_position(context, q_slider)
        self.set_object_velocity(context, v_slider)

        # set pusher position
        q_pusher = d['pusher']['position']
        v_pusher = d['pusher']['velocity']
        self.set_pusher_position(context, q_pusher)
        self.set_pusher_velocity(context, v_pusher)


    def set_initial_condition_from_dict(self, initial_cond):
        context = self.get_mutable_context()
        self.reset()  # sets velocities to zero
        self.set_pusher_position(context, initial_cond['q_pusher'])
        self.set_object_position(context, initial_cond['q_object'])
        self.step(np.array([0, 0]), dt=10.0)  # allow box to drop down

    def state_is_valid(self): # checks if state is valid
        return (self.slider_within_boundary() and self.pusher_within_boundary())


def collect_single_episode(env,  # gym env
                           action_seq,  # [N, action_dim]
                           episode_name=None):
    """
    Rollout the action sequence using the simulator
    Record the observations
    """

    if episode_name is None:
        episode_name = get_current_YYYY_MM_DD_hh_mm_ss_ms()

    episode_container = EpisodeContainer()
    episode_container.set_config(env.config)
    episode_container.set_name(episode_name)
    episode_container.set_metadata(env.get_metadata())

    obs_list = []
    N = action_seq.shape[0]
    obs_prev = env.get_observation()
    for i in range(N):
        action = action_seq[i]
        obs, reward, done, info = env.step(action)
        episode_container.add_obs_action(obs_prev, action)
        obs_prev = obs

        # terminate if outside boundary
        if not env.slider_within_boundary():
            print('slider outside boundary, terminating episode')
            break

        if not env.pusher_within_boundary():
            print('pusher ouside boundary, terminating episode')
            break

    return {'episode_container': episode_container}
