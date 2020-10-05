import time
import copy
import os
import numpy as np

import torch
from enum import Enum

from key_dynam.utils import utils, torch_utils
from key_dynam.controller.zmq_utils import ZMQServer
from key_dynam.dataset.online_episode_reader import OnlineEpisodeReader
from key_dynam.dataset.mpc_dataset import DynamicsModelInputBuilder
from key_dynam.controller.plan_container import PlanContainer
from key_dynam.dynamics.models_dy import rollout_model, rollout_action_sequences, get_object_and_robot_state_indices
from key_dynam.planner import utils as planner_utils
from key_dynam.utils import meshcat_utils
from key_dynam.experiments.exp_22_push_box_hardware.visualize_dynamics_model import \
    visualize_model_prediction_single_timestep

# dense correspondence
from dense_correspondence_manipulation.utils import meshcat_utils as pdc_meshcat_utils
from dense_correspondence_manipulation.utils.constants import DEPTH_IM_SCALE

PLAN_MSG_FILE = os.path.join(utils.get_project_root(), 'sandbox', 'plan_msg.p')
COMPUTE_CONTROL_ACTION_MSG_FILE = os.path.join(utils.get_project_root(), 'sandbox', 'compute_control_action_msg.p')

SAVE_MESSAGES = True


class ControllerState(Enum):
    STOPPED = 0
    PLAN_READY = 1
    RUNNING = 2


class Controller:
    """
    Controller for running in closed loop
    """

    def __init__(self):
        pass

    # enter data into "MPCOnlineEpisodeReader" . . .
    def compute_control_action(self, data):
        pass


"""
MESSAGE DEFINITIONS

Response message types

- STOP (this abort)
- CONTROL_ACTION (this means apply that control action)
"""


class ZMQController(Controller):
    """
    Acts as a ZMQ server
    """

    def __init__(self,
                 config=None,
                 port=5555,
                 zmq_enabled=True,
                 model_dy=None,  # dynamics model
                 action_function=None,
                 observation_function=None,
                 visual_observation_function=None,
                 planner=None,
                 debug=False,
                 camera_info=None,  # just for debugging purposes
                 ):

        super().__init__()

        if zmq_enabled:
            self._zmq_server = ZMQServer(port=port)

        self._config = config
        self._n_history = config['train']['n_history']
        self._model_dy = model_dy
        self._debug = debug
        self._planner = planner
        self._planner_config_copy =  copy.deepcopy(planner.config)

        self._state = None  # the internal state of the controller
        self._state = ControllerState.STOPPED
        self._state_dict = None  # for storing stateful information

        # used for the DynamicsModelInputBuilder
        self._action_function = action_function
        self._observation_function = observation_function
        self._visual_observation_function = visual_observation_function

        self._index_dict = get_object_and_robot_state_indices(config)
        self._object_indices = torch.LongTensor(self._index_dict['object_indices'])

        self._camera_info = camera_info

    def load_models(self):
        """
        Load the dynamics and vision models
        :return:
        :rtype:
        """
        pass

    def run(self):
        """
        Loop waiting for a request from the ZMQ Client
        """
        print("Waiting for request from client . . . ")

        while True:
            #  Wait for next request from client

            # blocks until we get a request
            # data = self._zmq_server.recv_data()
            msg = self._zmq_server.recv_data()

            resp = {'type': 'ERROR',
                    'data': 'ERROR',
                    }
            try:
                # dispatch based on type of message
                if msg['type'] == "PLAN":
                    resp = self._on_plan_msg(msg)
                elif msg['type'] == "COMPUTE_CONTROL_ACTION":
                    resp = self._on_compute_control_action(msg)
                elif msg['type'] == "RESET":
                    resp = self._on_reset_msg(msg)
                elif msg['type'] == "SAVE":
                    print("got SAVE message")
                    self.save_data()
                    resp = self.make_default_success_response_message()
            finally:
                # always respond to the zmq server
                self._zmq_server.send_data(resp)

    def _on_plan_msg(self, msg):
        """
        Received a plan

        - apply the 'visual_observation_function' to each image in the plan
        - store it in the same data struct
        - (maybe) make a 'Plan' object? this is probably wise
        - setup a new controller? maybe just re-use the old one?


        maybe wrap in try/finally block so we don't leave the python side hanging?
        :param msg:
        :type msg:
        :return:
        :rtype:
        """

        assert self._state == ControllerState.STOPPED

        print("\n\n----- RECEIVED PLAN ------")

        plan_data = msg['data']
        msg_data = msg['data']

        print("len(plan_data)", len(plan_data))

        self._state_dict = dict()
        sd = self._state_dict

        episode = OnlineEpisodeReader(no_copy=True)
        sd['episode'] = episode

        input_builder = DynamicsModelInputBuilder(observation_function=self._observation_function,
                                                  visual_observation_function=self._visual_observation_function,
                                                  action_function=self._action_function,
                                                  episode=episode)

        plan_container = PlanContainer(data=msg_data,
                                       dynamics_model_input_builder=input_builder,
                                       model_dy=self._model_dy,
                                       debug=False)

        self._state_dict = {'episode': episode,
                            'input_builder': input_builder,
                            'plan_msg': msg,
                            'plan': plan_container,
                            'action_counter': 0,
                            'timestamp_system': [],  # list of when we took the actions
                            'mpc_out': None,
                            }

        self._state = ControllerState.PLAN_READY
        resp = {'type': 'PLAN',
                'data': 'READY',
                }

        print("----FINISHED PROCESSING PLAN-----")
        return resp

    def _on_reset_msg(self, msg):
        """
        Clear the current plan object
        Reset the state
        :param msg:
        :type msg:
        :return:
        :rtype:
        """

        self._state = ControllerState.STOPPED
        self._state_dict = dict()
        self._planner.config = copy.deepcopy(self._planner_config_copy)

        resp = {'type': "RESET",
                'data': "READY"}
        return resp  # the response

    def _on_compute_control_action(self,
                                   msg,
                                   visualize=True,
                                   ):
        """
        Computes the current control action

        :param msg:
        :type msg:
        :return:
        :rtype:
        """
        print("\n\n----------")

        start_time = time.time()

        assert self._state in [ControllerState.PLAN_READY, ControllerState.RUNNING]

        # allow setting the visualize flag from the message
        try:
            visualize = bool(int(msg['debug']))
        except KeyError:
            pass

        # add data to the OnlineEpisodeReader
        episode = self._state_dict['episode']
        input_builder = self._state_dict['input_builder']
        plan = self._state_dict['plan']

        observation = msg['data']['observations']
        action_dict = msg['data']['actions']

        if self._state == ControllerState.PLAN_READY:
            # this is the first time through the loop
            # then add a few to populate n_his
            for i in range(self._n_history):
                # episode.add_observation_action(copy.deepcopy(observation), copy.deepcopy(action_dict))
                episode.add_data(copy.deepcopy(msg['data']))

        episode.add_data(msg['data'])

        # run the planner

        # seed with previous actions
        mpc_out = self._state_dict['mpc_out']
        n_look_ahead = None  # this is the MPC horizon
        mpc_horizon_type = self._planner.config['mpc']['hardware']['mpc_horizon']["type"]
        mpc_hardware_config = self._planner.config['mpc']['hardware']

        # previous actions to seed with
        # compute the MPC horizon
        action_seq_rollout_init = None
        if mpc_out is not None:
            # this means it's not our first time through the loop
            # the plan is already running
            action_seq_rollout_init = mpc_out['action_seq'][1:]

            if mpc_horizon_type == "PLAN_LENGTH":
                n_look_ahead = action_seq_rollout_init.shape[0]
                # this means we are at the end of the plan
                # so send the stop message
                if n_look_ahead == 0:
                    print("Plan finished, sending STOP message")
                    return self.make_stop_message()
            if mpc_horizon_type == "MIN_HORIZON":
                n_look_ahead = action_seq_rollout_init.shape[0]
                H_min = mpc_hardware_config['mpc_horizon']['H_min']
                n_look_ahead = max(H_min, n_look_ahead)
                # maybe set overwrite the config to be traj cost . . .
            elif mpc_horizon_type == "FIXED":
                n_look_ahead = mpc_hardware_config['mpc_horizon']["H_fixed"]
            else:
                raise ValueError("unknow mpc_horizon_type")

            # add zeros to the end of the action trajectory as a good
            # starting point
            # extend the previous action, either with zeros or by repeating last action
            if action_seq_rollout_init.shape[0] < n_look_ahead:
                if mpc_hardware_config['action_extension_type'] == "CONSTANT":
                    num_steps = n_look_ahead - action_seq_rollout_init.shape[0]
                    # [action_dim]
                    action_extend = action_seq_rollout_init[-1]
                    action_extend = action_extend.unsqueeze(0).expand([num_steps, -1])

                    # print("action_seq_rollout_init.shape", action_seq_rollout_init.shape)
                    # print("action_extend.shape", action_extend.shape)
                    action_seq_rollout_init = torch.cat((action_seq_rollout_init, action_extend), dim=0)
                elif mpc_hardware_config['action_extension_type'] == "ZERO":
                    num_steps = n_look_ahead - action_seq_rollout_init.shape[0]
                    action_seq_zero = torch.zeros([num_steps, 2]).to(action_seq_rollout_init.device)
                    action_seq_rollout_init = torch.cat((action_seq_rollout_init, action_seq_zero), dim=0)
        else:
            if mpc_horizon_type == "FIXED":
                n_look_ahead = mpc_hardware_config['mpc_horizon']["H_fixed"]
            else:
                n_look_ahead = mpc_hardware_config['mpc_horizon']["H_init"]

            action_seq_rollout_init = None

        print("n_look_ahead", n_look_ahead)
        print("plan_counter", plan.counter)
        start_time_tmp = time.time()
        idx = episode.get_latest_idx()
        mpc_input_data = input_builder.get_dynamics_model_input(idx,
                                                                n_history=self._n_history)
        print("computing dynamics model input took", time.time() - start_time_tmp)
        start_time_tmp = time.time()
        state_cur = mpc_input_data['states']
        action_his = mpc_input_data['actions']

        current_reward_data = None

        # run the planner
        with torch.no_grad():
            # z_goal_dict = plan.data[-1]['dynamics_model_input_data']['z']
            # obs_goal = plan.data[-1]['dynamics_model_input_data']['z']['z_object_flat']

            # convert state_cur to z_cur
            z_cur_dict = self._model_dy.compute_z_state(state_cur)
            z_cur = z_cur_dict['z']
            z_cur_no_his = z_cur[-1]
            print("z_cur.shape")

            # for computing current cost
            z_batch = None


            # for now we support just final state rather than trajectory costs
            z_goal = None
            if self._planner.config['mpc']['reward']["goal_type"] == "TRAJECTORY":
                z_goal = plan.get_trajectory_goal(counter=self._state_dict['action_counter'],

                                                  n_look_ahead=n_look_ahead)

                # [1, 1, z_dim]
                z_batch = z_cur_no_his.unsqueeze(0).unsqueeze(0)

                # [1, n_look_aheda, z_dim]
                z_batch = z_batch.expand([-1, n_look_ahead, -1])
            elif self._planner.config['mpc']['reward']["goal_type"] == "FINAL_STATE":
                z_goal = plan.get_final_state_goal()

                # [1, 1, z_dim]
                z_batch = z_cur_no_his.unsqueeze(0).unsqueeze(0)
            else:
                raise ValueError("unknown goal type")

            print("z_batch.shape", z_batch.shape)

            obs_goal = torch.index_select(z_goal, dim=-1, index=self._object_indices)
            print("obs_goal.shape", obs_goal.shape)

            # compute the cost of stopping now so we can compare it to the cost
            # of running the trajectory optimization

            
            # note this is on the CPU
            # [1, 1, z_dim]
            z_batch = z_cur_no_his.unsqueeze(0).unsqueeze(0)
            # [goal_dim]
            obs_goal_final = torch.index_select(plan.get_final_state_goal(), dim=-1, index=self._object_indices)
            current_reward_data = \
                planner_utils.evaluate_model_rollout(state_pred=z_batch,
                                                     obs_goal=obs_goal_final,
                                                     eval_indices=self._object_indices,

                                                     **self._planner.config['mpc']['reward'])

            print("current reward:", current_reward_data['best_reward'])
            if current_reward_data['best_reward'] > mpc_hardware_config['goal_reward_threshold']:
                print("Below goal reward threshold, stopping")
                return self.make_stop_message()

            # run the planner
            mpc_out = self._planner.trajectory_optimization(state_cur=z_cur,
                                                            action_his=action_his,
                                                            obs_goal=obs_goal,
                                                            model_dy=self._model_dy,
                                                            action_seq_rollout_init=action_seq_rollout_init,
                                                            n_look_ahead=n_look_ahead,
                                                            eval_indices=self._object_indices,
                                                            rollout_best_action_sequence=self._debug,
                                                            verbose=self._debug,
                                                            add_grid_action_samples=False,
                                                            )

        # update the action that was actually applied
        # equivalent to
        action_seq_mpc = mpc_out['action_seq']
        action_dict['ee_setpoint']['setpoint_linear_velocity']['x'] = float(action_seq_mpc[0][0])
        action_dict['ee_setpoint']['setpoint_linear_velocity']['x'] = float(action_seq_mpc[0][1])

        state_pred = mpc_out['state_pred']

        if mpc_out['reward'].cpu() < current_reward_data['best_reward'].cpu() + mpc_hardware_config['reward_improvement_tol']:

            if mpc_hardware_config['terminate_if_no_improvement']:
                print("Traj opt didn't yield successive improvement, STOPPING")
                return self.make_stop_message()
            else:
                print("Traj opt didn't yield successive improvement, HOLDING STILL")
                return self.make_zero_action_message()


        if self._debug:
            # pass
            print("action_seq_mpc\n", action_seq_mpc)

        if visualize:
            vis = meshcat_utils.make_default_visualizer_object()
            vis["mpc"].delete()

            visualize_model_prediction_single_timestep(vis,
                                                       self._config,
                                                       z_cur[0],
                                                       display_idx=0,
                                                       name_prefix="start",
                                                       color=[0, 0, 255])

            # visualize pointcloud of current position
            start_data = episode.get_data(0)
            depth = start_data['observations']['images'][self._camera_info['camera_name']][
                        'depth_int16'] / DEPTH_IM_SCALE
            rgb = start_data['observations']['images'][self._camera_info['camera_name']]['rgb']
            # pointcloud
            name = "mpc/pointclouds/start"
            pdc_meshcat_utils.visualize_pointcloud(vis,
                                                   name,
                                                   depth=depth,
                                                   K=self._camera_info['K'],
                                                   rgb=rgb,
                                                   T_world_camera=self._camera_info['T_world_camera'])

            #
            # visualize_model_prediction_single_timestep(vis,
            #                                            self._config,
            #                                            z_goal_dict['z'],
            #                                            display_idx=0,
            #                                            name_prefix="goal",
            #                                            color=[0, 255, 0])
            #
            # goal_data = plan.data[-1]
            # depth = goal_data['observations']['images'][self._camera_info['camera_name']][
            #             'depth_int16'] / DEPTH_IM_SCALE
            # rgb = goal_data['observations']['images'][self._camera_info['camera_name']]['rgb']
            # # pointcloud
            # name = "pointclouds/goal"
            # pdc_meshcat_utils.visualize_pointcloud(vis,
            #                                        name,
            #                                        depth=depth,
            #                                        K=self._camera_info['K'],
            #                                        rgb=rgb,
            #                                        T_world_camera=self._camera_info['T_world_camera'])

            for i in range(state_pred.shape[0]):
                visualize_model_prediction_single_timestep(vis,
                                                           self._config,
                                                           state_pred[i],
                                                           display_idx=(i + 1),
                                                           name_prefix="mpc",
                                                           color=[255, 0, 0],
                                                           )
            # show pointclouds . . .

        # store the results of this computation
        self._state_dict['action_counter'] += 1
        self._state_dict['timestamp_system'].append(time.time())
        self._state_dict['mpc_out'] = mpc_out
        self._state = ControllerState.RUNNING  # or maybe also STOPPED/FINISHED

        # data we want to save out from the MPC
        mpc_save_data = {'n_look_ahead': n_look_ahead,
                         'action_seq': mpc_out['action_seq'].cpu(),
                         'reward': mpc_out['reward'].cpu(),
                         'state_pred': mpc_out['state_pred'].cpu(),
                         'current_reward': current_reward_data,
                         }

        # add some data for saving out later
        idx = episode.get_latest_idx()
        episode_data = episode.get_data(idx)
        episode_data['mpc'] = mpc_save_data

        plan.increment_counter()
        action_seq_mpc_np = torch_utils.cast_to_numpy(action_seq_mpc)

        # todo: update the action in the episode reader. Currently
        # we have an off by one error
        if False:
            action = action_seq_mpc[0]
            idx = episode.get_latest_idx()
            episode_data = episode.get_data(idx)
            episode_data['actions']['mpc'] # would be pretty hacky . . . .
            episode_data['actions']['setpoint_linear_velocity']['x'] = action[0]
            episode_data['actions']['setpoint_linear_velocity']['y'] = action[1]


        resp_data = {'action': action_seq_mpc_np[0],
                     'action_seq': action_seq_mpc_np,
                     }

        resp = {'type': 'CONTROL_ACTION',
                'data': resp_data}

        # we are seeing averages around 0.12 seconds (total) for length 10 plan
        print("Compute Control Action Took %.3f" % (time.time() - start_time))

        return resp

    def make_stop_message(self):
        return {'type': "STOP",
                'data': "STOP",  # just for compatibility
                }

    def make_zero_action_message(self):
        action = np.array([0., 0.])
        data = {'action': action}
        return {'type': 'CONTROL_ACTION',
                'data': data,
                }

    def make_default_success_response_message(self):
        return {'type': 'SUCCESS',
                'data': 'SUCCESS',
                }

    def save_data(self,
                  save_dir=None,
                  ):

        """
        Saves data from the

        - PlanContainer
        - OnlineEpisode
        """

        if save_dir is None:
            save_dir = os.path.join(utils.get_data_root(), 'hardware_experiments/closed_loop_rollouts/sandbox', utils.get_current_YYYY_MM_DD_hh_mm_ss_ms())

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        print("saving MPC rollout data at: %s" %(save_dir))
        save_data = {'episode': self._state_dict['episode'].get_save_data(),
                     'plan': self._state_dict['plan'].get_save_data(),
                     }

        save_file = os.path.join(save_dir, "data.p")
        utils.save_pickle(save_data, save_file)
        print("done saving data")


if __name__ == "__main__":
    # d = {'name': 'Lucas', 'age': 29}
    # d2 = msgpack.unpackb(msgpack.packb(d), raw=True)
    # print(d2.keys())
    # quit()
    controller = ZMQController()
    controller.run()
