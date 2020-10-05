"""
Visualizes an MPC rollout. Allows you to step through it incrementally

- visualizes the result in meshcat

"""

import os
import numpy as np
import copy
import transforms3d
import time
import matplotlib.pyplot as plt

# need to import pydrake before pdc
import pydrake
import meshcat

# pdc for setting available GPU's
from dense_correspondence_manipulation.utils.utils import set_cuda_visible_devices

GPU_LIST = [0]
set_cuda_visible_devices(GPU_LIST)

import torch

from meshcat.geometry import triad

# dense correspondence
from dense_correspondence_manipulation.utils import constants
from dense_correspondence_manipulation.utils import utils as pdc_utils
from dense_correspondence.network import predict
import dense_correspondence_manipulation.utils.visualization as vis_utils

# key_dynam
from key_dynam.utils.utils import get_project_root, load_yaml, get_current_YYYY_MM_DD_hh_mm_ss_ms, get_data_root, \
    load_pickle
from key_dynam.envs.drake_pusher_slider import DrakePusherSliderEnv
from key_dynam.envs import utils as env_utils
from key_dynam.dynamics.models_dy import rollout_model, rollout_action_sequences, get_object_and_robot_state_indices
from key_dynam.dataset.function_factory import ObservationFunctionFactory, ActionFunctionFactory, \
    slider_pose_from_observation
from key_dynam.dataset.vision_function_factory import VisualObservationFunctionFactory
from key_dynam.dataset.online_episode_reader import OnlineEpisodeReader
from key_dynam.dataset.mpc_dataset import DynamicsModelInputBuilder
from key_dynam.utils import meshcat_utils
from key_dynam.planner.planners import RandomShootingPlanner, PlannerMPPI
from key_dynam.utils import transform_utils, torch_utils
from key_dynam.experiments.exp_09 import utils as exp_utils
from key_dynam.utils import drake_image_utils
from key_dynam.dynamics.utils import keypoints_3D_from_dynamics_model_output, set_seed
from key_dynam.eval.utils import compute_pose_error
from key_dynam.models import model_builder
from key_dynam.models.model_builder import build_dynamics_model
from key_dynam.autoencoder.autoencoder_models import ConvolutionalAutoencoder

# constants
set_seed(0)

USE_FIXED_MPC_HORIZON = False
USE_SHORT_HORIZON_MPC = False
TERMINAL_COST_ONLY = True
TRAJECTORY_GOAL = False
REPLAN = True

SEED_WITH_NOMINAL_ACTIONS = False

N = 2*10  # horizon length for ground truth push
N = 10
MPC_HORIZON = 10

PUSHER_VELOCITY = 0.20
PUSHER_ANGLE = np.deg2rad(20)
YAW = np.deg2rad(0) # global yaw
# GLOBAL_TRANSLATION = np.array([-0.1, 0.0, 0])
# GLOBAL_TRANSLATION = np.array([-0.05, 0.1, 0])
GLOBAL_TRANSLATION = np.array([-0.2, 0.0, 0])
# GLOBAL_TRANSLATION = np.array([0,0,0])
SEED_MPC_W_GT_ACTION_SEQUENCE = False
RANDOM_SEED = 2


OBJECT_YAW = np.deg2rad(0)



PLANNER_TYPE = "mppi"
# PLANNER_TYPE = "random_shooting"

BOX_ON_SIDE = True

def get_initial_state():
    # set initial condition
    slider_pos = np.array([0.0, 0.0, 0.3])
    slider_quat = np.array([1, 0, 0, 0])
    slider_quat = transforms3d.euler.euler2quat(0,0,OBJECT_YAW)

    # O denotes the canonical frame
    T_O_slider = transform_utils.transform_from_pose(slider_pos, slider_quat)
    # pusher_pos_homog_O = np.array([-0.15, 0, 0, 1])
    pusher_pos_homog_O = np.array([-0.075, 0, 0, 1])

    pusher_velocity_3d_O = PUSHER_VELOCITY * np.array([np.cos(PUSHER_ANGLE), np.sin(PUSHER_ANGLE), 0])

    # apply transform T_W_O to pusher and slider
    T_W_O = np.eye(4)
    T_W_O[:3, 3] = GLOBAL_TRANSLATION
    T_W_O[:3, :3] = transforms3d.euler.euler2mat(0, 0, YAW)

    T_W_slider = T_W_O @ T_O_slider
    pusher_pos_3d_W = T_W_O @ pusher_pos_homog_O
    pusher_velocity_3d_W = (T_W_O[:3, :3] @ pusher_velocity_3d_O)

    # slider
    slider_pose_dict = transform_utils.matrix_to_dict(T_W_slider)
    q_slider = np.concatenate([slider_pose_dict['quaternion'], slider_pose_dict['position']])
    q_pusher = pusher_pos_3d_W[:2]
    action = pusher_velocity_3d_W[:2]


    action_sequence = None
    if False:
        angle_1 = np.deg2rad(0)
        action_1 = np.array([np.cos(angle_1), np.sin(angle_1)])

        angle_2 = np.deg2rad(-20)
        action_2 = np.array([np.cos(angle_2), np.sin(angle_2)])

        # expand it to be an action sequence
        # [N, 2]
        action_sequence_1 = PUSHER_VELOCITY * torch.Tensor(action_1).unsqueeze(0).expand([N//2, -1])
        action_sequence_2 = PUSHER_VELOCITY * torch.Tensor(action_2).unsqueeze(0).expand([N//2, -1])

        action_sequence = torch.cat((action_sequence_1, action_sequence_2), dim=0)
        print("action_sequence.shape", action_sequence.shape)
    else:
        action_sequence = torch.Tensor(action).unsqueeze(0).expand([N, -1])

    return {'T_W_O': T_W_O,
            'pusher_pos_3d': pusher_pos_3d_W,
            'T_W_slider': T_W_slider,
            'q_slider': q_slider,
            'q_pusher': q_pusher,
            'action': action,
            'action_sequence': action_sequence,
            }

def reset_environment(env, q_pusher, q_slider):
    context = env.get_mutable_context()
    env.reset()  # sets velocities to zero
    env.set_pusher_position(context, q_pusher)
    env.set_slider_position(context, q_slider)
    env.step(np.array([0, 0]), dt=2.0)  # allow box to drop down


def get_color_intensity(n, N):
    return 1.0 - 0.5 * n * 1.0 / N


def load_model_state_dict(model_folder=None):

    models_root = os.path.join(get_data_root(), "dev/experiments/drake_pusher_slider_v2")


    # model_name = "dataset_2020-04-23-20-45-12-697915_T_aug_random_velocity_1000_angled_cam/trained_models/dynamics/DD_3D/2020-05-12-12-03-05-252242_DD_3D_spatial_z_n_his_2"

    model_name = "dataset_2020-04-23-20-45-12-697915_T_aug_random_velocity_1000_angled_cam/trained_models/dynamics/DD_3D/2020-05-11-19-44-35-085478_DD_3D_n_his_2"



    model_folder = os.path.join(models_root, model_name)

    model_dy = model_builder.load_dynamics_model_from_folder(model_folder)['model_dy']

    # load dense descriptor model
    metadata = load_pickle(os.path.join(model_folder, 'metadata.p'))
    model_dd_file = metadata['model_file']
    model_dd = torch.load(model_dd_file)
    model_dd = model_dd.eval()
    model_dd = model_dd.cuda()

    spatial_descriptor_data = load_pickle(os.path.join(model_folder, 'spatial_descriptors.p'))

    return {'model_dy': model_dy,
            'model_dd': model_dd,
            'spatial_descriptor_data': spatial_descriptor_data,
            'metadata': metadata}




def load_autoencoder_model():
    train_dir = "/home/manuelli/data/key_dynam/dev/experiments/drake_pusher_slider_v2/dataset_2020-04-20-14-58-21-418302_T_aug_random_velocity_1000/trained_models/dynamics/autoencoder/2020-06-08-19-08-45-269917_z_dim_16"

    chkpt_file = "net_dy_epoch_294_iter_100_state_dict.pth"
    ckpt_file = os.path.join(train_dir, chkpt_file)

    config = load_yaml(os.path.join(train_dir, 'config.yaml'))
    state_dict = torch.load(ckpt_file)

    # build dynamics model
    model_dy = build_dynamics_model(config)
    model_dy.load_state_dict(state_dict['dynamics'])
    model_dy = model_dy.eval()
    model_dy = model_dy.cuda()

    # build autoencoder file
    model_ae = ConvolutionalAutoencoder.from_global_config(config)
    model_ae.load_state_dict(state_dict['autoencoder'])
    model_ae = model_ae.eval()
    model_ae = model_ae.cuda()



    # visual observation function
    visual_observation_func = VisualObservationFunctionFactory.autoencoder_latent_state(config,
                                                              model_ae=model_ae)

    return {'model_ae': model_ae,
            'model_dy': model_dy,
            'visual_observation_function': visual_observation_func}



def main():
    # load dynamics model
    model_dict = load_autoencoder_model()
    model = model_dict['model_dy']
    model_ae = model_dict['model_ae']
    visual_observation_function = model_dict['visual_observation_function']


    config = model.config

    env_config = load_yaml(os.path.join(get_project_root(), 'experiments/exp_18_box_on_side/config.yaml'))
    env_config['env']['observation']['depth_int16'] = True
    n_history = config['train']['n_history']


    # create the environment
    # create the environment
    env = DrakePusherSliderEnv(env_config)
    env.reset()

    # create another environment for doing rollouts
    env2 = DrakePusherSliderEnv(env_config, visualize=False)
    env2.reset()

    action_function = ActionFunctionFactory.function_from_config(config)
    observation_function = ObservationFunctionFactory.drake_pusher_position_3D(config)

    episode = OnlineEpisodeReader()
    mpc_input_builder = DynamicsModelInputBuilder(observation_function=observation_function,
                                                  visual_observation_function=visual_observation_function,
                                                  action_function=action_function,
                                                  episode=episode)

    vis = meshcat_utils.make_default_visualizer_object()
    vis.delete()

    initial_cond = get_initial_state()
    reset_environment(env, initial_cond['q_pusher'], initial_cond['q_slider'])
    obs_init = env.get_observation()

    # visualize starting position of the object
    print("obs_init.keys()", obs_init.keys())
    print("obs_init['slider']['position']", obs_init['slider']['position'])
    T = DrakePusherSliderEnv.object_position_from_observation(obs_init)
    vis['start_pose'].set_object(triad(scale=0.1))
    vis['state_pose'].set_transform(T)





    #### ROLLOUT USING LEARNED MODEL + GROUND TRUTH ACTIONS ############
    reset_environment(env, initial_cond['q_pusher'], initial_cond['q_slider'])
    # add just some large number of these
    episode.clear()
    for i in range(n_history):
        action_zero = np.zeros(2)
        obs_tmp = env.get_observation()
        episode.add_observation_action(obs_tmp, action_zero)


    #### ROLLOUT THE ACTION SEQUENCE USING THE SIMULATOR ##########
    # rollout single action sequence using the simulator
    gt_rollout_data = env_utils.rollout_action_sequence(env, initial_cond['action_sequence'].cpu().numpy())
    env_obs_rollout_gt = gt_rollout_data['observations']
    gt_rollout_episode = gt_rollout_data['episode_reader']

    for i, env_obs in enumerate(gt_rollout_data['observations']):
        T = DrakePusherSliderEnv.object_position_from_observation(env_obs)
        vis_name = "GT_trajectory/%d" % (i)
        vis[vis_name].set_object(triad(scale=0.1))
        vis[vis_name].set_transform(T)


    action_state_gt = mpc_input_builder.get_action_state_tensors(start_idx=0,
                                                                 num_timesteps=N,
                                                                 episode=gt_rollout_episode)



    state_rollout_gt = action_state_gt['states']
    action_rollout_gt = action_state_gt['actions']
    z_object_rollout_gt = model.compute_z_state(state_rollout_gt)['z_object_flat']
    print('state_rollout_gt.shape', state_rollout_gt.shape)
    print("z_object_rollout_gt.shape", z_object_rollout_gt.shape)


    def goal_func(obs_tmp):
        state_tmp = mpc_input_builder.get_state_input_single_timestep({'observation': obs_tmp})['state']
        return model.compute_z_state(state_tmp.unsqueeze(0))['z_object'].flatten()

    # using the vision model to get "goal" keypoints
    z_object_goal = goal_func(env_obs_rollout_gt[-1])
    z_object_goal_np = torch_utils.cast_to_numpy(z_object_goal)

    # input("press Enter to continue")

    #### ROLLOUT USING LEARNED MODEL + GROUND TRUTH ACTIONS ############
    reset_environment(env, initial_cond['q_pusher'], initial_cond['q_slider'])
    # add just some large number of these
    episode.clear()
    for i in range(n_history):
        action_zero = np.zeros(2)
        obs_tmp = env.get_observation()
        episode.add_observation_action(obs_tmp, action_zero)

    # [n_history, state_dim]
    idx = episode.get_latest_idx()

    dyna_net_input = mpc_input_builder.get_dynamics_model_input(idx, n_history=n_history)
    state_init = dyna_net_input['states'].cuda() # [n_history, state_dim]
    action_init = dyna_net_input['actions'] # [n_history, action_dim]


    print("state_init.shape", state_init.shape)
    print("action_init.shape", action_init.shape)
    print("n_history", n_history)


    action_seq_gt_torch = initial_cond['action_sequence']
    action_input = torch.cat((action_init[:(n_history-1)], action_seq_gt_torch), dim=0).cuda()
    print("action_input.shape", action_input.shape)


    # rollout using the ground truth actions and learned model
    # need to add the batch dim to do that
    z_init = model.compute_z_state(state_init)['z']
    rollout_pred = rollout_model(state_init=z_init.unsqueeze(0),
                                 action_seq=action_input.unsqueeze(0),
                                 dynamics_net=model,
                                 compute_debug_data=True)

    state_pred_rollout = rollout_pred['state_pred'].squeeze(0)

    print("state_pred_rollout.shape", state_pred_rollout.shape)
    # input("press Enter to continue")

    # check L2 distance between predicted and actual
    # basically comparing state_pred_rollout and state_rollout_gt
    print("state_rollout_gt[-1]\n", state_rollout_gt[-1])
    print("state_pred_rollout[-1]\n", state_pred_rollout[-1])



    index_dict = get_object_and_robot_state_indices(config)
    object_indices = index_dict['object_indices']

    # reset the environment and use the MPC controller to stabilize this
    # now setup the MPC to try to stabilize this . . . .
    reset_environment(env, initial_cond['q_pusher'], initial_cond['q_slider'])
    episode.clear()

    # add just some large number of these
    for i in range(n_history):
        action_zero = np.zeros(2)
        obs_tmp = env.get_observation()
        episode.add_observation_action(obs_tmp, action_zero)

    # input("press Enter to continue")

    # make a planner config
    planner_config = copy.copy(config)
    config_tmp = load_yaml(os.path.join(get_project_root(), 'experiments/drake_pusher_slider/eval_config.yaml'))
    planner_config['mpc'] = config_tmp['mpc']

    planner_config['mpc']['mppi']['terminal_cost_only'] = TERMINAL_COST_ONLY
    planner_config['mpc']['random_shooting']['terminal_cost_only'] = TERMINAL_COST_ONLY

    planner = None
    if PLANNER_TYPE == "random_shooting":
        planner = RandomShootingPlanner(planner_config)
    elif PLANNER_TYPE == "mppi":
        planner = PlannerMPPI(planner_config)
    else:
        raise ValueError("unknown planner type: %s" % (PLANNER_TYPE))

    mpc_out = None
    action_seq_mpc = None
    state_pred_mpc = None
    counter = -1
    while True:
        counter += 1
        print("\n\n-----Running MPC Optimization: Counter (%d)-------" % (counter))

        obs_cur = env.get_observation()
        episode.add_observation_only(obs_cur)

        if counter == 0 or REPLAN:
            print("replanning")
            ####### Run the MPC ##########

            # [1, state_dim]

            n_look_ahead = N - counter
            if USE_FIXED_MPC_HORIZON:
                n_look_ahead = MPC_HORIZON
            elif USE_SHORT_HORIZON_MPC:
                n_look_ahead = min(MPC_HORIZON, N - counter)
            if n_look_ahead == 0:
                break

            start_idx = counter
            end_idx = counter + n_look_ahead

            print("start_idx:", start_idx)
            print("end_idx:", end_idx)
            print("n_look_ahead", n_look_ahead)

            # start_time = time.time()
            # idx of current observation
            idx = episode.get_latest_idx()
            mpc_start_time = time.time()
            mpc_input_data = mpc_input_builder.get_dynamics_model_input(idx, n_history=n_history)
            state_cur = mpc_input_data['states']
            action_his = mpc_input_data['actions']

            if SEED_WITH_NOMINAL_ACTIONS:
                action_seq_rollout_init = action_seq_gt_torch[start_idx:end_idx]
            else:
                if mpc_out is not None:
                    action_seq_rollout_init = mpc_out['action_seq'][1:]
                    print("action_seq_rollout_init.shape", action_seq_rollout_init.shape)

                    if action_seq_rollout_init.shape[0] < n_look_ahead:
                        num_steps = n_look_ahead - action_seq_rollout_init.shape[0]
                        action_seq_zero = torch.zeros([num_steps, 2])

                        action_seq_rollout_init = torch.cat((action_seq_rollout_init, action_seq_zero), dim=0)
                        print("action_seq_rollout_init.shape", action_seq_rollout_init.shape)
                else:
                    action_seq_rollout_init = None


            # run MPPI
            z_cur = None
            with torch.no_grad():
                z_cur = model.compute_z_state(state_cur.unsqueeze(0).cuda())['z'].squeeze(0)


            if action_seq_rollout_init is not None:
                n_look_ahead = action_seq_rollout_init.shape[0]

            obs_goal = None
            print("z_object_rollout_gt.shape", z_object_rollout_gt.shape)
            if TRAJECTORY_GOAL:
                obs_goal = z_object_rollout_gt[start_idx:end_idx]

                print("n_look_ahead", n_look_ahead)
                print("obs_goal.shape", obs_goal.shape)

                # add the final goal state on as needed
                if obs_goal.shape[0] < n_look_ahead:
                    obs_goal_final = z_object_rollout_gt[-1].unsqueeze(0)
                    num_repeat = n_look_ahead - obs_goal.shape[0]
                    obs_goal_final_expand = obs_goal_final.expand([num_repeat, -1])
                    obs_goal = torch.cat((obs_goal, obs_goal_final_expand), dim=0)
            else:
                obs_goal = z_object_rollout_gt[-1]

            obs_goal = torch_utils.cast_to_numpy(obs_goal)
            print("obs_goal.shape", obs_goal.shape)


            seed = 1

            set_seed(seed)
            mpc_out = planner.trajectory_optimization(state_cur=z_cur,
                                                      action_his=action_his,
                                                      obs_goal=obs_goal,
                                                      model_dy=model,
                                                      action_seq_rollout_init=action_seq_rollout_init,
                                                      n_look_ahead=n_look_ahead,
                                                      eval_indices=object_indices,
                                                      rollout_best_action_sequence=True,
                                                      verbose=True,
                                                      add_grid_action_samples=True,
                                                      )

            print("MPC step took %.4f seconds" %(time.time() - mpc_start_time))
            action_seq_mpc = torch_utils.cast_to_numpy(mpc_out['action_seq'])
            state_pred_mpc = torch_utils.cast_to_numpy(mpc_out['state_pred'])


        # Rollout with ground truth simulator dynamics
        env2.set_simulator_state_from_observation_dict(env2.get_mutable_context(), obs_cur)
        obs_mpc_gt = env_utils.rollout_action_sequence(env2, action_seq_mpc)['observations']

        vis['mpc_3D'].delete()
        vis['mpc_GT_3D'].delete()

        L = len(obs_mpc_gt)
        print("L", L)
        if L == 0:
            break
        for i in range(L):

            # ground truth rollout of the MPC action_seq
            name = "mpc_GT_3D/%d" % (i)
            T_W_obj = DrakePusherSliderEnv.object_position_from_observation(obs_mpc_gt[i])
            vis[name].set_object(triad(scale=0.1))
            vis[name].set_transform(T_W_obj)

        action_cur = action_seq_mpc[0]

        print("action_cur", action_cur)
        print("action_GT", initial_cond['action'])
        input("press Enter to continue")

        # add observation actions to the episode
        obs_cur = env.get_observation()
        episode.replace_observation_action(obs_cur, action_cur)

        # step the simulator
        env.step(action_cur)

        # update the trajectories, in case we aren't replanning
        action_seq_mpc = action_seq_mpc[1:]
        state_pred_mpc = state_pred_mpc[1:]

        pose_error = compute_pose_error(env_obs_rollout_gt[-1],
                                        obs_cur)

        print("position_error: %.3f" % (pose_error['position_error']))
        print("angle error degrees: %.3f" % (pose_error['angle_error_degrees']))


    obs_final = env.get_observation()

    pose_error = compute_pose_error(env_obs_rollout_gt[-1],
                                    obs_final)

    print("position_error: %.3f"  %(pose_error['position_error']))
    print("angle error degrees: %.3f" %(pose_error['angle_error_degrees']))


if __name__ == "__main__":
    main()
