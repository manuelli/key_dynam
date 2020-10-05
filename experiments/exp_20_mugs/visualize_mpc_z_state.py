"""
Visualizes an MPC rollout. Allows you to step through it incrementally

- visualizes the result in meshcat

"""

import os
import numpy as np
import copy
import transforms3d
import time
import math
import matplotlib.pyplot as plt

# need to import pydrake before pdc
import pydrake
import meshcat

# pdc for setting available GPU's
from dense_correspondence_manipulation.utils.utils import set_cuda_visible_devices

GPU_LIST = [0]
set_cuda_visible_devices(GPU_LIST)

import torch

# dense correspondence
from dense_correspondence_manipulation.utils import constants
from dense_correspondence_manipulation.utils import utils as pdc_utils
from dense_correspondence.network import predict
import dense_correspondence_manipulation.utils.visualization as vis_utils

# key_dynam
from key_dynam.utils.utils import get_project_root, load_yaml, get_current_YYYY_MM_DD_hh_mm_ss_ms, get_data_root, \
    load_pickle
from key_dynam.envs.drake_pusher_slider import DrakePusherSliderEnv
from key_dynam.envs.drake_mugs import DrakeMugsEnv
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
from key_dynam.dynamics.utils import keypoints_3D_from_dynamics_model_output
from key_dynam.eval.utils import compute_pose_error
from key_dynam.models import model_builder
from key_dynam.dataset.episode_dataset import MultiEpisodeDataset
from key_dynam.dynamics.utils import set_seed


from key_dynam.experiments.exp_20_mugs.collect_episodes import sample_random_mug, sample_random_color

# constants
USE_FIXED_MPC_HORIZON = False
MPC_HORIZON = 15

PUSH_LENGTH = 0.2
PUSHER_VELOCITY = 0.20
PUSHER_ANGLE = np.deg2rad(0)
YAW = np.deg2rad(0)
# GLOBAL_TRANSLATION = np.array([-0.1, 0.0, 0])
# GLOBAL_TRANSLATION = np.array([-0.05, 0.1, 0])
# GLOBAL_TRANSLATION = np.array([-0.1, 0.0, 0])  # doesn't work with any offset
GLOBAL_TRANSLATION = np.array([0,0,0])
SEED_MPC_W_GT_ACTION_SEQUENCE = False
RANDOM_SEED = 3

OBJECT_YAW = np.deg2rad(-90)

REPLAN = True

PLANNER_TYPE = "mppi"
# PLANNER_TYPE = "random_shooting"

BOX_ON_SIDE = True


set_seed(RANDOM_SEED)

def sample_object_position(T_aug=None,
                           upright=False):

    pos = np.array([0, 0, 0.1])
    quat = None
    if upright:
        quat = np.array([1,0,0,0])
    else:
        quat = transforms3d.euler.euler2quat(np.deg2rad(90), 0, 0)

    T_O_slider = transform_utils.transform_from_pose(pos, quat)

    # apply a random yaw to the object
    yaw = OBJECT_YAW
    quat_yaw = transforms3d.euler.euler2quat(0, 0, yaw)
    T_yaw = transform_utils.transform_from_pose([0,0,0], quat_yaw)

    T_O_slider = np.matmul(T_yaw, T_O_slider)

    T_W_slider = None
    if T_aug is not None:
        T_W_slider = T_aug @ T_O_slider
    else:
        T_W_slider = T_O_slider

    pose_dict = transform_utils.matrix_to_dict(T_W_slider)

    # note the quat/pos ordering
    q = np.concatenate((pose_dict['quaternion'], pose_dict['position']))

    return q


def sample_pusher_position_and_velocity(T_aug=None,
                                        N=1000,
                                        n_his=0,
                                        randomize_velocity=False):

    assert randomize_velocity is not None

    if T_aug is None:
        T_aug = np.eye(4)

    # low = np.array([-0.1, -0.05])
    # high = np.array([-0.09, 0.05])
    # q_pusher = np.array(Box(low, high).sample())

    q_pusher = np.array([-0.09, 0])
    q_pusher = np.array([-0.09, -0.02])

    q_pusher_3d_homog = np.array([0, 0, 0, 1.0])
    q_pusher_3d_homog[:2] = q_pusher


    # v_pusher_3d = sample_pusher_velocity_3D()
    v_pusher_3d = PUSHER_VELOCITY * np.array([np.cos(PUSHER_ANGLE), np.sin(PUSHER_ANGLE), 0])

    # 1000 timesteps of pusher velocity
    v_pusher_3d_seq = np.zeros([N, 3])
    for i in range(N):

        v_pusher_3d_tmp = None
        if randomize_velocity:
            v_pusher_3d_tmp = sample_pusher_velocity_3D()
        else:
            v_pusher_3d_tmp = v_pusher_3d

        # always moving to the right
        # v_pusher_3d = magnitude * np.array([1, 0, 0])

        v_pusher_3d_seq[i] = T_aug[:3, :3] @ v_pusher_3d_tmp

    action_seq = v_pusher_3d_seq[:, :2]

    if n_his > 0:
        action_zero_seq = np.zeros([n_his, 2])
        action_seq = np.concatenate((action_zero_seq, action_seq), axis=0)



    v_pusher_3d = T_aug[:3, :3] @ v_pusher_3d
    v_pusher = v_pusher_3d[:2]
    q_pusher_3d_homog = T_aug @ q_pusher_3d_homog
    q_pusher = q_pusher_3d_homog[:2]

    return {'q_pusher': q_pusher,
            'v_pusher': v_pusher,
            'action_sequence': action_seq,
            }


def sample_pusher_velocity_3D():
    vel_min = 0.15
    vel_max = 0.25
    magnitude = random_sample_in_range(vel_min, vel_max)

    angle_max = np.deg2rad(30)
    angle_min = -angle_max
    angle = random_sample_in_range(angle_min, angle_max)

    vel_3d = magnitude * np.array([np.cos(angle), np.sin(angle), 0])
    return vel_3d



def generate_initial_condition(config=None,
                               T_aug_enabled=False,
                               push_length=0.4,
                               n_his=0,  # whether to add some zero actions at the beginning
                               randomize_velocity=False,
                               randomize_sdf=True,
                               randomize_color=True,
                               upright=False,
                               ):
    T_aug = np.eye(4)
    if T_aug_enabled:
        bound = 0.25
        pos_min = np.array([-bound, -bound, 0])
        pos_max = np.array([bound, bound, 0])
        yaw_min = 0
        yaw_max = 2*np.pi
        T_aug = MultiEpisodeDataset.sample_T_aug(pos_min=pos_min,
                                                 pos_max=pos_max,
                                                 yaw_min=yaw_min,
                                                 yaw_max=yaw_max,)



    q_slider = sample_object_position(T_aug=T_aug, upright=upright)
    print("q_slider", q_slider)

    vel_avg = 0.2
    dt = config['env']['step_dt']
    N = math.ceil(push_length/(vel_avg * dt))
    d = sample_pusher_position_and_velocity(T_aug=T_aug,
                                            N=N,
                                            n_his=n_his,
                                            randomize_velocity=randomize_velocity)

    action_sequence = None
    action_sequence = d['action_sequence']

    config_copy = copy.deepcopy(config)
    if randomize_sdf:
        config_copy['env']['model']['sdf'] = sample_random_mug()

    if randomize_color:
        config_copy['env']['model']['color'] = sample_random_color()

    return {'q_slider': q_slider,
            'q_pusher': d['q_pusher'],
            'v_pusher': d['v_pusher'],
            'action_sequence': action_sequence,
            'config': config_copy,
            }


def reset_environment(env, q_pusher, q_slider):
    context = env.get_mutable_context()
    env.reset()  # sets velocities to zero
    env.set_pusher_position(context, q_pusher)
    env.set_object_position(context, q_slider)
    # env.step(np.array([0, 0]), dt=0.1)  # allow box to drop down
    # input("debugging, press Enter to continue")
    env.step(np.array([0, 0]), dt=10.0)  # allow box to drop down


def get_color_intensity(n, N):
    return 1.0 - 0.5 * n * 1.0 / N


def load_model_state_dict(model_folder=None):

    models_root = "/home/manuelli/data/key_dynam/dev/experiments/20/dataset_correlle_mug-small_many_colors_600/trained_models/dynamics"

    # model_name = "DD_3D/2020-06-04-19-18-48-274487_DD_3D_z_state_n_his_2_no_T_aug"
    model_name = "DD_3D/2020-06-05-00-25-39-676089_DD_3D_all_n_his_2"
    # model_name = "DD_3D/2020-06-05-15-25-01-580144_DD_3D_all_z_n_his_2"
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


def keypoints_world_frame_to_object_frame(keypoints_W,
                                          T_W_obj):
    T_obj_W = np.linalg.inv(T_W_obj)
    keypoints_obj = transform_utils.transform_points_3D(T_obj_W, keypoints_W)
    return keypoints_obj


def main():
    # load dynamics model
    model_dict = load_model_state_dict()
    model = model_dict['model_dy']
    model_dd = model_dict['model_dd']
    config = model.config

    env_config = load_yaml(os.path.join(get_project_root(), 'experiments/exp_20_mugs/config.yaml'))
    env_config['env']['observation']['depth_int16'] = True
    n_history = config['train']['n_history']

    initial_cond = generate_initial_condition(env_config, push_length=PUSH_LENGTH)
    env_config = initial_cond['config']

    # enable the right observations

    camera_name = model_dict['metadata']['camera_name']
    spatial_descriptor_data = model_dict['spatial_descriptor_data']
    ref_descriptors = spatial_descriptor_data['spatial_descriptors']
    K = ref_descriptors.shape[0]

    ref_descriptors = torch.Tensor(ref_descriptors).cuda()  # put them on the GPU

    print("ref_descriptors\n", ref_descriptors)
    print("ref_descriptors.shape", ref_descriptors.shape)

    # create the environment
    # create the environment
    env = DrakeMugsEnv(env_config)
    env.reset()

    T_world_camera = env.camera_pose(camera_name)
    camera_K_matrix = env.camera_K_matrix(camera_name)

    # create another environment for doing rollouts
    env2 = DrakeMugsEnv(env_config, visualize=False)
    env2.reset()

    action_function = ActionFunctionFactory.function_from_config(config)
    observation_function = ObservationFunctionFactory.drake_pusher_position_3D(config)
    visual_observation_function = \
        VisualObservationFunctionFactory.descriptor_keypoints_3D(config=config,
                                                                 camera_name=camera_name,
                                                                 model_dd=model_dd,
                                                                 ref_descriptors=ref_descriptors,
                                                                 K_matrix=camera_K_matrix,
                                                                 T_world_camera=T_world_camera,
                                                                 )

    episode = OnlineEpisodeReader()
    mpc_input_builder = DynamicsModelInputBuilder(observation_function=observation_function,
                                                  visual_observation_function=visual_observation_function,
                                                  action_function=action_function,
                                                  episode=episode)

    vis = meshcat_utils.make_default_visualizer_object()
    vis.delete()

    reset_environment(env, initial_cond['q_pusher'], initial_cond['q_slider'])
    obs_init = env.get_observation()

    #### ROLLOUT USING LEARNED MODEL + GROUND TRUTH ACTIONS ############
    reset_environment(env, initial_cond['q_pusher'], initial_cond['q_slider'])
    # add just some large number of these
    episode.clear()
    for i in range(n_history):
        action_zero = np.zeros(2)
        obs_tmp = env.get_observation()
        episode.add_observation_action(obs_tmp, action_zero)

    def goal_func(obs_tmp):
        state_tmp = mpc_input_builder.get_state_input_single_timestep({'observation': obs_tmp})['state']
        return model.compute_z_state(state_tmp.unsqueeze(0))['z_object'].flatten()


    #
    idx = episode.get_latest_idx()
    obs_raw = episode.get_observation(idx)
    z_object_goal = goal_func(obs_raw)
    z_keypoints_init_W = keypoints_3D_from_dynamics_model_output(z_object_goal, K)
    z_keypoints_init_W = torch_utils.cast_to_numpy(z_keypoints_init_W)

    z_keypoints_obj = keypoints_world_frame_to_object_frame(z_keypoints_init_W,
                                                          T_W_obj=slider_pose_from_observation(obs_init))

    color = [1, 0, 0]
    meshcat_utils.visualize_points(vis=vis,
                                   name="keypoints_W",
                                   pts=z_keypoints_init_W,
                                   color=color,
                                   size=0.02,
                                   )

    # input("press Enter to continue")

    # rollout single action sequence using the simulator
    action_sequence_np = torch_utils.cast_to_numpy(initial_cond['action_sequence'])
    N = action_sequence_np.shape[0]
    obs_rollout_gt = env_utils.rollout_action_sequence(env, action_sequence_np)[
        'observations']

    # using the vision model to get "goal" keypoints
    z_object_goal = goal_func(obs_rollout_gt[-1])
    z_object_goal_np = torch_utils.cast_to_numpy(z_object_goal)
    z_keypoints_goal = keypoints_3D_from_dynamics_model_output(z_object_goal, K)
    z_keypoints_goal = torch_utils.cast_to_numpy(z_keypoints_goal)

    # visualize goal keypoints
    color = [0, 1, 0]
    meshcat_utils.visualize_points(vis=vis,
                                   name="goal_keypoints",
                                   pts=z_keypoints_goal,
                                   color=color,
                                   size=0.02,
                                   )

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


    action_seq_gt_torch = torch_utils.cast_to_torch(initial_cond['action_sequence'])
    action_input = torch.cat((action_init[:(n_history-1)], action_seq_gt_torch), dim=0).cuda()
    print("action_input.shape", action_input.shape)


    # rollout using the ground truth actions and learned model
    # need to add the batch dim to do that
    z_init = model.compute_z_state(state_init)['z']
    rollout_pred = rollout_model(state_init=z_init.unsqueeze(0),
                                 action_seq=action_input.unsqueeze(0),
                                 dynamics_net=model,
                                 compute_debug_data=True)

    state_pred_rollout = rollout_pred['state_pred']

    print("state_pred_rollout.shape", state_pred_rollout.shape)

    for i in range(N):
        # vis GT for now
        name = "GT_3D/%d" % (i)
        T_W_obj = slider_pose_from_observation(obs_rollout_gt[i])
        # print("T_W_obj", T_W_obj)

        # green
        color = np.array([0, 1, 0]) * get_color_intensity(i, N)
        meshcat_utils.visualize_points(vis=vis,
                                       name=name,
                                       pts=z_keypoints_obj,
                                       color=color,
                                       size=0.01,
                                       T=T_W_obj)

        # red
        color = np.array([0, 0, 1]) * get_color_intensity(i, N)
        state_pred = state_pred_rollout[:, i, :]
        pts_pred = keypoints_3D_from_dynamics_model_output(state_pred, K).squeeze()
        pts_pred = pts_pred.detach().cpu().numpy()
        name = "pred_3D/%d" % (i)
        meshcat_utils.visualize_points(vis=vis,
                                       name=name,
                                       pts=pts_pred,
                                       color=color,
                                       size=0.01,
                                       )

    # input("finished visualizing GT rollout\npress Enter to continue")
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
            if n_look_ahead == 0:
                break

            # start_time = time.time()
            # idx of current observation
            idx = episode.get_latest_idx()
            mpc_start_time = time.time()
            mpc_input_data = mpc_input_builder.get_dynamics_model_input(idx, n_history=n_history)
            state_cur = mpc_input_data['states']
            action_his = mpc_input_data['actions']

            if mpc_out is not None:
                action_seq_rollout_init = mpc_out['action_seq'][1:]
            else:
                action_seq_rollout_init = None

            # run MPPI
            z_cur = None
            with torch.no_grad():
                z_cur = model.compute_z_state(state_cur.unsqueeze(0).cuda())['z'].squeeze(0)



            mpc_out = planner.trajectory_optimization(state_cur=z_cur,
                                                      action_his=action_his,
                                                      obs_goal=z_object_goal_np,
                                                      model_dy=model,
                                                      action_seq_rollout_init=action_seq_rollout_init,
                                                      n_look_ahead=n_look_ahead,
                                                      eval_indices=object_indices,
                                                      rollout_best_action_sequence=True,
                                                      verbose=True,
                                                      )

            print("MPC step took %.4f seconds" %(time.time() - mpc_start_time))
            action_seq_mpc = mpc_out['action_seq'].cpu().numpy()


        # Rollout with ground truth simulator dynamics
        action_seq_mpc = torch_utils.cast_to_numpy(mpc_out['action_seq'])
        env2.set_simulator_state_from_observation_dict(env2.get_mutable_context(), obs_cur)
        obs_mpc_gt = env_utils.rollout_action_sequence(env2, action_seq_mpc)['observations']
        state_pred_mpc = torch_utils.cast_to_numpy(mpc_out['state_pred'])

        vis['mpc_3D'].delete()
        vis['mpc_GT_3D'].delete()

        L = len(obs_mpc_gt)
        print("L", L)
        if L == 0:
            break
        for i in range(L):
            # red
            color = np.array([1, 0, 0]) * get_color_intensity(i, L)
            state_pred = state_pred_mpc[i, :]
            state_pred = np.expand_dims(state_pred, 0)  # may need to expand dims here
            pts_pred = keypoints_3D_from_dynamics_model_output(state_pred, K).squeeze()
            name = "mpc_3D/%d" % (i)
            meshcat_utils.visualize_points(vis=vis,
                                           name=name,
                                           pts=pts_pred,
                                           color=color,
                                           size=0.01,
                                           )

            # ground truth rollout of the MPC action_seq
            name = "mpc_GT_3D/%d" % (i)
            T_W_obj = slider_pose_from_observation(obs_mpc_gt[i])

            # green
            color = np.array([1, 1, 0]) * get_color_intensity(i, L)
            meshcat_utils.visualize_points(vis=vis,
                                           name=name,
                                           pts=z_keypoints_obj,
                                           color=color,
                                           size=0.01,
                                           T=T_W_obj)

        action_cur = action_seq_mpc[0]

        print("action_cur", action_cur)
        # print("action_GT", initial_cond['action'])
        input("press Enter to continue")

        # add observation actions to the episode
        obs_cur = env.get_observation()
        episode.replace_observation_action(obs_cur, action_cur)

        # step the simulator
        env.step(action_cur)

        # visualize current keypoint positions
        obs_cur = env.get_observation()
        T_W_obj = slider_pose_from_observation(obs_cur)

        # yellow
        color = np.array([1, 1, 0])
        meshcat_utils.visualize_points(vis=vis,
                                       name="keypoint_cur",
                                       pts=z_keypoints_obj,
                                       color=color,
                                       size=0.02,
                                       T=T_W_obj)

        action_seq_mpc = action_seq_mpc[1:]
        state_pred_mpc = state_pred_mpc[1:]

    obs_final = env.get_observation()

    pose_error = compute_pose_error(obs_rollout_gt[-1],
                                    obs_final)

    print("position_error: %.3f"  %(pose_error['position_error']))
    print("angle error degrees: %.3f" %(pose_error['angle_error_degrees']))


if __name__ == "__main__":
    main()
