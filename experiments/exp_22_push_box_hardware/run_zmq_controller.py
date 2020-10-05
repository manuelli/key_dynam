import os
import pydrake # to avoid symbol collisions
import copy


import torch

# dense_correspondence
from dense_correspondence_manipulation.utils import meshcat_utils as pdc_meshcat_utils
from dense_correspondence_manipulation.utils.constants import DEPTH_IM_SCALE

# key_dynam
from key_dynam.utils.utils import get_project_root, load_yaml, get_current_YYYY_MM_DD_hh_mm_ss_ms, get_data_root, \
    load_pickle, get_spartan_camera_info
from key_dynam.models.model_builder import build_dynamics_model
from key_dynam.utils.torch_utils import get_freer_gpu, set_cuda_visible_devices
from key_dynam.utils import torch_utils
from key_dynam.dataset.dynamic_spartan_episode_reader import DynamicSpartanEpisodeReader
from key_dynam.dataset.function_factory import ObservationFunctionFactory, ActionFunctionFactory
from key_dynam.dataset.vision_function_factory import VisualObservationFunctionFactory
from key_dynam.experiments.exp_22_push_box_hardware.utils import get_dataset_paths
from key_dynam.controller.controller import ZMQController, PLAN_MSG_FILE, COMPUTE_CONTROL_ACTION_MSG_FILE
from key_dynam.planner.planners import RandomShootingPlanner, PlannerMPPI
from key_dynam.controller import zmq_utils


PLANNER_TYPE = "mppi"

def load_model_and_data(K_matrix=None,
                        T_world_camera=None,
                        ):

    dataset_name = "push_box_hardware"

    model_name = "DD_3D/2020-07-02-17-59-21-362337_DD_3D_n_his_2_T_aug"
    train_dir = os.path.join(get_data_root(), "dev/experiments/22/dataset_push_box_hardware/trained_models/dynamics")
    # train_dir = "/home/manuelli/data/key_dynam/dev/experiments/22/dataset_push_box_hardware/trained_models/dynamics"

    train_dir = os.path.join(train_dir, model_name)
    ckpt_file = os.path.join(train_dir, "net_best_dy_state_dict.pth")

    train_config = load_yaml(os.path.join(train_dir, 'config.yaml'))
    state_dict = torch.load(ckpt_file)

    # build dynamics model
    model_dy = build_dynamics_model(train_config)
    # print("state_dict.keys()", state_dict.keys())
    model_dy.load_state_dict(state_dict)
    model_dy = model_dy.eval()
    model_dy = model_dy.cuda()


    # load the dataset
    dataset_paths = get_dataset_paths(dataset_name)
    dataset_root = dataset_paths['dataset_root']
    episodes_config = dataset_paths['episodes_config']

    spatial_descriptor_data = load_pickle(os.path.join(train_dir, 'spatial_descriptors.p'))
    metadata = load_pickle(os.path.join(train_dir, 'metadata.p'))

    ref_descriptors = spatial_descriptor_data['spatial_descriptors']
    ref_descriptors = torch_utils.cast_to_torch(ref_descriptors).cuda()

    # dense descriptor model
    model_dd_file = metadata['model_file']
    model_dd = torch.load(model_dd_file)
    model_dd = model_dd.eval()
    model_dd = model_dd.cuda()

    camera_name = train_config['dataset']['visual_observation_function']['camera_name']

    camera_info = None
    if (T_world_camera is not None) and (K_matrix is not None):
        camera_info = {"K": K_matrix,
                       'T_world_camera': T_world_camera,}
    else:
        camera_info = get_spartan_camera_info(camera_name)

    camera_info['camera_name'] = camera_name
    visual_observation_function = \
        VisualObservationFunctionFactory.descriptor_keypoints_3D(config=train_config,
                                                                 camera_name=camera_name,
                                                                 model_dd=model_dd,
                                                                 ref_descriptors=ref_descriptors,
                                                                 K_matrix=camera_info['K'],
                                                                 T_world_camera=camera_info['T_world_camera'],
                                                                 )


    action_function = ActionFunctionFactory.function_from_config(train_config)
    observation_function = ObservationFunctionFactory.function_from_config(train_config)



    #### PLANNER #######
    planner = None
    # make a planner config
    planner_config = copy.copy(train_config)
    config_tmp = load_yaml(os.path.join(get_project_root(), 'experiments/exp_22_push_box_hardware/config_DD_3D.yaml'))
    planner_config['mpc'] = config_tmp['mpc']
    if PLANNER_TYPE == "random_shooting":
        planner = RandomShootingPlanner(planner_config)
    elif PLANNER_TYPE == "mppi":
        planner = PlannerMPPI(planner_config)
    else:
        raise ValueError("unknown planner type: %s" % (PLANNER_TYPE))

    return {"model_dy": model_dy,
            'config': train_config,
            'spatial_descriptor_data': spatial_descriptor_data,
            'action_function': action_function,
            'observation_function': observation_function,
            'visual_observation_function': visual_observation_function,
            'planner': planner,
            'camera_info': camera_info,
            }


def main():
    d = load_model_and_data()
    controller = ZMQController(config=d['config'],
                               model_dy=d['model_dy'],
                               action_function=d['action_function'],
                               observation_function=d['observation_function'],
                               visual_observation_function=d['visual_observation_function'],
                               planner=d['planner'],
                               debug=True,
                               zmq_enabled=True,
                               camera_info=d['camera_info'],
                               )

    controller.run()

def debug():

    save_dir = os.path.join(get_project_root(), 'sandbox/mpc/push_right_box_horizontal')

    save_dir = "/home/manuelli/data/key_dynam/sandbox/2020-07-07-20-09-54_push_right_box_horizontal"
    plan_msg = load_pickle(os.path.join(save_dir, 'plan_msg.p'), encoding='latin1')
    plan_msg = zmq_utils.convert(plan_msg)

    compute_control_action_msg = load_pickle(os.path.join(save_dir, 'compute_control_action_msg.p'), encoding='latin1')
    compute_control_action_msg = zmq_utils.convert(compute_control_action_msg)

    K_matrix = None
    T_world_camera = None

    if 'K_matrix' in plan_msg:
        K_matrix = plan_msg['K_matrix']

    if 'T_world_camera' in plan_msg:
        T_world_camera = plan_msg['T_world_camera']

    d = load_model_and_data(K_matrix=K_matrix,
                            T_world_camera=T_world_camera,
                            )


    controller = ZMQController(config=d['config'],
                               model_dy=d['model_dy'],
                               action_function=d['action_function'],
                               observation_function=d['observation_function'],
                               visual_observation_function=d['visual_observation_function'],
                               planner=d['planner'],
                               debug=True,
                               zmq_enabled=False,
                               camera_info=d['camera_info'],
                               )



    controller._on_plan_msg(plan_msg)
    controller._on_compute_control_action(compute_control_action_msg)



def test_trajectory_plan():
    d = load_model_and_data()
    controller = ZMQController(config=d['config'],
                               model_dy=d['model_dy'],
                               action_function=d['action_function'],
                               observation_function=d['observation_function'],
                               visual_observation_function=d['visual_observation_function'],
                               planner=d['planner'],
                               debug=True,
                               zmq_enabled=True,
                               camera_info=d['camera_info'],
                               )


    plan_msg_file = "/home/manuelli/data/key_dynam/hardware_experiments/demonstrations/stable/2020-07-09-20-27-09_push_right_blue_tapes/plan_msg.p"
    plan_msg = load_pickle(plan_msg_file, encoding='latin1')
    plan_msg = zmq_utils.convert(plan_msg)

    controller._on_plan_msg(plan_msg)
    traj_goal = controller._state_dict['plan'].get_trajectory_goal(15,5)
    print("traj_goal.shape", traj_goal.shape)

    plan_msg_file = "/home/manuelli/data/key_dynam/hardware_experiments/demonstrations/stable/2020-07-09-20-27-09_push_right_blue_tapes/plan_msg.p"
    plan_msg = load_pickle(plan_msg_file, encoding='latin1')
    plan_msg = zmq_utils.convert(plan_msg)


    for i in range(4):
        compute_control_action_msg = {'type': "COMPUTE_CONTROL_ACTION",
                                      'data': plan_msg['data']['plan_data'][i]}
        controller._on_compute_control_action(compute_control_action_msg,
                                              visualize=True)
        input("press Enter to continue")



if __name__ == "__main__":
    main()
    # test_trajectory_plan()
    # debug()