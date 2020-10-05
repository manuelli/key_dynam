# system
import os
import numpy as np
import pygame
import matplotlib.pyplot as plt


# drake
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder

# testing
from pydrake.multibody.tree import SpatialInertia, UnitInertia
from pydrake.multibody.plant import (
    AddMultibodyPlantSceneGraph, CoulombFriction)
from pydrake.geometry import Box
from pydrake.math import RigidTransform, RollPitchYaw

# key_dynam
from key_dynam.envs.drake_pusher_slider import DrakePusherSliderDiagramWrapper, DrakePusherSliderEnv
from key_dynam.utils.utils import get_project_root, load_yaml
from key_dynam.utils.vis_utils import ImageVisualizer

def process_pygame_events():
    # print("processing pygame events")
    # print("pygame.key.get_focused()", pygame.key.get_focused())
    action = np.array([0., 0.])
    pressed = pygame.key.get_pressed()


    velocity = 1.0
    # print("pressed", pressed)


    if pressed[pygame.K_LEFT]:
        action += np.array([-velocity, 0])

    if pressed[pygame.K_RIGHT]:
        action += np.array([velocity, 0])

    if pressed[pygame.K_UP]:
        action += np.array([0, velocity])

    if pressed[pygame.K_DOWN]:
        action += np.array([0, -velocity])

    pygame.event.pump()  # process event queue
    return action

def run_gym_env():
    """
    Runs the gym env
    :return:
    :rtype:
    """
    DEBUG_PRINTS = True
    USE_PYGAME = True

    try:

        # setup pygame thing
        if USE_PYGAME:
            pygame.init()
            screen = pygame.display.set_mode((640, 480))  # needed for grabbing focus
            clock = pygame.time.Clock()




        default_action = np.zeros(2)
        velocity = 0.2

        if USE_PYGAME:
            action = process_pygame_events()
        else:
            action = default_action

        # config = load_yaml(os.path.join(get_project_root(), 'experiments/05/config.yaml'))
        config = load_yaml(os.path.join(get_project_root(), 'experiments/exp_10/config.yaml'))
        env = DrakePusherSliderEnv(config)
        env.reset()


        # set the box pose
        context = env.get_mutable_context()
        pos = np.array([1.56907481e-04, 1.11390697e-06, 5.11972761e-02])
        quat = np.array([ 7.13518047e-01, -6.69765583e-07, -7.00636851e-01, -6.82079212e-07])
        q_slider = np.concatenate((quat, pos))
        env.set_slider_position(context, q=q_slider)



        env.simulator.set_target_realtime_rate(1.0)

        # move box araound
        # context = env.get_mutable_context()
        # q_slider = [-0.05, 0, 0.03]

        num_model_instances = env._dps_wrapper.mbp.num_model_instances()
        print("num_model_instances", num_model_instances)
        print("num_positions", env._dps_wrapper.mbp.num_positions())

        label_db = env._dps_wrapper.get_label_db()
        print("label db:", label_db.all())

        mask_db = env._dps_wrapper.get_labels_to_mask()
        print("mask db:", mask_db.all())

        # context = env.get_mutable_context()
        #
        # # set the position of pusher
        # q_pusher = [0.2,0.2]
        # mbp = env._dps_wrapper.mbp
        # mbp_context = env.diagram.GetMutableSubsystemContext(mbp, context)
        # mbp.SetPositions(mbp_context, env._dps_wrapper.models['pusher'], q_pusher)

        camera_names = list(config['env']['rgbd_sensors']['sensor_list'].keys())
        camera_names.sort()

        image_vis = ImageVisualizer(len(camera_names), 1)


        print("running sim")

        while True:
            if USE_PYGAME:
                action = velocity*process_pygame_events()
            else:
                action = default_action

            # print("action:", action)
            obs, reward, done, info = env.step(action)

            # print("obs\n", obs)

            # visualize RGB images in matplotlib
            for idx, camera_name in enumerate(camera_names):
                rgb_image = obs['images'][camera_name]['rgb']
                image_vis.draw_image(idx, 0, rgb_image)

            image_vis.visualize_interactive()


            # # print unique depth values
            # depth_32F = obs['images']['camera_0']['depth_32F']
            # print("unique depth_32F vals", np.unique(depth_32F))
            # # print unique depth values
            # depth_16U = obs['images']['camera_0']['depth_16U']
            # print("unique depth_16U vals", np.unique(depth_16U))


        # build simulator
    except KeyboardInterrupt:
        pygame.quit()
        plt.close()

def test_procedural_geometry():
    """
    This test ensures we can draw procedurally added primitive
    geometry that is added to the world model instance (which has
    a slightly different naming scheme than geometry with a
    non-default / non-world model instance).
    """
    builder = DiagramBuilder()
    mbp, scene_graph = AddMultibodyPlantSceneGraph(builder)
    world_body = mbp.world_body()
    box_shape = Box(1., 2., 3.)
    # This rigid body will be added to the world model instance since
    # the model instance is not specified.
    box_body = mbp.AddRigidBody("box", SpatialInertia(
        mass=1.0, p_PScm_E=np.array([0., 0., 0.]),
        G_SP_E=UnitInertia(1.0, 1.0, 1.0)))
    mbp.WeldFrames(world_body.body_frame(), box_body.body_frame(),
                   RigidTransform())
    mbp.RegisterVisualGeometry(
        box_body, RigidTransform.Identity(), box_shape, "ground_vis",
        np.array([0.5, 0.5, 0.5, 1.]))
    mbp.RegisterCollisionGeometry(
        box_body, RigidTransform.Identity(), box_shape, "ground_col",
        CoulombFriction(0.9, 0.8))
    mbp.Finalize()

    # frames_to_draw = {"world": {"box"}}
    # visualizer = builder.AddSystem(PlanarSceneGraphVisualizer(scene_graph))
    # builder.Connect(scene_graph.get_pose_bundle_output_port(),
    #                 visualizer.get_input_port(0))

    diagram = builder.Build()


if __name__ == "__main__":
    # run_interactive()
    # run_New()
    run_gym_env()
    print("FINISHED_NORMALLY")
    # test_procedural_geometry()