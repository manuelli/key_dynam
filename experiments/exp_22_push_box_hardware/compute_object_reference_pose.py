import numpy as np
from key_dynam.utils.utils import load_pickle, save_pickle, save_yaml
from key_dynam.utils import meshcat_utils
from key_dynam.utils import transform_utils

data_file = "/home/manuelli/data/key_dynam/hardware_experiments/closed_loop_rollouts/stable/2020-07-10-22-16-08_long_push_on_long_side/mpc_rollouts/2020-07-10-22-19-03-591910/data.p"

data = load_pickle(data_file)
pts = data['plan']['plan_data'][-1]['dynamics_model_input_data']['visual_observation']['pts_W']
print("pts\n", pts)


centroid = np.mean(pts, axis=0)
pts_centered = pts - centroid
save_data = {'object_points': pts_centered.tolist()}
save_file = "object_points_master.yaml"
save_yaml(save_data, save_file)


# do some meshcat debug
vis = meshcat_utils.make_default_visualizer_object()
meshcat_utils.visualize_points(vis,
                               "object_points_centered",
                               pts_centered,
                               color=[0,0,255],
                               size=0.01
                               )


meshcat_utils.visualize_points(vis,
                               "object_points_world",
                               pts,
                               color=[0,255,0],
                               size=0.01
                               )


# do procustes
T = transform_utils.procrustes_alignment(pts_centered, pts)
pts_transformed = transform_utils.transform_points_3D(T,
                                                      pts_centered)

meshcat_utils.visualize_points(vis,
                               "object_points_aligned",
                               pts_centered,
                               color=[255,0,0],
                               size=0.01,
                               T=T
                               )
print("T\n", T)
