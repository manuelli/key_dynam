import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from matplotlib import cm
import copy

# key_dynam
from key_dynam.utils import utils
from key_dynam.utils import meshcat_utils
from key_dynam.experiments.exp_22_push_box_hardware.visualize_dynamics_model import \
    visualize_model_prediction_single_timestep

# pdc
from dense_correspondence_manipulation.utils.visualization import draw_reticles
from dense_correspondence_manipulation.utils.constants import DEPTH_IM_SCALE
from dense_correspondence_manipulation.utils import meshcat_utils as pdc_meshcat_utils

class PlanContainer:

    def __init__(self,
                 data,
                 dynamics_model_input_builder=None,
                 debug=False,
                 model_dy=None,
                 visualize=True,
                 ):
        self._data = data
        self._plan_data = data['plan_data']
        self._dynamics_model_input_builder = dynamics_model_input_builder
        self.model_dy = model_dy

        self._debug = debug
        self._visualize = visualize
        if visualize:
            self._vis = meshcat_utils.make_default_visualizer_object()

        self.counter = 0
        self._camera_name = "d415_01"

        self.process_plan(debug=self._debug)

    @property
    def plan_length(self):
        return len(self._plan_data)

    @property
    def data(self):
        return self._plan_data

    def process_plan(self,
                     debug=False,
                     vis_keypoints=True,
                     vis_pointclouds=False,
                     vis_first_pointcloud=True,
                     display_robot_state=False,
                     save_static_html=False):
        """
        Compute observation/visual_observation function for each entry

        """

        plan_start_time = self._plan_data[0]['observation']['timestamp_system']

        if self._visualize:
            self._vis.delete()



        self._timestamps = []
        for i, data in enumerate(self._plan_data):
            print("processing plan frame %d" %(i))

            # modifies data inplace by adding "dynamics_model_input_data" field
            self._dynamics_model_input_builder.get_state_input_single_timestep(data)

            state = data['dynamics_model_input_data']['dynamics_model_input']
            z_dict = self.model_dy.compute_z_state(state)
            data['dynamics_model_input_data']['z'] = z_dict


            action_tensor = self._dynamics_model_input_builder.get_action_tensor(data)
            data['action_tensor'] = action_tensor

            self._timestamps.append(data['observation']['timestamp_system'] - plan_start_time)

            image_obs = data['observations']['images'][self._camera_name]
            if self._visualize and ('K' in image_obs):

                name = "plan/pointclouds/%d" %(i)
                if vis_pointclouds or ((i == 0) and vis_first_pointcloud): # hack
                    rgb = np.copy(image_obs['rgb'])
                    depth = np.copy(image_obs['depth_int16']) / DEPTH_IM_SCALE
                    pdc_meshcat_utils.visualize_pointcloud(self._vis,
                                                           name,
                                                           depth=depth,
                                                           K=image_obs['K'],
                                                           rgb=rgb,
                                                           T_world_camera=image_obs['T_world_camera'])

                name_prefix = "plan/state"
                # color = [128,0,128]
                # if i == (len(self._plan_data) - 1):
                #     color = [0,255,0]

                # not sure why this isn't working
                val = i*1.0/len(self._plan_data)
                color_tmp = self.get_color(val)
                name_prefix = "plan_state_%d" %(i)
                visualize_model_prediction_single_timestep(self._vis,
                                                           self.model_dy.config,
                                                           z_dict['z'],
                                                           display_idx=i,
                                                           name_prefix=name_prefix,
                                                           color=color_tmp,
                                                           display_robot_state=display_robot_state)



            # if debug:
            #     rgb = np.copy(data['observation']['images']['d415_01']['rgb'])
            #     uv = data['dynamics_model_input_data']['visual_observation']['uv']
            #     draw_reticles(rgb, uv[:,0], uv[:,1], label_color=[255,0,0])
            #     plt.figure()
            #     plt.imshow(rgb)
            #
            #     plt.show(block=False)
            #     plt.pause(3)
            #     plt.close()
            #     # plt.show()


        # plan timestamps
        if save_static_html:
            filename = os.path.join(utils.get_project_root(), 'sandbox/meshcat.html')
            html_string = self._vis.static_html()
            print("len(html_string)", len(html_string))
            with open(filename, 'w') as f:
                f.write(html_string)

        self._timestamps = np.array(self._timestamps)

    def increment_counter(self):
        self.counter += 1

    def get_trajectory_goal(self,
                            counter,
                            n_look_ahead,
                            ): # torch.Tensor [n_look_ahead, z_state_dim]
        """
        Get trajectory goal
        Repeat final state if counter + n_look_ahead > plan_length
        """

        start_idx = counter + 1
        end_idx = counter + n_look_ahead

        plan_length = self.plan_length
        tensor_list = []
        for plan_idx in range(start_idx, end_idx + 1):
            plan_idx = min(plan_idx, plan_length-1)
            z = self._plan_data[plan_idx]['dynamics_model_input_data']['z']['z']
            tensor_list.append(z)


        traj_goal = torch.stack(tensor_list, dim=0)

        return traj_goal

    def get_final_state_goal(self,
                             ): # torch.Tensor [goal_dim]
        """
        Goal at the final state
        """
        return self._plan_data[-1]['dynamics_model_input_data']['z']['z'].clone()


    def get_save_data(self,
                      ): # dict
        """
        Converts the plan into data we can save
        """


        return {'plan_data': self._plan_data,
                'config': self.model_dy.config,
                'camera_name': self._camera_name,
                }

    def get_color(self, val):
        # color = colormap(val)
        color = cm.winter(val)
        color = [int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)]
        return color













