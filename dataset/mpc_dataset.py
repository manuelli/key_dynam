import torch
from torch.utils.data.dataloader import default_collate

class DynamicsModelInputBuilder(object):
    """
    Class to assist in constructing the input for the dynamics model
    when running live off of real data.
    """
    dynamics_model_input_key = "dynamics_model_input_data"

    def __init__(self,
                 observation_function, # function that returns a tensor
                 visual_observation_function=None, # function that returns a dict
                 action_function=None,
                 episode=None,  # EpisodeReader
                 ):

        self._episode = episode
        self._observation_function = observation_function
        self._visual_observation_function = visual_observation_function
        self._action_function = action_function


    def get_state_input_single_timestep(self,
                                        data_raw,  # raw observation/action dict
                                        ): # dict

        # just return it if it's already there
        key = DynamicsModelInputBuilder.dynamics_model_input_key
        if key in data_raw:
            return data_raw[key]


        obs_raw = data_raw['observation']
        obs_tensor = self._observation_function(obs_raw, data_raw=data_raw)


        visual_obs = None

        # grab the visual observation if it is specified
        if self._visual_observation_function:
            with torch.no_grad():
                visual_obs = self._visual_observation_function(obs_raw)
            dynamics_model_input = torch.cat((visual_obs['flattened'].to(obs_tensor.device), obs_tensor))
        else:
            dynamics_model_input = obs_tensor


        # create the data structure
        data = {'dynamics_model_input': dynamics_model_input, # [state_dim,]
                'visual_observation': visual_obs,
                'state': dynamics_model_input,
                }

        # store it in the environment observation so that
        # you can use it later if you query for the same observation again
        data_raw['dynamics_model_input_data'] = data

        return data

    def get_action_tensor(self,
                          data_raw):

        action_raw = data_raw['action']
        action_tensor = self._action_function(action_raw, data_raw=data_raw)

        return action_tensor


    def get_dynamics_model_input(self,
                                 idx, # int
                                 n_history, # int
                                 episode=None,
                                 ):

        if episode is None:
            episode = self._episode

        start_idx = idx - n_history + 1
        end_idx = start_idx + n_history
        input_tensor_list = []
        data_list = []

        action_tensor_list = []

        for i in range(start_idx, end_idx):
            data_raw = episode.get_data(i)
            data = self.get_state_input_single_timestep(data_raw)
            input_tensor_list.append(data['dynamics_model_input'])
            data_list.append(data)

            if i < (end_idx - 1):
                action_tensor = self.get_action_tensor(data_raw)
                action_tensor_list.append(action_tensor)


        input_tensor = default_collate(input_tensor_list)

        # deal with potentially empty tensors
        action_tensor = None
        if len(action_tensor_list) > 0:
            action_tensor = default_collate(action_tensor_list)
        else:
            action_tensor = torch.Tensor()

        return {'dynamics_model_input': input_tensor,
                'states': input_tensor, # [n_his, state_dim]
                'data_list': data_list,
                'actions': action_tensor, # [n_his-1, action_dim], could be empty
                }


    def get_action_state_tensors(self,
                                 start_idx,
                                 num_timesteps,
                                 episode=None,
                                 ):

        if episode is None:
            episode = self._episode

        end_idx = start_idx + num_timesteps
        input_tensor_list = []
        data_list = []

        action_tensor_list = []

        for i in range(start_idx, end_idx):
            data_raw = episode.get_data(i)
            data = self.get_state_input_single_timestep(data_raw)
            input_tensor_list.append(data['dynamics_model_input'])
            data_list.append(data)


            action_tensor = self.get_action_tensor(data_raw)
            action_tensor_list.append(action_tensor)


        input_tensor = default_collate(input_tensor_list)

        # deal with potentially empty tensors
        action_tensor = None
        if len(action_tensor_list) > 0:
            action_tensor = default_collate(action_tensor_list)
        else:
            action_tensor = torch.Tensor()

        return {'dynamics_model_input': input_tensor,
                'states': input_tensor, # [num_timesteps, state_dim]
                'data_list': data_list,
                'actions': action_tensor, # [num_timesteps, action_dim], could be empty
                }

