from __future__ import print_function

"""
A test of our dataset class using PusherSlider environment as a test
"""
import os
import numpy as np

# torch
import torch
from torchsummary import summary


from key_dynam.dataset.function_factory import ObservationFunctionFactory, ActionFunctionFactory
from key_dynam.utils.utils import load_yaml, get_project_root, load_pickle
from key_dynam.dynamics.utils import count_trainable_parameters, count_non_trainable_parameters
from key_dynam.dataset.episode_reader import PyMunkEpisodeReader
from key_dynam.dataset.episode_dataset import MultiEpisodeDataset
from key_dynam.dynamics.models_dy import DynaNetMLP
from key_dynam.dynamics.data_normalizer import DataNormalizer
from key_dynam.dataset.utils import load_episodes_from_config


def create_pusher_slider_dataset():
    # load some previously generated data

    action_function = ActionFunctionFactory.pusher_velocity
    obs_function = ObservationFunctionFactory.pusher_slider_pose

    project_root = get_project_root()
    config_file = os.path.join(project_root, "experiments/01/config.yaml")
    config = load_yaml(config_file)

    DATA_PATH = os.path.join(project_root, "test_data/pusher_slider_10_episodes/2019-10-22-21-30-02-536750.p")

    raw_data = load_pickle(DATA_PATH)
    episodes = PyMunkEpisodeReader.load_pymunk_episodes_from_raw_data(raw_data)

    # create MultiEpisodeDataset
    dataset = MultiEpisodeDataset(config, action_function=action_function,
                                  observation_function=obs_function,
                                  episodes=episodes)

    episode = dataset.get_random_episode()
    data_0 = episode.get_observation(0)
    data_1 = episode.get_observation(1)

    print("time 0", data_0["sim_time"])
    print("time 1", data_1["sim_time"])

    # episode_name = episodes.keys()[0]
    # episode = episodes[episode_name]
    # data = episode.data
    # print("episode.data.keys()", episode.data.keys())
    # print("test ", type(data["trajectory"][0].keys()))
    # print("test ", data["trajectory"][0].keys())
    return dataset, config

def create_pusher_slider_keypoint_dataset(config=None):
    # load some previously generated data



    project_root = get_project_root()
    if config is None:
        config_file = os.path.join(project_root, "experiments/02/config.yaml")
        config = load_yaml(config_file)

    action_function = ActionFunctionFactory.pusher_velocity
    obs_function = ObservationFunctionFactory.pusher_pose_slider_keypoints(config)

    DATA_PATH = os.path.join(project_root,
                             "test_data/pusher_slider_10_episodes/2019-10-22-21-30-02-536750.p")

    raw_data = load_pickle(DATA_PATH)
    episodes = PyMunkEpisodeReader.load_pymunk_episodes_from_raw_data(raw_data)

    # create MultiEpisodeDataset
    dataset = MultiEpisodeDataset(config, action_function=action_function,
                                  observation_function=obs_function,
                                  episodes=episodes)

    episode = dataset.get_random_episode()
    data_0 = episode.get_observation(0)
    data_1 = episode.get_observation(1)

    print("time 0", data_0["sim_time"])
    print("time 1", data_1["sim_time"])

    # episode_name = episodes.keys()[0]
    # episode = episodes[episode_name]
    # data = episode.data
    # print("episode.data.keys()", episode.data.keys())
    # print("test ", type(data["trajectory"][0].keys()))
    # print("test ", data["trajectory"][0].keys())
    return dataset, config

def test_pusher_slider_dataset():
    # dataset, config = create_pusher_slider_dataset()

    project_root = get_project_root()
    config_file = os.path.join(project_root, "experiments/01/config.yaml")
    config = load_yaml(config_file)

    # new dataset loading approach
    episodes = load_episodes_from_config(config)
    action_function = ActionFunctionFactory.function_from_config(config)
    observation_function = ObservationFunctionFactory.function_from_config(config)

    dataset = MultiEpisodeDataset(config,
                                  action_function=action_function,
                                  observation_function=observation_function,
                                  episodes=episodes,
                                  phase="train")

    data = dataset[0] # test the getitem
    print("type(data)", type(data))
    print("list(data)", list(data))

    print(type(data["observations"]))
    print("observations.shape",data["observations"].shape)
    print("actions.shape", data["actions"].shape)

    print("observations", data["observations"])
    print("actions", data["actions"])

    stats = dataset.compute_dataset_statistics()

    print("stats", stats)

    # create dataset normalizer

def test_pusher_slider_keypoint_dataset():
    project_root = get_project_root()
    config_file = os.path.join(project_root, "experiments/02/config.yaml")
    config = load_yaml(config_file)

    config["n_history"] = 1
    config["n_roll"] = 0

    # new dataset loading approach
    episodes = load_episodes_from_config(config)
    action_function = ActionFunctionFactory.function_from_config(config)
    observation_function = ObservationFunctionFactory.function_from_config(config)

    dataset = MultiEpisodeDataset(config,
                                  action_function=action_function,
                                  observation_function=observation_function,
                                  episodes=episodes,
                                  phase="train")

    # dataset, config = create_pusher_slider_keypoint_dataset(config=config)

    episode_names = dataset.get_episode_names()
    episode_names.sort()
    episode_name = episode_names[0]
    episode = dataset.episode_dict[episode_name]
    obs_raw = episode.get_observation(0)
    obs_raw['slider']['angle'] = 0

    dataset.observation_function(obs_raw)

    print("20 degrees\n\n\n\n")
    obs_raw['slider']['angle'] = np.deg2rad(90)
    dataset.observation_function(obs_raw)
    quit()

    data = dataset[0] # test the getitem
    print("type(data)", type(data))
    print("data.keys()", data.keys())

    print(type(data["observations"]))
    print("observations.shape",data["observations"].shape)
    print("actions.shape", data["actions"].shape)

    print("observations", data["observations"])
    print("actions", data["actions"])
    #
    # stats = dataset.compute_dataset_statistics()
    #
    # print("stats", stats)

    # create dataset normalizer


def test_dataset_keypoint_obs():
    dataset, config = create_pusher_slider_dataset()

    data = dataset[0]  # test the getitem
    print("type(data)", type(data))
    print("data.keys()", data.keys())

    print(type(data["observations"]))
    print("observations.shape", data["observations"].shape)
    print("actions.shape", data["actions"].shape)

    print("observations", data["observations"])
    print("actions", data["actions"])

    stats = dataset.compute_dataset_statistics()

    print("stats", stats)

    # create dataset normalizer



def test_dynanet_mlp():
    # just try doing a single forward pass
    dataset, config = create_pusher_slider_dataset()
    stats = dataset.compute_dataset_statistics()

    n_history = config["train"]["n_history"]
    # obs_mean_repeat = stats['observations']['mean'].repeat(n_history, 1)
    # obs_std_repeat = stats['observations']['std'].repeat(n_history, 1)
    obs_mean_repeat = stats['observations']['mean']
    obs_std_repeat = stats['observations']['std']
    observations_normalizer = DataNormalizer(obs_mean_repeat, obs_std_repeat)

    # action_mean_repeat = stats['actions']['mean'].repeat(n_history, 1)
    # action_std_repeat = stats['actions']['std'].repeat(n_history, 1)
    action_mean_repeat = stats['actions']['mean']
    action_std_repeat = stats['actions']['std']
    actions_normalizer = DataNormalizer(action_mean_repeat, action_std_repeat)


    config["dataset"]["state_dim"] = 5
    config["dataset"]["action_dim"] = 2
    model = DynaNetMLP(config)

    # print summary of model before adding new modules
    print("\n\n -----summary of model BEFORE adding normalization modules")
    print("num trainable parameters", count_trainable_parameters(model))
    print("num non-trainable parameters ", count_non_trainable_parameters(model))
    print("\n\n")

    # summary of model after adding new params
    model.set_action_normalizer(actions_normalizer)
    model.set_state_normalizer(observations_normalizer)

    print("\n\n -----summary of model AFTER adding normalization modules")
    print("num trainable parameters", count_trainable_parameters(model))
    print("num non-trainable parameters ", count_non_trainable_parameters(model))
    print("\n\n")

    # unsqueeze to mimic dataloader with batch size of 1
    data = dataset[0]  # test the getitem
    observations = data['observations'].unsqueeze(0)
    actions = data['actions'].unsqueeze(0)

    obs_slice = observations[:, :n_history, :]
    action_slice = actions[:, :n_history, :]

    print("action_slice.shape", action_slice.shape)
    print("obs_slice.shape", obs_slice.shape)

    # run the model forwards one timestep
    output = model.forward(obs_slice, action_slice)

    print("output.shape", output.shape)

    # save the model with torch.save and torch.load
    save_dir = os.path.join(get_project_root(), 'sandbox')
    model_save_file = os.path.join(save_dir, "model.pth")
    torch.save(model, model_save_file)

    # load the model
    model_load = torch.load(model_save_file)
    print("\n\n -----summary of model LOADED from disk")
    print("num trainable parameters", count_trainable_parameters(model_load))
    print("num non-trainable parameters ", count_non_trainable_parameters(model_load))
    print("\n\n")


    # now try doing the same but with the state dict
    # my hunch is that this won't work . . .
    params_save_file = os.path.join(save_dir, "model_params.pth")
    torch.save(model.state_dict(), params_save_file)

    # load the model
    model_load = DynaNetMLP(config)
    state_dict = torch.load(params_save_file)
    for param_tensor in state_dict:
        print(param_tensor, "\t", state_dict[param_tensor].size())

    # try creating some dummy DataNormalizer objects
    # model_load.set_state_normalizer(DataNormalizer(0.0,1.0))
    # model_load.set_action_normalizer(DataNormalizer(0.0,1.0))
    model_load.load_state_dict(state_dict)
    print("\n\n -----summary of model LOADED from disk with state_dict method")
    print("num trainable parameters", count_trainable_parameters(model_load))
    print("num non-trainable parameters ", count_non_trainable_parameters(model_load))
    print("\n\n")

    print("model_load._action_normalizer._mean", model_load.action_normalizer._mean)
    print("model._action_normalizer._mean", model.action_normalizer._mean)


if __name__ == "__main__":
    test_pusher_slider_dataset()
    # test_pusher_slider_keypoint_dataset()
    print("\n\n\n")
    # test_dynanet_mlp()