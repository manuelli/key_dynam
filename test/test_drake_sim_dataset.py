# must import pydrake BEFORE torch
import pydrake

# key_dynam
from key_dynam.utils import dev_utils
from key_dynam.dataset.episode_dataset import MultiEpisodeDataset
from key_dynam.dataset.function_factory import ObservationFunctionFactory, ActionFunctionFactory


multi_episode_dict = dev_utils.load_drake_pusher_slider_episodes()
config = dev_utils.load_simple_config()

action_function = ActionFunctionFactory.function_from_config(config)
observation_function = ObservationFunctionFactory.function_from_config(config)
dataset = MultiEpisodeDataset(config,
                              action_function=action_function,
                              observation_function=observation_function,
                              episodes=multi_episode_dict,
                              phase="train")


episode_name = dataset.get_episode_names()[0]
episode = dataset.episode_dict[episode_name]
idx = 5

data = dataset._getitem(episode,
                        idx,
                        rollout_length=5,
                        n_history=2,
                        visual_observation=False,
                        )
print("\n\ndata.keys()", data.keys())

data_w_vision = dataset._getitem(episode,
                        idx,
                        rollout_length=5,
                        n_history=2,
                        visual_observation=True,
                        )

print("\n\ndata_w_vision.items()", data_w_vision.keys())
print("finished_normally")