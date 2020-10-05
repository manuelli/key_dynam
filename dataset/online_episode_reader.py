from collections import deque
import copy

from key_dynam.dataset.episode_reader import EpisodeReader

class OnlineEpisodeReader(EpisodeReader):

    def __init__(self,
                 no_copy=True,):
        super(OnlineEpisodeReader, self).__init__()
        self._trajectory = deque()
        self._no_copy = no_copy

    @property
    def trajectory(self):
        return self._trajectory

    def __len__(self):
        return len(self.trajectory)

    def get_latest_idx(self):
        return len(self) - 1

    def add_data(self, data):
        self._trajectory.append(data)

    def add_observation_action(self, obs, action):
        d = None
        if self._no_copy:
            d = {"observation": obs,
                 "action": action}
        else:
            d = {"observation": obs.copy(),
                 "action": action.copy()}

        self._trajectory.append(d)

    def add_observation_only(self, obs):
        d = None
        if self._no_copy:
            d = {"observation": obs,
                 "action": None}
        else:
            d = {"observation": obs.copy(),
                 "action": None}
        self._trajectory.append(d)

    def replace_observation_action(self, obs, action):
        d = None
        if self._no_copy:
            d = {"observation": obs,
                 "action": action}
        else:
            d = {"observation": obs.copy(),
                 "action": action.copy()}
        self._trajectory[-1] = d # replace last one

    def replace_action(self, action):
        if self._no_copy:
            self._trajectory[-1]['action'] = action
        else:
            self._trajectory[-1]['action'] = copy.deepcopy(action)

    def reduce_list_size(self,
                         max_size, # int: discard old observations until you are below min size
                         ):
        """
        Discard old observations to reduce the list below a maxsize
        """
        while len(self.trajectory) > max_size:
            self.trajectory.popleft()

    def get_action(self, idx):
        return self.trajectory[idx]['action']

    def get_observation(self, idx):
        return self.trajectory[idx]['observation']

    def get_data(self, idx):
        return self.trajectory[idx]

    def clear(self):
        self._trajectory = deque()

    def get_save_data(self,
                      ): # dict
        """
        Returns the data so we can save it out
        """
        return {'trajectory': self._trajectory,
                }
