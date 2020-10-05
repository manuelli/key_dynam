import os
import torch
import time
import numpy as np
import scipy.stats as stats
import math
import functools
import copy

# key_dynam
from key_dynam.dynamics.models_dy import rollout_model, rollout_action_sequences
import key_dynam.planner.utils as planner_utils
from key_dynam.utils import torch_utils

DEBUG = False


class Planner(object):

    def __init__(self, config):
        self._config = config
        self.action_dim = config['dataset']['action_dim']
        self.state_dim = config['dataset']['state_dim']
        self.n_his = config['train']['n_history']

    @property
    def config(self):
        return self._config
    
    @config.setter
    def config(self, val):
        self._config = val

    def model_rollout(self,
                      state_cur_np,  # the current state, numpy, unnormalized, shape: [n_his, state_dim]
                      model_dy,  # the learned dynamics model
                      act_seqs_np,  # the sampled action sequences, unnormalized, shape: [n_sample, -1, action_dim]
                      n_look_ahead,  # the number of look ahead steps, integer
                      use_gpu):
        pass

    def evaluate_traj(self, obs_seqs, obs_goal):
        # obs_seqs: [n_sample, n_look_ahead, state_dim]
        # obs_goal: state_dim
        pass

    def optimize_action(self, act_seqs, reward_seqs):
        pass

    def trajectory_optimization(self,
                                state_cur,  # current state, shape: [n_his, state_dim]
                                action_his,  # action_his # [n_his - 1, action_dim]
                                obs_goal,  # goal, shape: [state_dim]
                                model_dy,
                                **kwargs,
                                ):
        raise NotImplementedError("subclass must implement")

        return {'action_seq': action_seq,  # best action sequence
                'state_pred': state_pred,  # state prediction under chosen action sequence
                'action_seq_with_n_his': action_seq,
                'reward': reward,  # reward of best action sequence
                'debug': debug_info,  # debug information
                }


class PlannerMPPI(Planner):

    def __init__(self,
                 config,
                 grid_action_sampler=None,  # action sampler used to do additional sampling
                 ):
        super(PlannerMPPI, self).__init__(config)

        self._grid_action_sampler = grid_action_sampler
        if grid_action_sampler is None:
            c = config['mpc']['mppi']
            angle_min = math.radians(c['angle_min_deg'])
            angle_max = math.radians(c['angle_max_deg'])
            angle_step = math.radians(c['angle_step_deg'])
            self._grid_action_sampler = functools.partial(planner_utils.sample_constant_action_sequences_grid,
                                                          vel_min=c['vel_min'],
                                                          vel_max=c['vel_max'],
                                                          vel_step=c['vel_step'],
                                                          angle_min=angle_min,
                                                          angle_max=angle_max,
                                                          angle_step=angle_step,
                                                          )

        self.trajectory_optimization = self.trajectory_optimization_new

    @staticmethod
    def sample_action_sequences_static(init_act_seq,
                                       # np.array, shape: [n_look_ahead, action_dim]. This is \mu_t in PDDM paper
                                       N=None,  # int, length of action sequence
                                       n_sample=None,  # integer, number of action trajectories to sample
                                       sigma=None,  # covariance matrix for u_t^i samples
                                       beta=None,
                                       action_lower_lim=None,
                                       # (optional) np.array of shape [action_dim,] limits to clip the action to
                                       action_upper_lim=None,
                                       # (optional) np.array of shape [action_dim,] limits to clip the action to
                                       noise_type="normal",

                                       ):  # tensor: [n_sample, N, action_dim]
        """
        Samples action sequences to test out with the MPC.
        """

        assert N is not None
        assert beta is not None
        assert sigma is not None

        H, action_dim = init_act_seq.shape

        # [n_sample, N, action_dim]
        act_seqs = np.stack([init_act_seq] * n_sample)

        # [n_sample, action_dim]
        # n_t in MPPI paper
        act_residual = np.zeros([n_sample, action_dim])

        # actions that go as input to the dynamics network
        for i in range(N):

            # noise = u_t in MPPI paper
            noise = 0
            if noise_type == "uniform":
                raise NotImplementedError

                # make time correlated random perturbation
                # this is in the [-1,1] range
                noise = (np.random.rand(n_sample, action_dim) - 0.5) * 2

                # scale it to be in some range
                # where does this magic 0.8 come from???
                noise *= (action_upper_lim - action_lower_lim) * 0.8
            elif noise_type == "normal":

                # [n_sample, action_dim]
                noise = np.random.multivariate_normal(np.zeros(action_dim), sigma, size=n_sample)
            else:
                raise ValueError("unknown noise type: %s" % (noise_type))

            # print("noise.shape", noise.shape)
            # noise = u_t in MPPI paper

            # act_residual = n_t in MPPI paper
            # should we clip act_residual also . . . , probably not since it is zero centered
            act_residual = beta * noise + act_residual * (1.0 - beta)

            # add the perturbation to the action sequence
            act_seqs[:, i] += act_residual

            # clip to range
            if (action_lower_lim is not None) and (action_upper_lim is not None):
                act_seqs[:, i] = np.clip(act_seqs[:, i], action_lower_lim, action_upper_lim)

        # always re-sample the one that was passed in . .
        act_seqs[:, 0] = init_act_seq

        # act_seqs: [n_sample, N, action_dim]
        return act_seqs

    def sample_action_sequences(self,
                                init_act_seq,  # unnormalized, shape: [n_his + n_look_ahead - 1, action_dim]
                                n_sample,  # integer, number of action trajectories to sample
                                action_lower_lim,  # unnormalized, shape: action_dim, lower limit of the action
                                action_upper_lim,  # unnormalized, shape: action_dim, upper limit of the action
                                noise_type="normal"):  # tensor: [n_sample, -1, action_dim]
        """
        Samples action sequences to test out with the MPC
        Doesn't touch the first n_his actions in the sequence
        """

        # HACK for now
        init_act_seq = torch_utils.cast_to_numpy(init_act_seq)

        beta_filter = self.config['mpc']['mppi']['beta_filter']

        # [n_sample, -1, action_dim]
        act_seqs = np.stack([init_act_seq] * n_sample)

        # [n_sample, action_dim]
        # n_t in MPPI paper
        act_residual = np.zeros([n_sample, self.action_dim])

        # for i in range(self.n_his - 1, init_act_seq.shape[0]):

        # only add noise to future actions init_act_seq[:(n_his-1)] are past
        # The action we are optimizing for the current timestep is in fact
        # act_seq[n_his - 1].

        # actions that go as input to the dynamics network
        for i in range(self.n_his - 1, init_act_seq.shape[0]):

            # noise = u_t in MPPI paper
            noise_sample = None
            if noise_type == "uniform":

                # make time correlated random perturbation
                # this is in the [-1,1] range
                noise = (np.random.rand(n_sample, self.action_dim) - 0.5) * 2

                # scale it to be in some range
                # where does this magic 0.8 come from???
                noise *= (action_upper_lim - action_lower_lim) * 0.8
            elif noise_type == "normal":
                sigma = self.config['mpc']['mppi']['action_sampling']['sigma']

                # [n_sample, action_dim]
                noise_sample = np.random.normal(0, sigma, (n_sample, self.action_dim))

            else:
                raise ValueError("unknown noise type: %s" % (noise_type))

            # print("noise.shape", noise.shape)
            # noise = u_t in MPPI paper

            # act_residual = n_t in MPPI paper
            # should we clip act_residual also . . . , probably not since it is zero centered
            act_residual = beta_filter * noise_sample + act_residual * (1. - beta_filter)

            # add the perturbation to the action sequence
            act_seqs[:, i] += act_residual

            # clip to range
            act_seqs[:, i] = np.clip(act_seqs[:, i], action_lower_lim, action_upper_lim)

        # act_seqs: [n_sample, -1, action_dim]
        return act_seqs

    def model_rollout(self,
                      state_cur_np,  # the current state, numpy, unnormalized, shape: [n_his, state_dim]
                      model_dy,  # the learned dynamics model
                      act_seqs_np,  # the sampled action sequences, unnormalized, shape: [n_sample, -1, action_dim]
                      n_look_ahead,  # the number of look ahead steps, integer
                      use_gpu):
        if DEBUG:
            print("\n----MPPIPlanner.model_rollout-------")

        if use_gpu:
            device = 'cuda'
        else:
            device = 'cpu'

        # NOTE: We no longer deal with state and action tensor normalization. Should have
        # already been taken care of before

        # need to expand states_cur to have batch_size dimension
        # currently has shape [n_his, state_dim]
        states_cur = torch.tensor(state_cur_np, device=device).float()
        n_his, state_dim = states_cur.shape

        # cast action and observation to torch tensors
        # [n_sample, -1, action_dim]
        act_seqs = torch_utils.cast_to_torch(act_seqs_np).to(device).float()

        # trim it to have the correct shape
        # [n_sample, n_his + (n_look_ahead - 1), action_dim]
        act_seqs = act_seqs[:, :(n_his + (n_look_ahead - 1))]
        n_sample = act_seqs.shape[0]

        # expand to have shape [n_sample, n_his, state_dim]
        states_cur = states_cur.unsqueeze(0).expand([n_sample, -1, -1])

        # rollout the model without gradients
        out = None
        with torch.no_grad():
            out = rollout_model(states_cur,
                                act_seqs,
                                model_dy,
                                compute_debug_data=True,  # for now
                                )

        # [n_sample, n_look_ahead, state_dim]
        state_pred_np = out['state_pred'].cpu().detach().numpy()

        out['state_pred_np'] = state_pred_np
        return out

    def evaluate_traj(self, obs_seqs, obs_goal):
        """
        Computes the reward as negative of l2 distance between obs_seqs[:, -1] and goal
        :param obs_seqs:
        :type obs_seqs:
        :param obs_goal:
        :type obs_goal:
        :return:
        :rtype:
        """
        # obs_seqs: [n_sample, n_look_ahead, state_dim]
        # obs_goal: state_dim
        # reward_seqs = -np.sum((obs_seqs[:, -1] - obs_goal) ** 2, 1)
        reward_seqs = -np.linalg.norm((obs_seqs[:, -1] - obs_goal), axis=1)

        # reward_seqs: n_sample
        return reward_seqs

    def optimize_action(self,
                        act_seqs,  # shape: [n_sample, -1, action_dim]
                        reward_seqs  # shape: [n_sample]
                        ):

        act_seqs = torch_utils.cast_to_torch(act_seqs)
        reward_seqs = torch_utils.cast_to_torch(reward_seqs)

        reward_weight = self.config['mpc']['mppi']['reward_weight']

        # [n_sample, 1, 1]
        reward_seqs_exp = torch.exp(reward_weight * reward_seqs).reshape(-1, 1, 1)

        # [-1, action_dim]
        act_seq = (reward_seqs_exp * act_seqs).sum(0) / (torch.sum(reward_seqs_exp) + 1e-10)
        return act_seq

    # def trajectory_optimization_old(self,
    #                                 state_cur,  # current state, shape: [n_his, state_dim]
    #                                 action_his,  # action_his # [n_his - 1, action_dim]
    #                                 obs_goal,  # goal, shape: m = len(eval_indices) [m] or [n_look_ahead, m]
    #                                 model_dy,  # the learned dynamics model
    #                                 action_seq_rollout_init=None,  # optional, nominal future action sequence
    #                                 n_sample=None,  # number of action sequences to sample for each update iter
    #                                 n_look_ahead=None,  # number of look ahead steps
    #                                 n_update_iter=None,  # number of update iteration
    #                                 action_lower_lim=None,
    #                                 action_upper_lim=None,
    #                                 use_gpu=True,
    #                                 eval_indices=None,
    #                                 # optional set of indices to use for evaluating the cost, make sure you properly set obs_goal in this case as well
    #                                 reward_scale_factor=1,
    #                                 # factor by which to scale obs_pred and obs_goal before computing reward
    #                                 rollout_best_action_sequence=False,
    #                                 # whether to compute rollout for best action sequence
    #                                 verbose=True,
    #                                 add_constant_action_samples=False,
    #                                 **kwargs,  # placeholder for other things
    #                                 ):
    #
    #     if use_gpu:
    #         device = 'cuda'
    #     else:
    #         device = 'cpu'
    #
    #     # fill in defaults
    #     if n_sample is None:
    #         n_sample = self.config['mpc']['mppi']['n_sample']
    #
    #     if n_update_iter is None:
    #         n_update_iter = self.config['mpc']['mppi']['n_update_iter']
    #
    #     if action_lower_lim is None:
    #         action_lower_lim = np.array(self.config['mpc']['mppi']['action_lower_lim'])
    #
    #     if action_upper_lim is None:
    #         action_upper_lim = np.array(self.config['mpc']['mppi']['action_upper_lim'])
    #
    #     if n_look_ahead is None:
    #         n_look_ahead = self.config['mpc']['n_look_ahead']
    #
    #     if action_seq_rollout_init is None:
    #         print("using random shooting planner")
    #         start_time_tmp = time.time()
    #         # use random shooting planner to sample the initial sequence
    #         random_shooting = RandomShootingPlanner(self.config)
    #         rs_out = random_shooting.trajectory_optimization(state_cur=state_cur,
    #                                                          action_his=action_his,
    #                                                          obs_goal=obs_goal,
    #                                                          model_dy=model_dy,
    #                                                          n_look_ahead=n_look_ahead,
    #                                                          eval_indices=eval_indices,
    #                                                          return_debug_info=True,
    #                                                          use_gpu=True,
    #                                                          )
    #
    #         # [n_look_ahead, action_dim]
    #         action_seq_rollout_init = rs_out['action_seq'].cpu()
    #         print("RandomShooting planner took %.2f seconds" % (time.time() - start_time_tmp))
    #         print("RandomShooting planner reward ", rs_out['reward'].item())
    #
    #     n_look_ahead = action_seq_rollout_init.shape[0]
    #
    #     constant_action_seq = None
    #     if add_constant_action_samples:
    #         config = self.config
    #         angle_min = math.radians(config['mpc']['random_shooting']['angle_min_deg'])
    #         angle_max = math.radians(config['mpc']['random_shooting']['angle_max_deg'])
    #         angle_step = math.radians(config['mpc']['random_shooting']['angle_step_deg'])
    #         constant_action_seq = planner_utils.sample_constant_action_sequences_grid(sequence_length=n_look_ahead,
    #                                                                                   vel_min=
    #                                                                                   config['mpc']['random_shooting'][
    #                                                                                       'vel_min'],
    #                                                                                   vel_max=
    #                                                                                   config['mpc']['random_shooting'][
    #                                                                                       'vel_max'],
    #                                                                                   vel_step=
    #                                                                                   config['mpc']['random_shooting'][
    #                                                                                       'vel_step'],
    #                                                                                   angle_min=angle_min,
    #                                                                                   angle_max=angle_max,
    #                                                                                   angle_step=angle_step)
    #
    #         if len(action_his > 0):
    #             B, _, _ = constant_action_seq.shape
    #             act_his_expand = action_his.unsqueeze(0).expand([B, -1, -1])
    #             constant_action_seq = torch.cat((act_his_expand, constant_action_seq), dim=1)
    #             constant_action_seq = constant_action_seq.to(device)
    #
    #     """
    #
    #     act_seq has dimensions [n_his + n_look_ahead, action_dim]
    #
    #     so act_seq[:n_his] matches up with state_cur
    #     """
    #     action_seq_rollout_init = torch.Tensor(action_seq_rollout_init)
    #     act_seq = torch.cat((action_his, action_seq_rollout_init), dim=0)
    #
    #     if DEBUG:
    #         print("\n\n MPPIPlanner.trajectory_optimization")
    #
    #     if DEBUG:
    #         print("act_seq\n", act_seq)
    #         print("act_seq.shape", act_seq.shape)
    #
    #     debug_data_list = []
    #
    #     for i in range(n_update_iter):
    #         start_time_tmp = time.time()
    #         if DEBUG:
    #             print("\n\n")
    #             print("iteration:", i)
    #         # sample action sequences
    #         # [n_samples, n_his + n_look_ahead - 1, action_dim]
    #
    #         act_seqs = self.sample_action_sequences(act_seq, n_sample, action_lower_lim, action_upper_lim)
    #         act_seqs = torch_utils.cast_to_torch(act_seqs).to(device)
    #
    #         # add the constant_action_seq samples on the first go around if it's warranted
    #         if i == 0 and (constant_action_seq is not None):
    #             # print("type(act_seqs)", type(act_seqs))
    #             # print("act_seqs.shape", act_seqs.shape)
    #             # print("act_seqs.device", act_seqs.device)
    #             # print("constant_action_se.device", constant_action_seq.device)
    #             act_seqs = torch_utils.cast_to_torch(act_seqs)
    #             act_seqs = torch.cat((act_seqs, constant_action_seq), dim=0)
    #             # print("act_seqs.shape", act_seqs.shape)
    #
    #         # rollout using the sampled action sequences and the learned model
    #         # [n_samples, n_look_ahead, state_dim]
    #         time_tmp = time.time()
    #         with torch.no_grad():
    #             out = self.model_rollout(state_cur, model_dy, act_seqs, n_look_ahead, use_gpu)
    #         obs_seqs = out['state_pred_np']
    #
    #         if DEBUG:
    #             print("act_seqs.shape", act_seqs.shape)
    #             print("obs_seqs.shape", obs_seqs.shape)
    #             print("np.max(act_seqs):", np.max(act_seqs))
    #             print("np.max(obs_seqs):", np.max(obs_seqs))
    #
    #         # if np.max(act_seqs) == np.nan:
    #         #     raise ValueError("encountered NAN in act_seqs")
    #
    #         # allows you to select just some of the indices to evaluate the cost on
    #         obs_seqs_all = obs_seqs
    #         if eval_indices is not None:
    #             obs_seqs = obs_seqs[:, :, eval_indices]
    #
    #         if DEBUG:
    #             print("obs_seqs.shape v2:", obs_seqs.shape)
    #
    #         # calculate reward for each sampled trajectory
    #         # note that these should have reasonable size
    #
    #         # cast to numpy if it isn't already
    #         # we have a messy mix of torch and numpy code . . .
    #         # obs_goal = torch_utils.cast_to_numpy(obs_goal)
    #
    #         # print("type(obs_goal)", type(obs_goal))
    #         obs_goal_torch = torch_utils.cast_to_torch(obs_goal).to(out['state_pred'].device)
    #         reward_data = planner_utils.evaluate_model_rollout(state_pred=out['state_pred'],
    #                                                            obs_goal=obs_goal_torch,
    #                                                            eval_indices=eval_indices,
    #                                                            terminal_cost_only=self.config['mpc']['mppi'][
    #                                                                'terminal_cost_only'],
    #                                                            p=self.config['mpc']['mppi']['cost_norm']
    #                                                            )
    #
    #         # print("reward_data.keys()", reward_data.keys())
    #         # print("reward_data['reward'].shape", reward_data['reward'].shape)
    #         reward_seqs = reward_scale_factor * reward_data['reward']
    #         # reward_seqs = torch_utils.cast_to_numpy(reward_seqs)
    #         # reward_seqs = reward_scale_factor * self.evaluate_traj(obs_seqs, obs_goal)
    #
    #         if DEBUG:
    #             print('reward_seqs.shape', reward_seqs.shape)
    #
    #         if verbose:
    #             print('update_iter %d/%d, max: %.4f, mean: %.4f, std: %.4f, time: %.4f' % (
    #                 i, n_update_iter, torch.max(reward_seqs), torch.mean(reward_seqs), torch.std(reward_seqs),
    #                 time.time() - start_time_tmp))
    #
    #         # optimize the action sequence according to the rewards
    #         # note: (manuelli) this seems to actually be making things worse for some
    #         # settings of the 'reward_weight' parameter
    #         act_seq = self.optimize_action(act_seqs, reward_seqs)
    #
    #         reward_best, idx_best = torch.max(reward_seqs, dim=0)
    #
    #         # print("reward_best: %.4f, idx_best: %d" %(reward_best, idx_best))
    #
    #         debug_data = {'action_sequence_samples': act_seqs,
    #                       'action_sequence_optimized': act_seq,
    #                       'reward_seqs': reward_seqs,
    #                       'obs_seqs': obs_seqs,
    #                       'action_seq_best': act_seqs[idx_best],
    #                       'reward_best': reward_best,
    #                       'idx_best': idx_best,
    #                       'obs_seqs_best': obs_seqs[idx_best],
    #                       }
    #
    #         debug_data_list.append(debug_data)
    #
    #     debug_data = debug_data_list[-1]
    #     idx_best = debug_data['idx_best']
    #     action_seq_best = debug_data['action_seq_best']
    #     reward_best = debug_data['reward_best']
    #     obs_seq_best = debug_data['obs_seqs_best']
    #
    #     # # observation sequence for the best action sequence
    #     # # that was found
    #     #
    #     # if rollout_best_action_sequence:
    #     #     act_seq_tensor = torch.tensor(act_seq, device='cuda').unsqueeze(0)
    #     #
    #     #     out = self.model_rollout(state_cur, model_dy, act_seq_tensor, n_look_ahead, use_gpu)
    #     #     obs_seq_best = out['state_pred_np']
    #     #
    #     #     reward_seq_best = reward_scale_factor * self.evaluate_traj(obs_seq_best[:, :, eval_indices], obs_goal)
    #     #     reward_best_tmp = np.max(reward_seq_best)
    #     #     obs_seq_best = out['state_pred_np'].squeeze(axis=0)
    #     #     print("reward best from MPPI avg", reward_best_tmp)
    #
    #     action_seq_rollout = action_seq_best[self.n_his - 1:]
    #     return {'action_seq': action_seq_rollout.cpu(),
    #             'action_seq_full': action_seq_best,
    #             'debug_data_list': debug_data_list,
    #             'observation_sequence': obs_seq_best,  # [n_roll, obs_dim]
    #             'state_pred': obs_seq_best,
    #             'reward': reward_best,
    #             }

    def trajectory_optimization_new(self,
                                    state_cur,  # current state, shape: [n_his, state_dim]
                                    action_his,  # action_his # [n_his - 1, action_dim]
                                    obs_goal,  # goal, shape: m = len(eval_indices) [m] or [n_look_ahead, m]
                                    model_dy,  # the learned dynamics model
                                    action_seq_rollout_init=None,
                                    # optional, nominal future action sequence [n_look_ahead, action_dim]
                                    n_sample=None,  # number of action sequences to sample for each update iter
                                    n_look_ahead=None,  # number of look ahead steps
                                    n_update_iter=None,  # number of update iteration
                                    action_lower_lim=None,
                                    action_upper_lim=None,
                                    use_gpu=True,
                                    eval_indices=None,
                                    # optional set of indices to use for evaluating the cost, make sure you properly set obs_goal in this case as well
                                    reward_scale_factor=1,
                                    # factor by which to scale obs_pred and obs_goal before computing reward
                                    rollout_best_action_sequence=False,
                                    # whether to compute rollout for best action sequence
                                    verbose=True,
                                    add_grid_action_samples=True,
                                    **kwargs,  # placeholder for other things
                                    ):

        if verbose:
            print("\n\n-----MPPI-----")

        start_time = time.time()
        if use_gpu:
            device = 'cuda'
        else:
            device = 'cpu'

        # fill in defaults
        if n_sample is None:
            n_sample = self.config['mpc']['mppi']['n_sample']

        if n_update_iter is None:
            n_update_iter = self.config['mpc']['mppi']['n_update_iter']

        if action_lower_lim is None:
            action_lower_lim = np.array(self.config['mpc']['mppi']['action_lower_lim'])

        if action_upper_lim is None:
            action_upper_lim = np.array(self.config['mpc']['mppi']['action_upper_lim'])

        if n_look_ahead is None:
            n_look_ahead = self.config['mpc']['n_look_ahead']

        n_his = state_cur.shape[0]
        obs_goal = torch_utils.cast_to_torch(obs_goal).to(device)
        c = self.config['mpc']['mppi']
        mppi_config = self.config['mpc']['mppi']

        # initial action sequence samples
        action_his_idx = list(range(n_his - 1))
        action_future_idx = list(range(n_his - 1, n_his + n_look_ahead - 1))
        action_seq_init_list = []

        ################# SAMPLE ACTIONS AROUND INIT SEQUENCE #################
        # only do this if action_seq_rollout_init was passed in
        if action_seq_rollout_init is not None:
            print("action_seq_rollout_init\n", action_seq_rollout_init)
            action_seq_init_list.append(action_seq_rollout_init)

        # ############ SAMPLE ACTIONS USING GRID SAMPLER ###############
        # doing this tended to make things worse so I disabled it
        # action_seq_samples_grid = None
        # if add_grid_action_samples:
        #     action_seq_samples_grid = self._grid_action_sampler(n_look_ahead)
        # else:
        #     action_seq_samples_grid = torch.Tensor([])
        #
        # action_seq_samples_full = torch.cat((action_seq_samples_grid, action_seq_samples_init), dim=0).to(device)
        # action_seq_samples_full = planner_utils.concatenate_action_his(action_his, action_seq_samples_full)

        ############### RUN RANDOM SHOOTING PLANNER ##############
        # basically rolls out a bunch of action sequences and selects the best one
        # how is this being used?
        if (len(action_seq_init_list) == 0) or mppi_config['action_sampling']['add_samples_around_random_shooting_trajectory']:
            random_shooting = RandomShootingPlanner(self.config)
            rs_out = random_shooting.trajectory_optimization(state_cur=state_cur,
                                                             action_his=action_his,
                                                             obs_goal=obs_goal,
                                                             model_dy=model_dy,
                                                             n_look_ahead=n_look_ahead,
                                                             eval_indices=eval_indices,
                                                             use_gpu=use_gpu,
                                                             action_seq_rollout_init=action_seq_rollout_init)

            if verbose:
                print("-------Random Shooting Planner-------")
                print("reward: %.4f" % (rs_out['reward']))
                print("rs_out['action_seq_full']", rs_out['action_seq_full'])
                print("torch.norm(action_seq[0]]", torch.norm(rs_out['action_seq'][0]))

            action_seq_init_list.append(rs_out['action_seq'])

        # [B, n_look_ahead, action_dim], note that we could have B = 1
        # if len(action_seq_list) = 1
        action_seq_init_tensor = torch.stack(action_seq_init_list)


        if verbose:
            print("len(action_seq_init_list)", len(action_seq_init_list))
            print("action_seq_init_tensor.shape", action_seq_init_tensor.shape)

        # now sample action_seq_samples using MPPI sampler
        # these will be used the first time through the loop
        action_seq_samples_first_iter = planner_utils.sample_action_sequences_MPPI(
            init_act_seq=action_seq_init_tensor,
            n_sample=c['n_sample'],
            action_lower_lim=c['action_lower_lim'],
            action_upper_lim=c['action_upper_lim'],
            noise_type=c['action_sampling'][
                'noise_type'],
            beta_filter=c['beta_filter'],
            sigma=c['action_sampling']['sigma'],
        )


        debug_data_list = []
        iter_data = None

        for i in range(n_update_iter):
            start_time_tmp = time.time()
            if DEBUG:
                print("\n\n")
                print("iteration:", i)


            # action_seq_samples.shape = [B, n_look_ahead, action_dim]
            # if not passed in (should only be done in later iterations of the loop)

            action_seq_samples = None
            if i == 0:
                action_seq_samples = action_seq_samples_first_iter
            else:
                print("sampling new action sequences")
                action_seq_samples = planner_utils.sample_action_sequences_MPPI(
                    init_act_seq=iter_data["action_seq_cur"].unsqueeze(0),
                    n_sample=c['n_sample'],
                    action_lower_lim=c['action_lower_lim'],
                    action_upper_lim=c['action_upper_lim'],
                    noise_type=c['action_sampling'][
                        'noise_type'],
                    beta_filter=c['beta_filter'],
                    sigma=c['action_sampling']['sigma'],
                )

            # this could potentially be made more efficient
            # [B, n_his + n_look_ahead -1, action_dim]
            action_seq_samples_full = planner_utils.concatenate_action_his(action_his,
                                                                           action_seq_samples)

            # rollout using the sampled action sequences and the learned model
            # [n_samples, n_look_ahead, state_dim]
            with torch.no_grad():
                rollout_data = rollout_action_sequences(state_cur,
                                                        action_seq_samples_full,
                                                        model_dy=model_dy,
                                                        )

            reward_data = planner_utils.evaluate_model_rollout(state_pred=rollout_data['state_pred'],
                                                               obs_goal=obs_goal.to(device),
                                                               eval_indices=eval_indices,
                                                               **self.config['mpc']['reward'],
                                                               )

            reward_seqs = reward_scale_factor * reward_data['reward']

            if DEBUG:
                print('reward_seqs.shape', reward_seqs.shape)

            if verbose:
                print('update_iter %d/%d, max: %.4f, mean: %.4f, std: %.4f, time: %.4f' % (
                    i, n_update_iter, torch.max(reward_seqs), torch.mean(reward_seqs), torch.std(reward_seqs),
                    time.time() - start_time_tmp))

            # optimize the action sequence according to the rewards
            # note: (manuelli) this seems to actually be making things worse for some
            # settings of the 'reward_weight' parameter
            action_seq_cur_full = self.optimize_action(action_seq_samples_full, reward_seqs)
            reward_best, idx_best = torch.max(reward_seqs, dim=0)

            # print("reward_best: %.4f, idx_best: %d" %(reward_best, idx_best))

            state_pred_samples = rollout_data['state_pred']

            # could potentially make this more efficient, not store it around . . . 
            debug_data = {'action_sequence_samples_full': action_seq_samples_full,
                          'action_sequence_samples': action_seq_samples_full[:, action_future_idx],
                          'action_sequence_optimized': action_seq_cur_full,
                          'reward_seqs': reward_seqs,
                          'obs_seqs': state_pred_samples,
                          'action_seq_best': action_seq_samples_full[idx_best],
                          'reward_best': reward_best,
                          'idx_best': idx_best,
                          'obs_seqs_best': state_pred_samples[idx_best],
                          }

            debug_data_list.append(debug_data)

            # could potentially make this more efficient, not store it around . . . 
            iter_data = {'action_seq_samples_full': action_seq_samples_full,
                         'reward_seqs': reward_seqs,
                         'obs_seqs': state_pred_samples,
                         'action_seq_cur_full': action_seq_cur_full,
                         'action_seq_cur': action_seq_cur_full[action_future_idx]
                         }

            
            #### end for

        # compute best action sequence
        reward_best, idx_best = torch.max(iter_data['reward_seqs'], dim=0)
        action_seq_best_full = iter_data['action_seq_samples_full'][idx_best]
        action_seq_best = action_seq_best_full[action_future_idx]
        obs_seq_best = iter_data['obs_seqs'][idx_best]

        if verbose:
            print("mpc took %.3f seconds" % (time.time() - start_time))

        return {'action_seq': action_seq_best,
                'action_seq_full': action_seq_best_full,
                'debug_data_list': debug_data_list,
                'observation_sequence': obs_seq_best,  # [n_roll, obs_dim]
                'state_pred': obs_seq_best,
                'reward': reward_best,
                }


class RandomShootingPlanner():

    def __init__(self,
                 config,
                 action_sampler=None,  # action sampler that takes n_look_ahead as input
                 ):

        self._config = config

        self._action_sampler = action_sampler
        if action_sampler is None:
            angle_min = math.radians(config['mpc']['random_shooting']['angle_min_deg'])
            angle_max = math.radians(config['mpc']['random_shooting']['angle_max_deg'])
            angle_step = math.radians(config['mpc']['random_shooting']['angle_step_deg'])
            self._action_sampler = functools.partial(planner_utils.sample_constant_action_sequences_grid,
                                                     vel_min=config['mpc']['random_shooting']['vel_min'],
                                                     vel_max=config['mpc']['random_shooting']['vel_max'],
                                                     vel_step=config['mpc']['random_shooting']['vel_step'],
                                                     angle_min=angle_min,
                                                     angle_max=angle_max,
                                                     angle_step=angle_step,
                                                     )

    @property
    def config(self):
        return self._config

    def trajectory_optimization(self,
                                state_cur,  # current state, shape: [n_his, state_dim]
                                action_his,  # action_his # [n_his - 1, action_dim]
                                obs_goal,  # goal, shape: [obs_dim,] (could be different than state_dim)
                                model_dy,  # the learned dynamics model
                                n_look_ahead,
                                eval_indices=None,  # the indices of state_pred to use for evaluation
                                use_gpu=True,
                                return_debug_info=False,  # if true then return debugging info
                                action_seq_rollout_init=None,
                                verbose=False,
                                # torch tensor [B, n_look_ahead, action_dim] additional action sequences to try
                                **kwargs,  # for backwards compatibility
                                ):

        # [B, n_look_ahead, action_dim]
        action_seq_samples = self._action_sampler(n_look_ahead)

        # n_history = self.config['train']['n_history']
        n_history = state_cur.shape[0]

        # convert things to tensors if they aren't already
        # this is needed since if x is a torch.cuda.Tensor then
        # calling torch.Tensor(x) leads to a segfault, see
        # https://github.com/pytorch/pytorch/issues/33899
        state_cur = torch_utils.cast_to_torch(state_cur)
        action_his = torch_utils.cast_to_torch(action_his)
        obs_goal = torch_utils.cast_to_torch(obs_goal)

        if action_seq_rollout_init is not None:
            if action_seq_rollout_init.dim() == 2:
                action_seq_rollout_init = action_seq_rollout_init.unsqueeze(0)

            _, n_look_ahead_tmp, action_dim_tmp = action_seq_rollout_init.shape

            assert action_seq_rollout_init.shape[1] == action_seq_samples.shape[1]
            assert action_seq_rollout_init.shape[2] == action_seq_samples.shape[2]

            # print("action_seq_samples.device", action_seq_samples.device)
            # print("action_seq_rollout_init.device", action_seq_rollout_init.device)
            device = action_seq_samples.device
            action_seq_samples = torch.cat((action_seq_samples, action_seq_rollout_init.to(device)), dim=0)

        # expand them out
        if len(action_his > 0):
            action_seq_samples = planner_utils.concatenate_action_his(action_his, action_seq_samples)

        if use_gpu:
            state_cur = state_cur.cuda()
            action_seq_samples = action_seq_samples.cuda()
            obs_goal = obs_goal.cuda()

        # rollout the model
        rollout_data = rollout_action_sequences(state_cur,
                                                action_seq_samples,
                                                model_dy=model_dy
                                                )

        # evaluate the action sequences
        reward_data = planner_utils.evaluate_model_rollout(rollout_data['state_pred'],
                                                           obs_goal,
                                                           eval_indices=eval_indices,
                                                           **self.config['mpc']['reward'],
                                                           )

        idx = reward_data['best_idx']
        reward = reward_data['best_reward']
        action_seq_with_n_his = action_seq_samples[idx]
        action_seq = action_seq_with_n_his[n_history - 1:]
        state_pred = rollout_data['state_pred'][idx]

        if verbose:
            print("reward:", reward)

        debug_info = None
        if return_debug_info:
            debug_info = {'action_seq_sample': action_seq_samples,
                          'rollout_data': rollout_data,
                          'reward_data': reward_data,
                          }

        return {'action_seq': action_seq,  # best action sequence
                'action_seq_full': action_seq_with_n_his,
                'state_pred': state_pred,  # state prediction under chosen action sequence
                'action_seq_with_n_his': action_seq,
                'reward': reward,  # reward of best action sequence
                'debug': debug_info,  # debug information
                }


class GradientDescentPlanner(Planner):

    def __init__(self,
                 config,
                 action_sampler=None,  # action sampler that takes n_look_ahead as input
                 ):
        super(GradientDescentPlanner, self).__init__(config)

        self._action_sampler = action_sampler
        if action_sampler is None:
            angle_min = math.radians(config['mpc']['gradient_descent']['angle_min_deg'])
            angle_max = math.radians(config['mpc']['gradient_descent']['angle_max_deg'])
            angle_step = math.radians(config['mpc']['gradient_descent']['angle_step_deg'])
            self._action_sampler = functools.partial(planner_utils.sample_constant_action_sequences_grid,
                                                     vel_min=config['mpc']['gradient_descent']['vel_min'],
                                                     vel_max=config['mpc']['gradient_descent']['vel_max'],
                                                     vel_step=config['mpc']['gradient_descent']['vel_step'],
                                                     angle_min=angle_min,
                                                     angle_max=angle_max,
                                                     angle_step=angle_step,
                                                     )

    def trajectory_optimization(self,
                                state_cur,  # current state, shape: [n_his, state_dim]
                                action_his,  # action_his # [n_his - 1, action_dim]
                                obs_goal,  # goal, shape: [obs_dim,] (could be different than state_dim)
                                model_dy,  # the learned dynamics model
                                action_seq_rollout_init=None,  # [M, n_look_ahead, action_dim]
                                n_look_ahead=None,
                                n_update_iter=None,
                                eval_indices=None,  # the indices of state_pred to use for evaluation
                                use_gpu=True,
                                return_debug_info=False,  # if true then return debugging info
                                verbose=True,
                                **kwargs,  # for backwards compatibility
                                ):
        """
        Proceeds in several steps

        1. Sample a bunch of action sequences
        2. Optimize them using gradient descent
        3. Return the best one

        """

        start_time = time.time()
        device = None
        if use_gpu:
            device = 'cuda'
        else:
            device = 'cpu'

        if n_update_iter is None:
            n_update_iter = self.config['mpc']['gradient_descent']['n_update_iter']

        obs_goal = torch_utils.cast_to_torch(obs_goal).to(device)

        n_his = state_cur.shape[0]

        ### SAMPLE ACTION SEQUENCES
        # 2 types of sampling (i) random grid (ii) MPPI style based on existing one

        # [B, n_look_ahead, action_dim]
        action_seq_samples_grid = self._action_sampler(n_look_ahead)

        action_seq_samples_init = torch.Tensor([])  # empty tensor

        if action_seq_rollout_init is not None:
            if action_seq_rollout_init.dim() == 2:
                action_seq_rollout_init = action_seq_rollout_init.unsqueeze(0)

            c = self.config['mpc']['gradient_descent']
            action_seq_samples_init = \
                planner_utils.sample_action_sequences_MPPI(init_act_seq=action_seq_rollout_init,
                                                           n_sample=c['n_sample'],
                                                           action_lower_lim=c['action_lower_lim'],
                                                           action_upper_lim=c['action_upper_lim'],
                                                           noise_type=c['action_sampling'][
                                                               'noise_type'],
                                                           beta_filter=c['beta_filter'],
                                                           sigma=c['action_sampling']['sigma'], )

        # print("action_seq_samples_grid.shape", action_seq_samples_grid.shape)
        # print("action_seq_samples_init", action_seq_samples_init.shape)
        action_seq_samples = torch.cat((action_seq_samples_grid, action_seq_samples_init), dim=0).to(device)

        # print("action_seq_samples.shape", action_seq_samples.shape)
        # print("action_seq_samples.device", action_seq_samples.device)

        # add n_his actions to front
        action_seq_samples = planner_utils.concatenate_action_his(action_his, action_seq_samples)

        # make it a leaf and set it to require gradient
        action_seq_samples.requires_grad = True

        action_his_idx = list(range(n_his))
        action_future_idx = list(range(n_his - 1, action_seq_samples.shape[1]))
        # print("n_his", n_his)
        # print("action_future_idx", action_future_idx)
        # print("len(action_future_idx)", len(action_future_idx))
        params = [action_seq_samples]
        optimizer = torch.optim.SGD(params, lr=0.1)

        print("action_seq_samples.shape", action_seq_samples.shape)
        print("sampling action sequences took ", (time.time() - start_time))

        # be careful, print out number of trainable params

        debug_data_list = []

        action_seq_best = None
        reward_best = np.inf

        ### OPTIMIZE ACTION SEQUENCES
        for i in range(n_update_iter):
            # rollout the model
            start_time_tmp = time.time()
            rollout_data = rollout_action_sequences(state_cur,
                                                    action_seq_samples,
                                                    model_dy=model_dy,
                                                    )

            conf = self.config['mpc']['gradient_descent']
            reward_data = planner_utils.evaluate_model_rollout(rollout_data['state_pred'],
                                                               obs_goal,
                                                               eval_indices=eval_indices,
                                                               terminal_cost_only=conf[
                                                                   'terminal_cost_only'],
                                                               p=conf['cost_norm'])
            #
            # optimizer.zero_grad()
            # test_loss = torch.mean(rollout_data['state_pred'])
            # test_loss.backward()
            # print("test")

            reward_seqs = reward_data['reward']
            reward_best, idx_best = torch.max(reward_seqs, dim=0)

            action_seq_samples_detach = action_seq_samples.detach().clone()
            action_seq_best = action_seq_samples_detach[idx_best]
            debug_data = {'reward_data': reward_data,
                          'action_seq_samples_full': action_seq_samples_detach,
                          'action_seq_samples': action_seq_samples[:, action_future_idx],
                          'reward_best': reward_best.detach().clone(),
                          'idx_best': idx_best,
                          'action_seq_best': action_seq_best[action_future_idx],
                          'action_seq_best_full': action_seq_best,
                          'state_pred': rollout_data['state_pred'],
                          'state_pred_best': rollout_data['state_pred'][idx_best]
                          }

            debug_data_list.append(debug_data)

            if i < n_update_iter - 1:
                # take a gradient step
                optimizer.zero_grad()
                loss = -1.0 * torch.sum(reward_data['reward'])  # want to minimize the loss = max reward
                loss.backward()

                # zero out the gradient on the first n_his actions
                action_seq_samples.grad[:, action_his_idx] = 0
                optimizer.step()

            if verbose:
                print('update_iter %d/%d, max: %.4f, mean: %.4f, std: %.4f, time: %.4f' % (
                    i + 1, n_update_iter, torch.max(reward_seqs), torch.mean(reward_seqs), torch.std(reward_seqs),
                    time.time() - start_time_tmp))

        debug_data = debug_data_list[-1]

        if verbose:
            print("mpc took %.3f seconds" % (time.time() - start_time))

        return {'debug_data_list': debug_data_list,
                'action_seq': debug_data['action_seq_best'],
                'reward': debug_data['reward_best'],
                'state_pred': debug_data['state_pred_best']
                }


class CEMPlanner(Planner):

    def __init__(self,
                 config,
                 ):
        super(CEMPlanner, self).__init__(config)
        raise NotImplementedError("needs more testing")

    @staticmethod
    def compute_mean_var(action_seq_elites,  # [E, H, action_dim]
                         ):  # dict
        """
        Computes the mean and std-dev
        :param action_seq_elites:
        :type action_seq_elites:
        :return:
        :rtype:
        """

        E, H, action_dim = action_seq_elites.shape
        mean = np.zeros([H, action_dim])
        var = np.zeros([H, action_dim, action_dim])

        for t in range(H):
            # [E, action_dim]
            data = action_seq_elites[:, t]
            mean[t] = np.mean(data, axis=0)
            var[t] = np.cov(data, rowvar=True)

        return {'mean': mean,
                'var': var}

    @staticmethod
    def sample_action_sequences(mean,  # [H, action_dim]
                                var,  # [H, action_dim, action_dim]
                                N,  # num samples
                                ):

        H, action_dim = mean.shape
        action_seq = np.zeros([N, H, action_dim])

        for t in range(H):
            action_seq[:, t] = np.random.multivariate_normal(mean[t], var[t], size=N)

        return action_seq

    def rollout_model_and_select_elites(self,
                                        state_cur,  # [n_his, obs_dim]
                                        action_seq,  # [N, n_his + n_rollout-1, action_dim]
                                        model_dy,
                                        obs_goal,
                                        elite_fraction,
                                        eval_indices=None,
                                        ):

        N, T, action_dim = action_seq.shape

        n_his = self.config['train']['n_history']
        n_roll = T - (n_his - 1)

        # rollout the model
        rollout_data = rollout_action_sequences(state_cur,
                                                action_seq,
                                                model_dy=model_dy
                                                )

        # evaluate the action sequences
        reward_data = planner_utils.evaluate_model_rollout(rollout_data['state_pred'],
                                                           obs_goal,
                                                           eval_indices=eval_indices,
                                                           terminal_cost_only=self.config['mpc']['terminal_cost_only'],
                                                           p=self.config['mpc']['cost_norm'])

        # get idx of the elites
        reward = reward_data['reward']  # torch tensor

        # sort this tensor
        # high to low
        sorted_idx = torch.argsort(reward, descending=True)

        num_elites = math.ceil(elite_fraction * N)
        elites_idx = sorted_idx[:num_elites]

        return {'action_seq_elites': action_seq[elites_idx],
                'reward_elites': reward[elites_idx],
                }

    def sample_action_sequences_given_elites(self,
                                             action_seq_elites,  # [E, n_his + n_roll - 1, action_dim]
                                             num_samples,
                                             ):
        """
        Sample new action_sequence using the mean and variance computed from the
        'elites'. Keeps the first 'n_his-1' values the same

        :param action_seq_elites:
        :type action_seq_elites:
        :param num_samples:
        :type num_samples:
        :return:
        :rtype:
        """

        E, T, action_dim = action_seq_elites.shape
        action_seq = np.zeros([num_samples, T, action_dim])

        n_his = self.config['train']['n_history']

        # keep first n_his - 1 actions same as they were
        idx = n_his - 1
        action_seq[:, :idx] = action_seq_elites[0, :idx]

        elites_stats = CEMPlanner.compute_mean_var(action_seq_elites[:, idx:])

        # elites_stats['mean'] shape [E, n_roll, action_dim]
        action_seq_future = CEMPlanner.sample_action_sequences(elites_stats['mean'],
                                                               elites_stats['var'],
                                                               num_samples)

        action_seq[:, idx:] = action_seq_future

        return action_seq
