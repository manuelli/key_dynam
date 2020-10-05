# system
import math

# torch
import torch

# key_dynam
from key_dynam.utils import torch_utils


def sample_constant_action_sequences_grid(sequence_length,
                                          add_zero_sample=True,
                                          vel_min=0.15,
                                          vel_max=0.25,
                                          vel_step=0.01,
                                          angle_min=0,
                                          angle_max=2 * math.pi,
                                          angle_step=math.radians(1)
                                          ):  # torch Tensor [num_samples, sequence_length, 2]

    """
    Samples action sequences which have a constant action for the entire trajectory
    Sequences are sampled from a uniform grid in both velocity and angle
    """

    # sample pusher velocity
    magnitude_samples = torch.arange(vel_min, vel_max, step=vel_step)
    M = len(magnitude_samples)

    angle_samples = torch.arange(angle_min, angle_max, step=angle_step)
    N = len(angle_samples)

    # [N, M]
    magnitude_samples = magnitude_samples.unsqueeze(0).expand([N, M])

    # [N, M]
    angle_samples = angle_samples.unsqueeze(1).expand([N, M])

    magnitude_samples_flat = magnitude_samples.flatten()
    angle_samples_flat = angle_samples.flatten()

    # [N*M,2]
    # the unsqueeze is needed so that torch broadcasting functions correctly
    vel = magnitude_samples_flat.unsqueeze(-1) * torch.stack(
        (torch.cos(angle_samples_flat), torch.sin(angle_samples_flat)), dim=-1)

    # expand to be an action_seq
    # [N*M, sequence_length, 2]
    action_seq = vel.unsqueeze(1).expand([-1, sequence_length, 2])

    if add_zero_sample:
        action_seq = torch.cat([action_seq, torch.zeros([1, sequence_length, 2])])

    return action_seq


def sample_action_sequences_MPPI(init_act_seq,  # unnormalized, shape: [B, n_look_ahead - 1, action_dim]
                                 n_sample,  # integer, number of action trajectories to sample
                                 action_lower_lim,  # unnormalized, shape: action_dim, lower limit of the action
                                 action_upper_lim,  # unnormalized, shape: action_dim, upper limit of the action
                                 noise_type="normal",
                                 beta_filter=None,
                                 sigma=None
                                 ):  # tensor: [n_sample, n_look_ahead, action_dim]
    """
    Samples action sequences to test out with the MPC
    """

    device = init_act_seq.device
    B, n_look_ahead, action_dim = init_act_seq.shape

    num_repeat = math.ceil(n_sample / B)
    n_sample = num_repeat * B

    # [n_sample, n_look_ahead, action_dim]
    act_seqs = init_act_seq.repeat(num_repeat, 1, 1)

    # need to redo this . . . .

    # [n_sample, action_dim]
    # n_t in MPPI paper
    act_residual = torch.zeros([n_sample, action_dim], dtype=init_act_seq.dtype).to(device=device)

    # actions that go as input to the dynamics network
    for i in range(n_look_ahead):

        # noise = u_t in MPPI paper
        noise_sample = None
        if noise_type == 'normal':
            distn = torch.distributions.normal.Normal(loc=0, scale=sigma)

            # [n_sample, action_dim]
            noise_sample = distn.sample(sample_shape=[n_sample, action_dim]).to(device)
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
        # have to do this since torch doesn't support clamp with
        # vector min and max
        for j in range(action_dim):
            act_seqs[:, i, j] = torch.clamp(act_seqs[:, i, j], min=action_lower_lim[j], max=action_upper_lim[j])


    # act_seqs: [n_sample, n_look_ahead, action_dim]
    return act_seqs


def concatenate_action_his(action_his, # torch.Tensor [n_his, action_dim]
                           action_seqs, # torch.Tensor [B, n_look_ahead, action_dim]
                           ): # torch.Tensor [B, n_his + n_look_ahead, action_dim]

    if len(action_his) == 0:
        return action_seqs

    B = action_seqs.shape[0]
    action_his = torch_utils.cast_to_torch(action_his).to(action_seqs.device)

    # [B, n_his, action_dim]
    action_his_expand = action_his.unsqueeze(0).expand([B, -1, -1])

    action_seqs_cat = torch.cat((action_his_expand, action_seqs), dim=1)
    return action_seqs_cat

def evaluate_model_rollout(state_pred,  # [B, rollout_len, state_dim]
                           obs_goal,  # [obs_dim, ] or [rollout_len, state_dim]
                           eval_indices=None,
                           terminal_cost_only=True,  # whether to compute the loss only at final idx
                           p=2,  # which norm to use, [L1, L2, etc.]
                           normalize=False,
                           goal_type=None, # for compatibility with hardware
                           ):
    # print("state_pred.shape", state_pred.shape)
    # print("obs_goal.shape", obs_goal.shape)
    # print("eval_indices", eval_indices)

    if eval_indices is not None:
        state_pred = state_pred[:, :, eval_indices]

    # [B, state_dim]
    delta = None
    reward = None
    if terminal_cost_only:

        # [B, rollout_len, state_dim]
        delta = state_pred - obs_goal
        delta = delta[:, -1]
        reward = -torch.norm(delta, dim=-1, p=p)

        if normalize:
            D = delta.shape[-1]
            reward = reward*1.0/D
    else:
        # [B, rollout_len, state_dim]
        delta = state_pred - obs_goal
        reward = -torch.norm(delta, dim=-1, p=p)

        if normalize:
            D = delta.shape[-1]
            reward = reward*1.0/D

        # print('before mean reduction: reward.shape', reward.shape)
        reward = torch.mean(reward, dim=-1)
        # print('reward.shape', reward.shape)

    best_idx = torch.argmax(reward)
    best_reward = torch.max(reward)

    return {'reward': reward,
            'best_idx': best_idx,
            'best_reward': best_reward,
            }
