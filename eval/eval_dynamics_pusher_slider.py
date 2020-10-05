import os
import random
import numpy as np
import time
import copy

# from progressbar import ProgressBar

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from key_dynam.dataset.episode_dataset import MultiEpisodeDataset
from key_dynam.dynamics.utils import rand_int, count_trainable_parameters, Tee, AverageMeter, get_lr, to_np, set_seed
from key_dynam.dataset.function_factory import ObservationFunctionFactory, ActionFunctionFactory
from key_dynam.dataset.utils import load_drake_sim_episodes_from_config
from key_dynam.utils.utils import save_yaml, load_pickle, get_data_root
from key_dynam.dynamics.models_dy import rollout_model
from key_dynam.models.model_builder import build_dynamics_model

"""
Training for DrakePusherSlider without vision
"""

DEBUG = False

def save_model(model, save_base_path):
    # save both the model in binary form, and also the state dict
    torch.save(model.state_dict(), save_base_path + "_state_dict.pth")
    torch.save(model, save_base_path + "_model.pth")

def eval_dynamics(config,
                  eval_dir,  # str: directory to save output
                  multi_episode_dict=None,
                  n_rollout_list=None,
                  model_dy=None,  # should already be in eval mode
                  phase_list=None,  # typically it's
                  num_epochs=10,
                  ):

    assert n_rollout_list is not None
    assert model_dy is not None
    assert multi_episode_dict is not None

    if phase_list is None:
        phase_list = ["valid"]

    # set random seed for reproduction
    set_seed(config['train']['random_seed'])

    tensorboard_dir = os.path.join(eval_dir, "tensorboard")
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    writer = SummaryWriter(log_dir=tensorboard_dir)

    # save the config
    save_yaml(config, os.path.join(eval_dir, "config.yaml"))

    action_function = ActionFunctionFactory.function_from_config(config)
    observation_function = ObservationFunctionFactory.function_from_config(config)

    use_gpu = torch.cuda.is_available()

    best_valid_loss = np.inf
    global_iteration = 0
    counters = {'train': 0, 'valid': 0}
    epoch_counter_external = 0
    stats = dict()

    for n_rollout in n_rollout_list:
        stats[n_rollout] = dict()
        config_tmp = copy.copy(config)
        config_tmp['train']['n_rollout'] = n_rollout
        for phase in phase_list:
            stats[n_rollout][phase] = dict()
            print("Loading data for %s" % phase)
            dataset = MultiEpisodeDataset(config_tmp,
                                                  action_function=action_function,
                                                  observation_function=observation_function,
                                                  episodes=multi_episode_dict,
                                                  phase=phase)

            dataloader = DataLoader(
                dataset, batch_size=config['train']['batch_size'],
                shuffle=True,
                num_workers=config['train']['num_workers'], drop_last=True)


            loss_tensor_container = {"l2_avg": [],
                                     "l2_final_step": []}

            step_duration_meter = AverageMeter()
            global_iteration = 0

            for epoch in range(num_epochs):
                for i, data in enumerate(dataloader):

                    loss_container = dict() # store the losses for this step
                    # types of losses ["l2_avg", "l2_final_step"]

                    step_start_time = time.time()
                    global_iteration += 1
                    counters[phase] += 1

                    with torch.no_grad():
                        n_his = config['train']['n_history']
                        n_roll = n_rollout
                        n_samples = n_his + n_roll

                        if DEBUG:
                            print("global iteration: %d" %(global_iteration))
                            print("n_samples", n_samples)


                        # [B, n_samples, obs_dim]
                        observations = data['observations']

                        # [B, n_samples, action_dim]
                        actions = data['actions']
                        B = actions.shape[0]

                        if use_gpu:
                            observations = observations.cuda()
                            actions = actions.cuda()

                        # states, actions = data
                        assert actions.shape[1] == n_samples
                        loss_mse = 0.


                        # we don't have any visual observations, so states are observations
                        states = observations

                        # state_cur: B x n_his x state_dim
                        # state_cur = states[:, :n_his]

                        # [B, n_his, state_dim]
                        state_init = states[:, :n_his]

                        # We want to rollout n_roll steps
                        # actions = [B, n_his + n_roll, -1]
                        # so we want action_seq.shape = [B, n_roll, -1]
                        action_start_idx = 0
                        action_end_idx = n_his + n_roll - 1
                        action_seq = actions[:, action_start_idx:action_end_idx, :]

                        if DEBUG:
                            print("states.shape", states.shape)
                            print("state_init.shape", state_init.shape)
                            print("actions.shape", actions.shape)
                            print("action_seq.shape", action_seq.shape)

                        # try using models_dy.rollout_model instead of doing this manually
                        rollout_data = rollout_model(state_init=state_init,
                                                     action_seq=action_seq,
                                                     dynamics_net=model_dy,
                                                     compute_debug_data=False)

                        # [B, n_roll, state_dim]
                        state_rollout_pred = rollout_data['state_pred']

                        # [B, n_roll, state_dim]
                        state_rollout_gt = states[:, n_his:]


                        if DEBUG:
                            print("state_rollout_gt.shape", state_rollout_gt.shape)
                            print("state_rollout_pred.shape", state_rollout_pred.shape)

                        # the loss function is between
                        # [B, n_roll, state_dim]
                        state_pred_err = state_rollout_pred - state_rollout_gt

                        # [B]
                        l2_avg_tensor = torch.mean(torch.norm(state_pred_err, dim=-1), dim=1).detach().cpu()
                        l2_avg = l2_avg_tensor.mean()

                        # [B]
                        l2_final_step_tensor = torch.norm(state_pred_err[:, -1], dim=-1).detach().cpu()
                        l2_final_step = l2_final_step_tensor.mean()

                        loss_tensor_container["l2_avg"].append(l2_avg_tensor)
                        loss_container["l2_avg"] = l2_avg

                        loss_tensor_container["l2_final_step"].append(l2_final_step_tensor)
                        loss_container["l2_final_step"] = l2_final_step


                    step_duration_meter.update(time.time() - step_start_time)

                    if (i % config['train']['log_per_iter'] == 0) or (global_iteration % config['train']['log_per_iter'] == 0):
                        # print some logging information
                        log = ""
                        log += ', step time %.6f' % (step_duration_meter.avg)


                        # log data to tensorboard
                        for loss_type, loss_obj in loss_container.items():
                            plot_name = "%s/n_roll_%s/%s" %(loss_type, n_roll, phase)
                            writer.add_scalar(plot_name, loss_obj.item(), global_iteration)

                            log += " %s: %.6f," %(plot_name, loss_obj.item())

                        print(log)


                    writer.flush() # flush SummaryWriter events to disk

            stats[n_rollout][phase] = dict()
            for loss_type in loss_tensor_container:
                t = torch.cat(loss_tensor_container[loss_type])
                mean = t.mean()
                median = t.median()
                std = t.std()

                stats[n_rollout][phase][loss_type] = {'mean': mean,
                                                      'median': median,
                                                      'std': std}


                for stat_type, val in stats[n_rollout][phase][loss_type].items():
                    plot_name = "stats/%s/n_roll_%d/%s/%s" %(loss_type, n_roll, phase, stat_type)

                    for idx_tmp in [0,10,100]:
                        writer.add_scalar(plot_name, val, idx_tmp)




