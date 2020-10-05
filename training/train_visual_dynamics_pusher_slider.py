import os
import random
import numpy as np
import time

# from progressbar import ProgressBar

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from key_dynam.dataset.episode_dataset import MultiEpisodeDataset
from key_dynam.dynamics.utils import rand_int, count_trainable_parameters, Tee, AverageMeter, get_lr, to_np, set_seed
from key_dynam.dataset.function_factory import ObservationFunctionFactory, ActionFunctionFactory
from key_dynam.dataset.utils import load_drake_sim_episodes_from_config
from key_dynam.utils.utils import save_yaml, load_pickle, get_data_root
from key_dynam.models.model_builder import build_visual_dynamics_model
from key_dynam.dense_correspondence.descriptor_net import sample_descriptors

DEBUG = False

def save_model(model, save_base_path):
    # save both the model in binary form, and also the state dict
    torch.save(model.state_dict(), save_base_path + "_state_dict.pth")
    torch.save(model, save_base_path + "_model.pth")

def train_dynamics(config,
                   train_dir, # str: directory to save output
                   ):

    use_precomputed_keypoints = config['dataset']['visual_observation']['enabled'] and config['dataset']['visual_observation']['descriptor_keypoints']

    # set random seed for reproduction
    set_seed(config['train']['random_seed'])

    st_epoch = config['train']['resume_epoch'] if config['train']['resume_epoch'] > 0 else 0
    tee = Tee(os.path.join(train_dir, 'train_st_epoch_%d.log' % st_epoch), 'w')

    tensorboard_dir = os.path.join(train_dir, "tensorboard")
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    writer = SummaryWriter(log_dir=tensorboard_dir)

    # save the config
    save_yaml(config, os.path.join(train_dir, "config.yaml"))

    # load the data
    episodes = load_drake_sim_episodes_from_config(config)

    action_function = ActionFunctionFactory.function_from_config(config)
    observation_function = ObservationFunctionFactory.function_from_config(config)

    datasets = {}
    dataloaders = {}
    data_n_batches = {}
    for phase in ['train', 'valid']:
        print("Loading data for %s" % phase)
        datasets[phase] = MultiEpisodeDataset(config,
                                              action_function=action_function,
                                              observation_function=observation_function,
                                              episodes=episodes,
                                              phase=phase)

        dataloaders[phase] = DataLoader(
            datasets[phase], batch_size=config['train']['batch_size'],
            shuffle=True if phase == 'train' else False,
            num_workers=config['train']['num_workers'], drop_last=True)

        data_n_batches[phase] = len(dataloaders[phase])

    use_gpu = torch.cuda.is_available()

    # compute normalization parameters if not starting from pre-trained network . . .


    '''
    define model for dynamics prediction
    '''

    model_dy = build_visual_dynamics_model(config)
    K = config['vision_net']['num_ref_descriptors']

    print("model_dy.vision_net._reference_descriptors.shape", model_dy.vision_net._ref_descriptors.shape)
    print("model_dy.vision_net.descriptor_dim", model_dy.vision_net.descriptor_dim)
    print("model_dy #params: %d" % count_trainable_parameters(model_dy))

    camera_name = config['vision_net']['camera_name']
    W = config['env']['rgbd_sensors']['sensor_list'][camera_name]['width']
    H = config['env']['rgbd_sensors']['sensor_list'][camera_name]['height']
    diag = np.sqrt(W**2 + H**2) # use this to scale the loss

    # sample reference descriptors unless using precomputed keypoints
    if not use_precomputed_keypoints:
        # sample reference descriptors
        episode_names = list(datasets["train"].episode_dict.keys())
        episode_names.sort()
        episode_name = episode_names[0]
        episode = datasets["train"].episode_dict[episode_name]
        episode_idx = 0
        camera_name = config["vision_net"]["camera_name"]
        image_data = episode.get_image_data(camera_name, episode_idx)
        des_img = torch.Tensor(image_data['descriptor'])
        mask_img = torch.Tensor(image_data['mask'])
        ref_descriptor_dict = sample_descriptors(des_img,
                                                 mask_img,
                                                 config['vision_net']['num_ref_descriptors'])



        model_dy.vision_net._ref_descriptors.data = ref_descriptor_dict['descriptors']
        model_dy.vision_net.reference_image = image_data['rgb']
        model_dy.vision_net.reference_indices = ref_descriptor_dict['indices']
    else:
        metadata_file = os.path.join(get_data_root(), config['dataset']['descriptor_keypoints_dir'], 'metadata.p')
        descriptor_metadata = load_pickle(metadata_file)

        # [32, 2]
        ref_descriptors = torch.Tensor(descriptor_metadata['ref_descriptors'])

        # [K, 2]
        ref_descriptors = ref_descriptors[:K]
        model_dy.vision_net._ref_descriptors.data = ref_descriptors
        model_dy.vision_net._ref_descriptors_metadata = descriptor_metadata

        # this is just a sanity check
        assert model_dy.vision_net.num_ref_descriptors == K

    print("reference_descriptors", model_dy.vision_net._ref_descriptors)

    # criterion
    criterionMSE = nn.MSELoss()
    l1Loss = nn.L1Loss()

    # optimizer
    params = model_dy.parameters()
    lr = float(config['train']['lr'])
    optimizer = optim.Adam(params, lr=lr, betas=(config['train']['adam_beta1'], 0.999))

    # setup scheduler
    sc = config['train']['lr_scheduler']
    scheduler = ReduceLROnPlateau(optimizer,
                                  mode='min',
                                  factor=sc['factor'],
                                  patience=sc['patience'],
                                  threshold_mode=sc['threshold_mode'],
                                  cooldown= sc['cooldown'],
                                  verbose=True)

    if use_gpu:
        print("using gpu")
        model_dy = model_dy.cuda()

    print("model_dy.vision_net._ref_descriptors.device", model_dy.vision_net._ref_descriptors.device)
    print("model_dy.vision_net #params: %d" %(count_trainable_parameters(model_dy.vision_net)))


    best_valid_loss = np.inf
    global_iteration = 0
    epoch_counter_external = 0

    try:
        for epoch in range(st_epoch, config['train']['n_epoch']):
            phases = ['train', 'valid']
            epoch_counter_external = epoch

            writer.add_scalar("Training Params/epoch", epoch, global_iteration)
            for phase in phases:
                model_dy.train(phase == 'train')

                meter_loss_rmse = AverageMeter()
                step_duration_meter = AverageMeter()


                # bar = ProgressBar(max_value=data_n_batches[phase])
                loader = dataloaders[phase]

                for i, data in enumerate(loader):

                    step_start_time = time.time()

                    global_iteration += 1

                    with torch.set_grad_enabled(phase == 'train'):
                        n_his, n_roll = config['train']['n_history'], config['train']['n_rollout']
                        n_samples = n_his + n_roll

                        if DEBUG:
                            print("global iteration: %d" %(global_iteration))


                        # visual_observations = data['visual_observations']
                        visual_observations_list = data['visual_observations_list']
                        observations = data['observations']
                        actions = data['actions']

                        if use_gpu:
                            observations = observations.cuda()
                            actions = actions.cuda()

                        # states, actions = data
                        assert actions.size(1) == n_samples

                        B = actions.size(0)
                        loss_mse = 0.


                        # compute the output of the visual model for all timesteps
                        visual_model_output_list = []
                        for visual_obs in visual_observations_list:
                            # visual_obs is a dict containing observation for a single
                            # time step (of course across a batch however)
                            # visual_obs[<camera_name>]['rgb_tensor'] has shape [B, 3, H, W]

                            # probably need to cast input to cuda
                            dynamics_net_input = None
                            if use_precomputed_keypoints:
                                # note precomputed descriptors stored on disk are of size
                                # K = 32. We need to trim it down to the appropriate size
                                # [B, K_disk, 2] where K_disk is num keypoints on disk
                                keypoints = visual_obs[camera_name]['descriptor_keypoints']


                                # [B, 32, 2] where K is num keypoints
                                keypoints = keypoints[:,:K]

                                if DEBUG:
                                    print("keypoints.shape", keypoints.shape)

                                dynamics_net_input = keypoints.flatten(start_dim=1)
                            else:
                                out_dict = model_dy.vision_net.forward(visual_obs)

                                # [B, vision_model_out_dim]
                                dynamics_net_input = out_dict['dynamics_net_input']

                            visual_model_output_list.append(dynamics_net_input)

                        # concatenate this into a tensor
                        # [B, n_samples, vision_model_out_dim]
                        visual_model_output = torch.stack(visual_model_output_list, dim=1)

                        # cast this to float so it can be concatenated below
                        visual_model_output = visual_model_output.type_as(observations)

                        if DEBUG:
                            print('visual_model_output.shape', visual_model_output.shape)
                            print("observations.shape", observations.shape)
                            print("actions.shape", actions.shape)

                        # states is gotten by concatenating visual_observations and observations
                        # [B, n_samples, vision_model_out_dim + obs_dim]
                        states = torch.cat((visual_model_output, observations), dim=-1)

                        # state_cur: B x n_his x state_dim
                        state_cur = states[:, :n_his]

                        if DEBUG:
                            print("states.shape", states.shape)

                        for j in range(n_roll):

                            if DEBUG:
                                print("n_roll j: %d" %(j))

                            state_des = states[:, n_his + j]

                            # action_cur: B x n_his x action_dim
                            action_cur = actions[:, j : j + n_his] if actions is not None else None

                            # state_pred: B x state_dim
                            # state_pred: B x state_dim
                            input = {'observation': state_cur,
                                     'action': action_cur,
                                     }

                            if DEBUG:
                                print("state_cur.shape", state_cur.shape)
                                print("action_cur.shape", action_cur.shape)

                            state_pred = model_dy.dynamics_net(input)

                            # normalize by diag to ensure the loss is in [0,1] range
                            loss_mse_cur = criterionMSE(state_pred/diag, state_des/diag)
                            loss_mse += loss_mse_cur / n_roll

                            # l1Loss
                            loss_l1 = l1Loss(state_pred, state_des)

                            # update state_cur
                            # state_pred.unsqueeze(1): B x 1 x state_dim
                            # state_cur: B x n_his x state_dim
                            state_cur = torch.cat([state_cur[:, 1:], state_pred.unsqueeze(1)], 1)

                            meter_loss_rmse.update(np.sqrt(loss_mse.item()), B)

                    step_duration_meter.update(time.time() - step_start_time)
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss_mse.backward()
                        optimizer.step()

                    if (i % config['train']['log_per_iter'] == 0) or (global_iteration % config['train']['log_per_iter'] == 0):
                        log = '%s [%d/%d][%d/%d] LR: %.6f' % (
                            phase, epoch, config['train']['n_epoch'], i, data_n_batches[phase],
                            get_lr(optimizer))
                        log += ', rmse: %.6f (%.6f)' % (
                            np.sqrt(loss_mse.item()), meter_loss_rmse.avg)

                        log += ', step time %.6f' %(step_duration_meter.avg)
                        step_duration_meter.reset()


                        print(log)

                        # log data to tensorboard
                        # only do it once we have reached 100 iterations
                        if global_iteration > 100:
                            writer.add_scalar("Params/learning rate", get_lr(optimizer), global_iteration)
                            writer.add_scalar("Loss_MSE/%s" %(phase), loss_mse.item(), global_iteration)
                            writer.add_scalar("L1/%s" %(phase), loss_l1.item(), global_iteration)
                            writer.add_scalar("L1_fraction/%s" %(phase), loss_l1.item()/diag, global_iteration)
                            writer.add_scalar("RMSE average loss/%s" %(phase), meter_loss_rmse.avg, global_iteration)

                    if phase == 'train' and i % config['train']['ckp_per_iter'] == 0:
                        save_model(model_dy, '%s/net_dy_epoch_%d_iter_%d' % (train_dir, epoch, i))



                log = '%s [%d/%d] Loss: %.6f, Best valid: %.6f' % (
                    phase, epoch, config['train']['n_epoch'], meter_loss_rmse.avg, best_valid_loss)
                print(log)

                if phase == 'valid':
                    if config['train']['lr_scheduler']['enabled']:
                        scheduler.step(meter_loss_rmse.avg)

                    # print("\nPhase == valid")
                    # print("meter_loss_rmse.avg", meter_loss_rmse.avg)
                    # print("best_valid_loss", best_valid_loss)
                    if meter_loss_rmse.avg < best_valid_loss:
                        best_valid_loss = meter_loss_rmse.avg
                        save_model(model_dy, '%s/net_best_dy' % (train_dir))

                writer.flush() # flush SummaryWriter events to disk

    except KeyboardInterrupt:
        # save network if we have a keyboard interrupt
        save_model(model_dy, '%s/net_dy_epoch_%d_keyboard_interrupt' % (train_dir, epoch_counter_external))
        writer.flush() # flush SummaryWriter events to disk
