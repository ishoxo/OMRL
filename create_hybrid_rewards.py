
import os
import argparse
from torchkit import pytorch_utils as ptu
import ipdb
import torch
from torchkit.pytorch_utils import set_gpu_mode
from models.vae import VAE
from offline_metalearner import OfflineMetaLearner
import utils.config_utils as config_utl
from utils import offline_utils as off_utl
from offline_config import args_ant_semicircle_sparse, \
    args_cheetah_vel, args_point_robot_sparse, args_gridworld

"""
scale intrinsic rewards
add them to normal reward with weight
save new hybrid rewards with IR type and weight
"""


def transform_mdp_to_bamdp_rollouts(vae, args, obs, actions, rewards, next_obs, terminals):
    '''

    :param vae:
    :param args:
    :param obs: shape (trajectory_len, n_rollouts, dim)
    :param actions:
    :param rewards:
    :param next_obs:
    :param terminals:
    :return:
    '''

    # augmented_obs = ptu.zeros((obs.shape[0], obs.shape[1] + 2 * args.task_embedding_size))
    augmented_obs = ptu.zeros((obs.shape[0], obs.shape[1], obs.shape[2] + 2 * args.task_embedding_size))
    # augmented_next_obs = ptu.zeros((obs.shape[0], obs.shape[1] + 2 * args.task_embedding_size))
    augmented_next_obs = ptu.zeros((obs.shape[0], obs.shape[1], obs.shape[2] + 2 * args.task_embedding_size))
    if args.belief_rewards:
        belief_rewards = ptu.zeros_like(rewards)
    else:
        belief_rewards = None

    KL_exploration_rewards = ptu.zeros_like(rewards)
    EUC_exploration_rewards = ptu.zeros_like(rewards)
    VAR_RED_exploration_rewards = ptu.zeros_like(rewards)
    gauss_dim = 5
    with torch.no_grad():
        # _, mean, logvar, hidden_state = vae.encoder.prior(batch_size=1)
        _, prior_mean, prior_logvar, hidden_state = vae.encoder.prior(batch_size=obs.shape[1])
        augmented_obs[0, :, :] = torch.cat((obs[0], prior_mean[0], prior_logvar[0]), dim=-1)
    for step in range(args.trajectory_len):
        # update encoding
        _, mean, logvar, hidden_state = utl.update_encoding(
            encoder=vae.encoder,
            obs=next_obs[step].unsqueeze(0),
            action=actions[step].unsqueeze(0),
            reward=rewards[step].unsqueeze(0),
            done=terminals[step].unsqueeze(0),
            hidden_state=hidden_state
        )


        KL_exploration_reward = 0.5 * (torch.sum(prior_logvar, dim=-1) - torch.sum(logvar, dim=-1) - gauss_dim +
                        torch.sum(1 / torch.exp(prior_logvar) * torch.exp(logvar), dim=-1) + ((prior_mean - mean) /
                                                                                              torch.exp(prior_logvar) * (prior_mean - mean)).sum(
                        dim=-1)).view(-1, 1)

        EUC_exploration_reward = torch.norm(mean - prior_mean, dim=-1).view(-1, 1)

        VAR_RED_exploration_reward = torch.sum(torch.exp(prior_logvar) - torch.exp(logvar), dim=-1).view(-1, 1)

        prior_mean = mean
        prior_logvar = logvar
        # ipdb.set_trace()
        # augment data
        augmented_next_obs[step, :, :] = torch.cat((next_obs[step], mean, logvar), dim=-1)
        KL_exploration_rewards[step, :, :] = KL_exploration_reward
        EUC_exploration_rewards[step, :, :] = EUC_exploration_reward
        VAR_RED_exploration_rewards[step, :, :] = VAR_RED_exploration_reward
        if args.belief_rewards:
            with torch.no_grad():
                belief_rewards[step, :, :] = vae.compute_belief_reward(mean.unsqueeze(dim=0),
                                                                       logvar.unsqueeze(dim=0),
                                                                       obs[step].unsqueeze(dim=0),
                                                                       next_obs[step].unsqueeze(dim=0),
                                                                       actions[step].unsqueeze(dim=0))

    augmented_obs[1:, :, :] = augmented_next_obs[:-1, :, :].clone()
    exploration_rewards = [KL_exploration_rewards, EUC_exploration_rewards,
                           VAR_RED_exploration_rewards]

    return augmented_obs, belief_rewards, augmented_next_obs, exploration_rewards

def transform_mdps_ds_to_bamdp_ds_IR(dataset, vae, args, IR_type, IR_weight):
    '''

    :param dataset: list of lists of lists. each list is list of arrays
    (s,a,r,s',done) arrays of size (traj_len, n_trajs, dim)
    :param vae: trained vae model
    :return:
    '''

    bamdp_dataset = []

    for i, set in enumerate(dataset):
        obs, actions, rewards, next_obs, terminals = set
        augmented_obs, belief_rewards, augmented_next_obs, exploration_rewards = \
            transform_mdp_to_bamdp_rollouts(vae, args,
                                            ptu.FloatTensor(obs),
                                            ptu.FloatTensor(actions),
                                            ptu.FloatTensor(rewards),
                                            ptu.FloatTensor(next_obs),
                                            ptu.FloatTensor(terminals))
        rewards = belief_rewards if belief_rewards is not None else ptu.FloatTensor(rewards)


        exploration_rewards = [ptu.get_numpy(item) for item in exploration_rewards]

        bamdp_dataset.append([ptu.get_numpy(augmented_obs), actions, ptu.get_numpy(rewards),
                              ptu.get_numpy(augmented_next_obs), terminals, exploration_rewards])
        print('{} datasets were processed.'.format(i + 1))
    return bamdp_dataset