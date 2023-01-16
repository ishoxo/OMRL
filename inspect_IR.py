import os
import numpy as np
import ipdb
import torch
# env_name = "PointRobotSparse-v0"
env_name = "GridNavi-v2"
gauss_dim = 5

# IR_type = 'EUC'
IR_type = 'VAR'
# IR_type = 'KL'


file_path = "batch_data/GridNavi-v2/data_bamdp/goal_0.0_3.0"


def get_prior_and_post_belief(obs, new_obs):
    prior = obs[3:]
    post = new_obs[3:]

    prior_mean = prior[:5]
    prior_logvar = prior[5:]

    mean = post[:5]
    logvar = post[5:]
    return prior_mean, prior_logvar, mean, logvar


# def KL_Divergence(prior_mean, prior_logvar, mean, logvar):

def get_IR(prior_mean, prior_logvar, mean, logvar, IR_type):
    if IR_type == 'KL':

        IR = 0.5 * (torch.sum(prior_logvar, dim=-1) - torch.sum(logvar, dim=-1) - gauss_dim +
                    torch.sum(1 / torch.exp(prior_logvar) * torch.exp(logvar), dim=-1) + ((prior_mean - mean) /
                                                                                          torch.exp(prior_logvar) * (
                                                                                                      prior_mean - mean)).sum(
                    dim=-1))

    elif IR_type == "EUC":
        IR = torch.norm(mean - prior_mean)

    elif IR_type == "VAR":
        prev_var = torch.exp(prior_logvar)
        var = torch.exp(logvar)
        IR = torch.sum(prev_var - var)

    return IR

def Check_IR(IR_type, file_path):

    exploration_rewards = np.load(file_path + '/{}_rewards.npy'.format(IR_type))
    obs = np.load(file_path + '/obs.npy')
    next_obs = np.load(file_path + '/next_obs.npy')
    actions = np.load(file_path + '/actions.npy')
    rewards = np.load(file_path + '/rewards.npy')

    obs = torch.from_numpy(obs)
    next_obs = torch.from_numpy(next_obs)

    initial = obs[0][0]

    for timestep in range(len(obs)):
        for rollout in range(len(obs[0])):
            observation = obs[timestep][rollout]
            next_observation = next_obs[timestep][rollout]
            prior_mean, prior_logvar, mean, logvar = get_prior_and_post_belief(observation, next_observation)

            # exploration_reward = 0.5 * (torch.sum(prior_logvar, dim=-1) - torch.sum(logvar, dim=-1) - gauss_dim +
            #                                torch.sum(1 / torch.exp(prior_logvar) * torch.exp(logvar), dim=-1) + (
            #                                            (prior_mean - mean) /
            #                                            torch.exp(prior_logvar) * (prior_mean - mean)).sum(
            #             dim=-1)).view(-1, 1)

            exploration_reward = get_IR(prior_mean, prior_logvar, mean, logvar, IR_type)


            # print('----------------------------')
            # print(KL_exploration_reward)
            # print(exploration_rewards[timestep][rollout])
            # ipdb.set_trace()
            print(float(exploration_rewards[timestep][rollout]) == float(exploration_reward))
            # print('----------------------------')




Check_IR(IR_type, file_path)
