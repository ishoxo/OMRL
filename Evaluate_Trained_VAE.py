import os
import argparse

import ipdb
import numpy as np
from torchkit import pytorch_utils as ptu

from utils import helpers as utl, offline_utils as off_utl
import torch
from torchkit.pytorch_utils import set_gpu_mode
from models.vae import VAE
from offline_metalearner import OfflineMetaLearner
import utils.config_utils as config_utl
from utils import offline_utils as off_utl
from offline_config import args_ant_semicircle_sparse, \
    args_cheetah_vel, args_point_robot_sparse, args_gridworld

""""
run through n environments, record mean change in KL of belief 1)when not on goal square 2)When on goal square,
3) when making first move. 
"""

parser = argparse.ArgumentParser()
parser.add_argument('--env-type', default='gridworld')
# parser.add_argument('--env-type', default='point_robot_sparse')
args, rest_args = parser.parse_known_args()
env = args.env_type

if env == 'gridworld':
    args = args_gridworld.get_args(rest_args)
# --- PointRobot ---
elif env == 'point_robot_sparse':
    args = args_point_robot_sparse.get_args(rest_args)



# args.vae_model_name = 'relabel__08_01_17_38_11' ##prior
# args.vae_model_name = 'relabel__06_01_23_15_26' ## no learnt prior
args.vae_model_name = 'relabel__10_01_13_38_20' ## single head

set_gpu_mode(torch.cuda.is_available() and args.use_gpu)
vae_args = config_utl.load_config_file(os.path.join(args.vae_dir, args.env_name,
                                                        args.vae_model_name, 'online_config.json'))

args = config_utl.merge_configs(vae_args, args)
learner = OfflineMetaLearner(args)


def get_IR(IR_type, mean, prior_mean, logvar, prior_logvar, gauss_dim):
    if IR_type == 'KL':
        exploration_reward = 0.5 * (torch.sum(prior_logvar, dim=-1) - torch.sum(logvar, dim=-1) - gauss_dim +
                                       torch.sum(1 / torch.exp(prior_logvar) * torch.exp(logvar), dim=-1) + (
                                               (prior_mean - mean) /
                                               torch.exp(prior_logvar) * (prior_mean - mean)).sum(
                    dim=-1)).view(-1, 1)
    elif IR_type == 'EUC':

        exploration_reward = torch.norm(mean - prior_mean, dim=-1).view(-1, 1)
    elif IR_type == 'VAR':

        exploration_reward = torch.sum(torch.exp(prior_logvar) - torch.exp(logvar), dim=-1).view(-1, 1)
    else:
        exploration_reward = None
    return exploration_reward




def Eval_reward_shape(num_environments, IR_type):
    goal_IR = []
    non_goal_IR = []
    first_move_IR = []
    num_episodes = learner.args.max_rollouts_per_task
    num_steps_per_episode = learner.env.unwrapped._max_episode_steps
    tasks = learner.env.unwrapped.get_all_task_idx()
    for i in range(num_environments):
        obs = ptu.from_numpy(learner.env.reset(tasks[i]))
        obs = obs.reshape(-1, obs.shape[-1])
        step = 0

        # get prior parameters
        with torch.no_grad():
            task_sample, task_mean, task_logvar, hidden_state = learner.vae.encoder.prior(batch_size=1)

        for episode_idx in range(num_episodes):
            running_reward = 0.
            for step_idx in range(num_steps_per_episode):
                # add distribution parameters to observation - policy is conditioned on posterior
                augmented_obs = learner.get_augmented_obs(obs, task_mean, task_logvar)
                if learner.args.policy == 'dqn':
                    action, value = learner.agent.act(obs=augmented_obs, deterministic=False)
                else:
                    action, _, _, log_prob = learner.agent.act(obs=augmented_obs,
                                                            deterministic=learner.args.eval_deterministic,
                                                            return_log_prob=True)

                print(action)
                print(augmented_obs)

                # observe reward and next obs
                next_obs, reward, done, info = utl.env_step(learner.env, action.squeeze(dim=0))
                running_reward += reward.item()
                # done_rollout = False if ptu.get_numpy(done[0][0]) == 0. else True
                # update encoding
                new_task_sample, new_task_mean, new_task_logvar, new_hidden_state = learner.update_encoding(obs=next_obs,
                                                                                         action=action,
                                                                                         reward=reward,
                                                                                         done=done,
                                                                                         hidden_state=hidden_state)

                IR = get_IR(IR_type=IR_type, mean=new_task_mean, prior_mean=task_mean,
                            logvar=new_task_logvar, prior_logvar=task_logvar, gauss_dim=5)



                if "is_goal_state" in dir(learner.env.unwrapped) and learner.env.unwrapped.is_goal_state():
                    goal_IR.append(IR)
                else:
                    non_goal_IR.append(IR)

                if step==0:
                    first_move_IR.append(IR)
                # set: obs <- next_obs
                task_sample, task_mean, task_logvar, hidden_state = new_task_sample.clone(), new_task_mean.clone(), \
                                                                    new_task_logvar.clone(), new_hidden_state.clone()
                obs = next_obs.clone()
                step += 1

    print('Non goal IR: {} instances mean of {}'.format(len(non_goal_IR), sum(non_goal_IR)/(len(non_goal_IR)+1)))
    print('Goal IR: {} instances mean of {}'.format(len(goal_IR), sum(goal_IR) / (len(goal_IR)+1)))
    print('First Move: {} instances mean of {}'.format(len(first_move_IR), sum(first_move_IR) / (len(first_move_IR)+1)))


def Get_Compression_Improvement(learner, num_envs, IR_type):
    num_episodes = learner.args.max_rollouts_per_task
    num_steps_per_episode = learner.env.unwrapped._max_episode_steps
    num_tasks = learner.args.num_eval_tasks
    obs_size = learner.env.unwrapped.observation_space.shape[0]

    returns_per_episode = np.zeros((num_tasks, num_episodes))
    success_rate = np.zeros(num_tasks)

    actions = np.zeros((num_tasks, learner.args.trajectory_len))
    rewards = np.zeros((num_tasks, learner.args.trajectory_len))
    reward_preds = np.zeros((num_tasks, learner.args.trajectory_len))
    observations = np.zeros((num_tasks, learner.args.trajectory_len + 1, obs_size))
    if learner.args.policy == 'sac':
        log_probs = np.zeros((num_tasks, learner.args.trajectory_len))

    for task in learner.env.unwrapped.get_all_task_idx():
        obs = ptu.from_numpy(learner.env.reset(task))
        obs = obs.reshape(-1, obs.shape[-1])
        step = 0

        # get prior parameters
        with torch.no_grad():
            task_sample, task_mean, task_logvar, hidden_state = learner.vae.encoder.prior(batch_size=1)

        observations[task, step, :] = ptu.get_numpy(obs[0, :obs_size])

        for episode_idx in range(num_episodes):
            running_reward = 0.
            for step_idx in range(num_steps_per_episode):
                # add distribution parameters to observation - policy is conditioned on posterior
                augmented_obs = learner.get_augmented_obs(obs, task_mean, task_logvar)
                if learner.args.policy == 'dqn':
                    action, value = learner.agent.act(obs=augmented_obs, deterministic=True)
                    action = ptu.FloatTensor([[[learner.env.action_space.sample()]]]).long().squeeze(dim=0)
                    # ipdb.set_trace()
                else:
                    action, _, _, log_prob = learner.agent.act(obs=augmented_obs,
                                                            deterministic=learner.args.eval_deterministic,
                                                            return_log_prob=True)

                # observe reward and next obs
                next_obs, reward, done, info = utl.env_step(learner.env, action.squeeze(dim=0))
                running_reward += reward.item()
                # done_rollout = False if ptu.get_numpy(done[0][0]) == 0. else True
                # update encoding
                new_task_sample, new_task_mean, new_task_logvar, new_hidden_state  = learner.update_encoding(obs=next_obs,
                                                                                         action=action,
                                                                                         reward=reward,
                                                                                         done=done,
                                                                                         hidden_state=hidden_state)
                print('Action: ', action)
                print('next obs: ', next_obs)
                actions[task, step] = action
                rewards[task, step] = reward.item()
                reward_preds[task, step] = ptu.get_numpy(
                    learner.vae.reward_decoder(new_task_sample, next_obs, obs, action)[0, 0])

                # ipdb.set_trace()


                observations[task, step + 1, :] = ptu.get_numpy(next_obs[0, :obs_size])

                history = [observations[task], actions[task], rewards[task]]

                compression_improvement = Eval_Improvement(learner, history, new_task_sample, step) - Eval_Improvement(learner, history, task_sample, step)
                print('compression_improvement: ', compression_improvement)
                print('length of history: ', step+1)


                if learner.args.policy != 'dqn':
                    log_probs[task, step] = ptu.get_numpy(log_prob[0])

                if "is_goal_state" in dir(learner.env.unwrapped) and learner.env.unwrapped.is_goal_state():
                    success_rate[task] = 1.
                # set: obs <- next_obs
                task_sample, task_mean, task_logvar, hidden_state = new_task_sample.clone(), new_task_mean.clone(), \
                                                                    new_task_logvar.clone(), new_hidden_state.clone()
                obs = next_obs.clone()
                step += 1

            returns_per_episode[task, episode_idx] = running_reward

    if learner.args.policy == 'dqn':
        return returns_per_episode, success_rate, observations, rewards, reward_preds

    else:
        return returns_per_episode, success_rate, log_probs, observations, rewards, reward_preds


def Eval_Improvement(learner, history, task_sample, step):
    all_obs = history[0]
    all_action = history[1]
    all_rewards = history[2]

    print(all_obs)
    print(all_action)
    print(all_rewards)

    compression_error = 0
    # ipdb.set_trace()i
    for i in range(step):
        pred_rew = learner.vae.reward_decoder(task_sample, all_obs[i+1], all_obs[i], all_action[i])
        reward = all_rewards[i]
        error = (reward-pred_rew)**2

        ipdb.set_trace()

        compression_error += sum(sum(error))
    return compression_error




# Eval_reward_shape(5, 'KL')

Get_Compression_Improvement(learner, 5, None)



