import ipdb
import torch
import os
import argparse
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from torch.nn import functional as F
import math
import utils.config_utils as config_utl
from torchkit import pytorch_utils as ptu
from utils import helpers as utl, offline_utils as off_utl
from torchkit.pytorch_utils import set_gpu_mode
from offline_config import args_ant_semicircle_sparse, \
    args_cheetah_vel, args_point_robot_sparse, args_gridworld

parser = argparse.ArgumentParser()

parser.add_argument('--env-type', default='gridworld')

args, rest_args = parser.parse_known_args()
env = args.env_type
if env == 'gridworld':
    args = args_gridworld.get_args(rest_args)
# --- PointRobot ---
elif env == 'point_robot_sparse':
    args = args_point_robot_sparse.get_args(rest_args)



# args.vae_model_name = 'relabel__08_01_17_38_11' ##prior
args.vae_model_name = 'relabel__06_01_23_15_26' ## no learnt prior
set_gpu_mode(torch.cuda.is_available() and args.use_gpu)
vae_args = config_utl.load_config_file(os.path.join(args.vae_dir, args.env_name,
                                                        args.vae_model_name, 'online_config.json'))
from offline_metalearner import OfflineMetaLearner

args = config_utl.merge_configs(vae_args, args)
learner = OfflineMetaLearner(args)

def visualise_behaviour(env,
                        args,
                        policy,
                        iter_idx,
                        encoder=None,
                        reward_decoder=None,
                        image_folder=None,
                        **kwargs
                        ):
    """
    Visualises the behaviour of the policy, together with the latent state and belief.
    The environment passed to this method should be a SubProcVec or DummyVecEnv, not the raw env!
    """

    num_episodes = args.max_rollouts_per_task
    # unwrapped_env = env.venv.unwrapped.envs[0]

    # --- initialise things we want to keep track of ---

    episode_all_obs = [[] for _ in range(num_episodes)]
    episode_prev_obs = [[] for _ in range(num_episodes)]
    episode_next_obs = [[] for _ in range(num_episodes)]
    episode_actions = [[] for _ in range(num_episodes)]
    episode_rewards = [[] for _ in range(num_episodes)]

    episode_returns = []
    episode_lengths = []

    episode_goals = []
    # if args.pass_belief_to_policy and (encoder is None):
    #     episode_beliefs = [[] for _ in range(num_episodes)]
    # else:
    #     episode_beliefs = None

    if encoder is not None:
        # keep track of latent spaces
        episode_latent_samples = [[] for _ in range(num_episodes)]
        episode_latent_means = [[] for _ in range(num_episodes)]
        episode_latent_logvars = [[] for _ in range(num_episodes)]
    else:
        episode_latent_samples = episode_latent_means = episode_latent_logvars = None

    curr_latent_sample = curr_latent_mean = curr_latent_logvar = None

    # --- roll out policy ---

    env.reset_task()
    # state = ptu.from_numpy(env.reset())
    # [state, belief, task] = utl.reset_env(env, args)

    state = ptu.from_numpy(env.reset()).unsqueeze(dim=0)
    start_obs = state.clone()

    for episode_idx in range(args.max_rollouts_per_task):

        curr_goal = env.get_task()
        curr_rollout_rew = []
        curr_rollout_goal = []

        if encoder is not None:

            if episode_idx == 0:
                # reset to prior
                curr_latent_sample, curr_latent_mean, curr_latent_logvar, hidden_state = encoder.prior(1)
                curr_latent_sample = curr_latent_sample[0]
                curr_latent_mean = curr_latent_mean[0]
                curr_latent_logvar = curr_latent_logvar[0]

            episode_latent_samples[episode_idx].append(curr_latent_sample[0].clone())
            episode_latent_means[episode_idx].append(curr_latent_mean[0].clone())
            episode_latent_logvars[episode_idx].append(curr_latent_logvar[0].clone())

        episode_all_obs[episode_idx].append(start_obs.clone())
        # if args.pass_belief_to_policy and (encoder is None):
        #     episode_beliefs[episode_idx].append(belief)
        # ipdb.set_trace()
        for step_idx in range(1, env.unwrapped._max_episode_steps + 1):

            if step_idx == 1:
                episode_prev_obs[episode_idx].append(start_obs.clone())
            else:
                episode_prev_obs[episode_idx].append(state.clone())

            # act
            # ipdb.set_trace()
            augmented_obs = get_augmented_obs(state, curr_latent_mean, curr_latent_logvar)
            if args.policy == 'dqn':
                action, value = policy.act(obs=augmented_obs, deterministic=False)
                # print(action)
            else:
                action, _, _, log_prob = policy.act(obs=augmented_obs,
                                                        deterministic=args.eval_deterministic,
                                                        return_log_prob=True)

            # observe reward and next obs


            state, rew_raw, done, info = utl.env_step(env, action.squeeze(dim=0))
            # ipdb.set_trace()

            if encoder is not None:
                # update task embedding
                # ipdb.set_trace()
                curr_latent_sample, curr_latent_mean, curr_latent_logvar, hidden_state = encoder(
                    action.float(),
                    state,
                    rew_raw.reshape((1, 1)).float(),
                    hidden_state,
                    return_prior=False)

                episode_latent_samples[episode_idx].append(curr_latent_sample[0].clone())
                episode_latent_means[episode_idx].append(curr_latent_mean[0].clone())
                episode_latent_logvars[episode_idx].append(curr_latent_logvar[0].clone())

            episode_all_obs[episode_idx].append(state.clone())
            episode_next_obs[episode_idx].append(state.clone())
            episode_rewards[episode_idx].append(rew_raw.clone())
            episode_actions[episode_idx].append(action.clone())

            curr_rollout_rew.append(rew_raw.clone())
            curr_rollout_goal.append(env.get_task().copy())

            # if args.pass_belief_to_policy and (encoder is None):
            #     episode_beliefs[episode_idx].append(belief)
            # ipdb.set_trace()
            if info['done_mdp'] and not done:
                # start_obs = info[0]['start_state']
                # start_obs = torch.from_numpy(start_obs).float().reshape((1, -1))
                break

        episode_returns.append(sum(curr_rollout_rew))
        episode_lengths.append(step_idx)
        episode_goals.append(curr_goal)

    # clean up

    if encoder is not None:
        episode_latent_means = [torch.stack(e) for e in episode_latent_means]
        episode_latent_logvars = [torch.stack(e) for e in episode_latent_logvars]

    episode_prev_obs = [torch.cat(e) for e in episode_prev_obs]
    episode_next_obs = [torch.cat(e) for e in episode_next_obs]
    episode_actions = [torch.cat(e) for e in episode_actions]
    episode_rewards = [torch.cat(e) for e in episode_rewards]

    # plot behaviour & visualise belief in env

    rew_pred_means, rew_pred_vars = plot_bb(env, args, episode_all_obs, episode_goals, reward_decoder,
                                            episode_latent_means, episode_latent_logvars,
                                            image_folder, iter_idx, None)

    # if reward_decoder:
    #     plot_rew_reconstruction(env, rew_pred_means, rew_pred_vars, image_folder, iter_idx)

    return episode_latent_means, episode_latent_logvars, \
           episode_prev_obs, episode_next_obs, episode_actions, episode_rewards, \
           episode_returns


def get_augmented_obs(obs, mean, logvar):
    mean = mean.reshape((-1, mean.shape[-1]))
    logvar = logvar.reshape((-1, logvar.shape[-1]))
    return torch.cat((obs, mean, logvar), dim=-1)


def plot_rew_reconstruction(env,
                            rew_pred_means,
                            rew_pred_vars,
                            image_folder,
                            iter_idx,
                            ):
    """
    Note that env might need to be a wrapped env!
    """

    num_rollouts = len(rew_pred_means)

    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    test_rew_mus = torch.cat(rew_pred_means).cpu().detach().numpy()
    plt.plot(range(test_rew_mus.shape[0]), test_rew_mus, '.-', alpha=0.5)
    plt.plot(range(test_rew_mus.shape[0]), test_rew_mus.mean(axis=1), 'k.-')
    for tj in np.cumsum([0, *[env.unwrapped._max_episode_steps for _ in range(num_rollouts)]]):
        span = test_rew_mus.max() - test_rew_mus.min()
        plt.plot([tj + 0.5, tj + 0.5], [test_rew_mus.min() - 0.05 * span, test_rew_mus.max() + 0.05 * span], 'k--',
                 alpha=0.5)
    plt.title('output - mean')

    plt.subplot(1, 3, 2)
    test_rew_vars = torch.cat(rew_pred_vars).cpu().detach().numpy()
    plt.plot(range(test_rew_vars.shape[0]), test_rew_vars, '.-', alpha=0.5)
    plt.plot(range(test_rew_vars.shape[0]), test_rew_vars.mean(axis=1), 'k.-')
    for tj in np.cumsum([0, *[env.unwrapped._max_episode_steps for _ in range(num_rollouts)]]):
        span = test_rew_vars.max() - test_rew_vars.min()
        plt.plot([tj + 0.5, tj + 0.5], [test_rew_vars.min() - 0.05 * span, test_rew_vars.max() + 0.05 * span],
                 'k--', alpha=0.5)
    plt.title('output - variance')

    plt.subplot(1, 3, 3)
    rew_pred_entropy = -(test_rew_vars * np.log(test_rew_vars)).sum(axis=1)
    plt.plot(range(len(test_rew_vars)), rew_pred_entropy, 'r.-')
    for tj in np.cumsum([0, *[env.unwrapped._max_episode_steps for _ in range(num_rollouts)]]):
        span = rew_pred_entropy.max() - rew_pred_entropy.min()
        plt.plot([tj + 0.5, tj + 0.5], [rew_pred_entropy.min() - 0.05 * span, rew_pred_entropy.max() + 0.05 * span],
                 'k--', alpha=0.5)
    plt.title('Reward prediction entropy')

    plt.tight_layout()
    if image_folder is not None:
        plt.savefig('{}/{}_rew_decoder'.format(image_folder, iter_idx))
        plt.close()
    else:
        plt.savefig("mygraph.png")
        # plt.show()


def plot_bb(env, args, episode_all_obs, episode_goals, reward_decoder,
            episode_latent_means, episode_latent_logvars, image_folder, iter_idx, episode_beliefs):
    """
    Plot behaviour and belief.
    """

    plt.figure(figsize=(1.5 * env.unwrapped._max_episode_steps, 1.5 * args.max_rollouts_per_task))

    num_episodes = len(episode_all_obs)
    num_steps = len(episode_all_obs[0])

    rew_pred_means = [[] for _ in range(num_episodes)]
    rew_pred_vars = [[] for _ in range(num_episodes)]

    # loop through the experiences
    for episode_idx in range(num_episodes):
        for step_idx in range(num_steps):

            curr_obs = episode_all_obs[episode_idx][:step_idx + 1]
            curr_goal = episode_goals[episode_idx]

            if episode_latent_means is not None:
                curr_means = episode_latent_means[episode_idx][:step_idx + 1]
                curr_logvars = episode_latent_logvars[episode_idx][:step_idx + 1]

            # choose correct subplot
            plt.subplot(args.max_rollouts_per_task,
                        math.ceil(env.unwrapped._max_episode_steps) + 1,
                        1 + episode_idx * (1 + math.ceil(env.unwrapped._max_episode_steps)) + step_idx),

            # plot the behaviour
            plot_behaviour(env, curr_obs, curr_goal)

            if reward_decoder is not None:
                # visualise belief in env
                rm, rv = compute_beliefs(env,
                                         args,
                                         reward_decoder,
                                         curr_means[-1],
                                         curr_logvars[-1],
                                         curr_goal)
                rew_pred_means[episode_idx].append(rm)
                rew_pred_vars[episode_idx].append(rv)
                plot_belief(env, rm, args)
            elif episode_beliefs is not None:
                curr_beliefs = episode_beliefs[episode_idx][step_idx]
                plot_belief(env, curr_beliefs, args)
            else:
                rew_pred_means = rew_pred_vars = None

            if episode_idx == 0:
                plt.title('t = {}'.format(step_idx))

            if step_idx == 0:
                plt.ylabel('Episode {}'.format(episode_idx + 1))

    if reward_decoder is not None:
        rew_pred_means = [torch.stack(r) for r in rew_pred_means]
        rew_pred_vars = [torch.stack(r) for r in rew_pred_vars]

    # save figure that shows policy behaviour
    plt.tight_layout()
    if image_folder is not None:
        plt.savefig('{}/{}_behaviour'.format(image_folder, iter_idx))
        plt.close()
    else:
        plt.show()
        plt.savefig("mybel.png")

    return rew_pred_means, rew_pred_vars


def plot_behaviour(env, observations, goal):
    num_cells = int(env.observation_space.high[0] + 1)

    # draw grid
    for i in range(num_cells):
        for j in range(num_cells):
            pos_i = i
            pos_j = j
            rec = Rectangle((pos_i, pos_j), 1, 1, facecolor='none', alpha=0.5,
                            edgecolor='k')
            plt.gca().add_patch(rec)

    # shift obs and goal by half a stepsize
    if isinstance(observations, tuple) or isinstance(observations, list):
        observations = torch.cat(observations)
    observations = observations.cpu().numpy() + 0.5
    goal = np.array(goal) + 0.5

    # visualise behaviour, current position, goal
    # ipdb.set_trace()
    # plt.plot(observations[:, 0], observations[:, 1], 'b-')
    # plt.plot(observations[-1, 0], observations[-1, 1], 'b.')
    plt.plot(observations[0], observations[1], 'b-')
    plt.plot(observations[0], observations[1], 'b.')
    plt.plot(goal[0], goal[1], 'kx')

    # make it look nice
    plt.xticks([])
    plt.yticks([])
    plt.xlim([0, num_cells])
    plt.ylim([0, num_cells])


def compute_beliefs(env, args, reward_decoder, latent_mean, latent_logvar, goal):
    num_cells = env.observation_space.high[0] + 1
    # unwrapped_env = env.venv.unwrapped.envs[0]
    unwrapped_env = env.unwrapped


    if not args.disable_stochasticity_in_latent:
        # take several samples fromt he latent distribution
        samples = utl.sample_gaussian(latent_mean.view(-1), latent_logvar.view(-1), 100)
    else:
        samples = torch.cat((latent_mean.view(-1), latent_logvar.view(-1))).unsqueeze(0)

    # compute reward predictions for those
    if reward_decoder.multi_head:
        rew_pred = reward_decoder(samples, None)
        if args.rew_pred_type == 'categorical':
            rew_pred = F.softmax(rew_pred, dim=-1)
        elif args.rew_pred_type == 'bernoulli':
            rew_pred = torch.sigmoid(rew_pred)
        rew_pred_means = torch.mean(rew_pred, dim=0)  # .reshape((1, -1))
        rew_pred_vars = torch.var(rew_pred, dim=0)  # .reshape((1, -1))
    else:
        tsm = []
        tsv = []
        for st in range(num_cells ** 2):
            task_id = unwrapped_env.id_to_task(torch.tensor([st]))
            curr_state = unwrapped_env.goal_to_onehot_id(task_id).expand((samples.shape[0], 2))
            if unwrapped_env.oracle:
                if isinstance(goal, np.ndarray):
                    goal = torch.from_numpy(goal)
                curr_state = torch.cat((curr_state, goal.repeat(curr_state.shape[0], 1).float()), dim=1)
            rew_pred = reward_decoder(samples, curr_state)
            if args.rew_pred_type == 'bernoulli':
                rew_pred = torch.sigmoid(rew_pred)
            tsm.append(torch.mean(rew_pred))
            tsv.append(torch.var(rew_pred))
        rew_pred_means = torch.stack(tsm).reshape((1, -1))
        rew_pred_vars = torch.stack(tsv).reshape((1, -1))
    # rew_pred_means = rew_pred_means[-1][0]

    return rew_pred_means, rew_pred_vars


def plot_belief(env, beliefs, args):
    """
    Plot the belief by taking 100 samples from the latent space and plotting the average predicted reward per cell.
    """

    num_cells = int(env.observation_space.high[0] + 1)
    # unwrapped_env = env.venv.unwrapped.envs[0]
    unwrapped_env = env.unwrapped


    # draw probabilities for each grid cell
    alphas = []
    for i in range(num_cells):
        for j in range(num_cells):
            pos_i = i
            pos_j = j
            idx = unwrapped_env.task_to_id(torch.tensor([[pos_i, pos_j]]))
            alpha = beliefs[idx]
            alphas.append(alpha.item())
    alphas = np.array(alphas)
    # cut off values (this only happens if we don't use sigmoid/softmax)
    alphas[alphas < 0] = 0
    alphas[alphas > 1] = 1
    # alphas = (np.array(alphas)-min(alphas)) / (max(alphas) - min(alphas))
    count = 0
    for i in range(num_cells):
        for j in range(num_cells):
            pos_i = i
            pos_j = j
            rec = Rectangle((pos_i, pos_j), 1, 1, facecolor='r', alpha=alphas[count],
                            edgecolor='k')
            plt.gca().add_patch(rec)
            count += 1


visualise_behaviour(learner.env, learner.args, learner.agent, 0, learner.vae.encoder,
                    learner.vae.reward_decoder, None)