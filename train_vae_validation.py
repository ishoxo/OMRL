import argparse
import datetime
import os
import gym
import ipdb
from torch.nn import functional as F
import numpy as np
import torch
from utils import helpers as utl, offline_utils as off_utl
from utils import evaluation as utl_eval
import utils.config_utils as config_utl
from torchkit.pytorch_utils import set_gpu_mode
import matplotlib.pyplot as plt
from torchkit import pytorch_utils as ptu
from models.vae import VAE
from tensorboardX import SummaryWriter
import time
from vae_config import args_gridworld, args_point_robot_sparse, args_cheetah_vel, args_ant_semicircle_sparse

BAR_LENGTH = 32  # for nicely printing training progress

validation_ratio = 0.3


def baseline_task_predictor_loss(dec_task, vae):
    # make some predictions and compute individual losses
    #make this mean of all possible goals
    if vae.args.task_pred_type == 'task_description':
     x = torch.tensor([2.19, 2.19])
     task_pred = x.repeat(dec_task.size()[0], dec_task.size()[1], 1)

    if vae.args.task_pred_type == 'task_id':
     x = torch.tensor([0., 0., 0.05, 0.05, 0.05, 0., 0., 0., 0.05, 0.05, 0.05,
       0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
       0.05, 0.05, 0.05])
     task_pred = x.repeat(dec_task.size()[0], dec_task.size()[1], 1)

    if vae.args.task_pred_type == 'task_id':
        env = gym.make(vae.args.env_name)
        dec_task = env.goal_to_onehot_id(dec_task)

        # dec_task = dec_task.expand(task_pred.shape[:-1]).view(-1)
        # loss for the data we fed into encoder
        task_pred_shape = task_pred.shape

        input = task_pred
        target = dec_task
        loss_task = F.cross_entropy(input, target, reduction='none')


        # env = gym.make(vae.args.env_name)
        # dec_task = env.task_to_id(dec_task)
        # dec_task = dec_task.expand(task_pred.shape[:-1]).view(-1)
        # # loss for the data we fed into encoder
        # task_pred_shape = task_pred.shape

        # loss_task = F.cross_entropy(task_pred.view(-1, task_pred.shape[-1]), dec_task, reduction='none').reshape(
        #     task_pred_shape[:-1])
    elif vae.args.task_pred_type == 'task_description':
        loss_task = (task_pred - dec_task).pow(2).mean(dim=1)

    return loss_task


def vis_train_tasks(env, goals):
    env.plot_env()
    for goal in goals:
        circle = plt.Circle((goal[0], goal[1]), radius=env.goal_radius, alpha=0.3)
        plt.gca().add_artist(circle)


def get_loss_on_validation_set(vae, dataset, tasks, args):

    obs, actions, rewards, next_obs, goals = [], [], [], [], []
    # ipdb.set_trace()
    for idx in range(len(dataset)):
        # random_subset = np.random.permutation(dataset[idx][0].shape[1], )
        # random_subset = np.random.choice(dataset[idx][0].shape[1], args.vae_batch_num_rollouts_per_task)
        goals.append(ptu.FloatTensor(tasks[idx][:, :, :]))
        obs.append(ptu.FloatTensor(dataset[idx][0][:, :, :]))
        actions.append(ptu.FloatTensor(dataset[idx][1][:, :, :]))
        rewards.append(ptu.FloatTensor(dataset[idx][2][:, :, :]))
        next_obs.append(ptu.FloatTensor(dataset[idx][3][:, :, :]))
    obs = torch.cat(obs, dim=1)
    actions = torch.cat(actions, dim=1)
    rewards = torch.cat(rewards, dim=1)
    next_obs = torch.cat(next_obs, dim=1)
    goals = torch.cat(goals, dim=1)
    with torch.no_grad():
        rew_recon_loss, state_recon_loss, kl_term, task_recon_loss, baseline_task_recon_loss \
        = update_step(vae, obs, actions, rewards, next_obs, args, goals, grad=False)

    # take average (this is the expectation over p(M))
    loss = args.rew_loss_coeff * rew_recon_loss + \
           args.state_loss_coeff * state_recon_loss + \
           args.kl_weight * kl_term + args.task_loss_coeff * task_recon_loss
    # ipdb.set_trace()
    return loss, rew_recon_loss, state_recon_loss, \
           task_recon_loss, kl_term, baseline_task_recon_loss


def eval_vae(dataset, vae, args):
    num_tasks = len(dataset)
    reward_preds = np.zeros((num_tasks, args.trajectory_len))
    rewards = np.zeros((num_tasks, args.trajectory_len))
    random_tasks = np.random.choice(len(dataset), 10)  # which trajectory to evaluate

    for task_idx, task in enumerate(random_tasks):
        traj_idx_random = np.random.choice(dataset[0][0].shape[1])  # which trajectory to evaluate
        # get prior parameters
        with torch.no_grad():
            task_sample, task_mean, task_logvar, hidden_state = vae.encoder.prior(batch_size=1)
        for step in range(args.trajectory_len):
            # update encoding
            task_sample, task_mean, task_logvar, hidden_state = utl.update_encoding(
                encoder=vae.encoder,
                obs=ptu.FloatTensor(dataset[task][3][step, traj_idx_random]).unsqueeze(0),
                action=ptu.FloatTensor(dataset[task][1][step, traj_idx_random]).unsqueeze(0),
                reward=ptu.FloatTensor(dataset[task][2][step, traj_idx_random]).unsqueeze(0),
                done=ptu.FloatTensor(dataset[task][4][step, traj_idx_random]).unsqueeze(0),
                hidden_state=hidden_state
            )

            rewards[task_idx, step] = dataset[task][2][step, traj_idx_random].item()
            reward_preds[task_idx, step] = ptu.get_numpy(
                vae.reward_decoder(task_sample.unsqueeze(0),
                                   ptu.FloatTensor(dataset[task][3][step, traj_idx_random]).unsqueeze(0).unsqueeze(0),
                                   ptu.FloatTensor(dataset[task][0][step, traj_idx_random]).unsqueeze(0).unsqueeze(0),
                                   ptu.FloatTensor(dataset[task][1][step, traj_idx_random]).unsqueeze(0).unsqueeze(0))[
                    0, 0])

    return rewards, reward_preds


def update_step(vae, obs, actions, rewards, next_obs, args, tasks, grad=True):
    episode_len, num_episodes, _ = obs.shape

    # get time-steps for ELBO computation
    if args.vae_batch_num_elbo_terms is not None:
        elbo_timesteps = np.stack(
            [np.random.choice(range(0, args.trajectory_len + 1), args.vae_batch_num_elbo_terms, replace=False)
             for _ in range(num_episodes)])
    else:
        elbo_timesteps = np.repeat(np.arange(0, args.trajectory_len + 1).reshape(1, -1),
                                   num_episodes, axis=0)

    # pass through encoder (outputs will be: (max_traj_len+1) x number of rollouts x latent_dim -- includes the prior!)
    _, latent_mean, latent_logvar, _ = vae.encoder(actions=actions,
                                                   states=next_obs,
                                                   rewards=rewards,
                                                   hidden_state=None,
                                                   return_prior=True)

    rew_recon_losses, state_recon_losses, task_recon_losses, kl_terms = [], [], [], []
    baseline_task_recon_losses = []

    # for each task we have in our batch
    for episode_idx in range(num_episodes):

        # get the embedding values (size: traj_length+1 * latent_dim; the +1 is for the prior)
        curr_means = latent_mean[:episode_len + 1, episode_idx, :]
        curr_logvars = latent_logvar[:episode_len + 1, episode_idx, :]
        # take one sample for each ELBO term
        curr_samples = vae.encoder._sample_gaussian(curr_means, curr_logvars)

        # select data from current rollout (result is traj_length * obs_dim)
        curr_obs = obs[:, episode_idx, :]
        curr_next_obs = next_obs[:, episode_idx, :]
        curr_actions = actions[:, episode_idx, :]
        curr_rewards = rewards[:, episode_idx, :]
        curr_tasks = tasks[:, episode_idx, :]


        num_latents = curr_samples.shape[0]  # includes the prior
        num_decodes = curr_obs.shape[0]

        # expand the latent to match the (x, y) pairs of the decoder
        dec_embedding = curr_samples.unsqueeze(0).expand((num_decodes, *curr_samples.shape)).transpose(1, 0)

        # expand the (x, y) pair of the encoder
        dec_obs = curr_obs.unsqueeze(0).expand((num_latents, *curr_obs.shape))
        dec_next_obs = curr_next_obs.unsqueeze(0).expand((num_latents, *curr_next_obs.shape))
        dec_actions = curr_actions.unsqueeze(0).expand((num_latents, *curr_actions.shape))
        dec_rewards = curr_rewards.unsqueeze(0).expand((num_latents, *curr_rewards.shape))
        dec_task = curr_tasks.unsqueeze(0).expand((num_latents, *curr_tasks.shape))

        if args.decode_reward:
            # compute reconstruction loss for this trajectory
            # (for each timestep that was encoded, decode everything and sum it up)
            rrl = vae.compute_rew_reconstruction_loss(dec_embedding, dec_obs, dec_next_obs,
                                                      dec_actions, dec_rewards)
            # sum along the trajectory which we decoded (sum in ELBO_t)
            if args.decode_only_past:
                curr_idx = 0
                past_reconstr_sum = []
                for i, idx_timestep in enumerate(elbo_timesteps[episode_idx]):
                    dec_until = idx_timestep
                    if dec_until != 0:
                        past_reconstr_sum.append(rrl[curr_idx:curr_idx + dec_until].sum())
                    curr_idx += dec_until
                rrl = torch.stack(past_reconstr_sum)
            else:
                rrl = rrl.sum(dim=1)
            rew_recon_losses.append(rrl)
        if args.decode_state:
            srl = vae.compute_state_reconstruction_loss(dec_embedding, dec_obs, dec_next_obs, dec_actions)
            srl = srl.sum(dim=1)
            state_recon_losses.append(srl)

        if args.decode_task:
            trl = vae.compute_task_reconstruction_loss(dec_embedding, dec_task)
            trl = trl.sum(dim=1)
            task_recon_losses.append(trl)

            btrl = baseline_task_predictor_loss(dec_task, vae)
            btrl = btrl.sum(dim=1)
            baseline_task_recon_losses.append(btrl)

        if not args.disable_stochasticity_in_latent:
            # compute the KL term for each ELBO term of the current trajectory
            kl = vae.compute_kl_loss(curr_means, curr_logvars, elbo_timesteps[episode_idx])
            kl_terms.append(kl)

    # sum the ELBO_t terms per task
    if args.decode_reward:
        rew_recon_losses = torch.stack(rew_recon_losses)
        rew_recon_losses = rew_recon_losses.sum(dim=1)
    else:
        rew_recon_losses = ptu.zeros(1)  # 0 -- but with option of .mean()

    if args.decode_state:
        state_recon_losses = torch.stack(state_recon_losses)
        state_recon_losses = state_recon_losses.sum(dim=1)
    else:
        state_recon_losses = ptu.zeros(1)

    if args.decode_task:
        task_recon_losses = torch.stack(task_recon_losses)
        task_recon_losses = task_recon_losses.sum(dim=1)

        baseline_task_recon_losses = torch.stack(baseline_task_recon_losses)
        baseline_task_recon_losses = baseline_task_recon_losses.sum(dim=1)


    else:
        task_recon_losses = ptu.zeros(1)
        baseline_task_recon_losses = ptu.zeros(1)


    if not args.disable_stochasticity_in_latent:
        kl_terms = torch.stack(kl_terms)
        kl_terms = kl_terms.sum(dim=1)
    else:
        kl_terms = ptu.zeros(1)

    # make sure we can compute gradients
    if grad:
        if not args.disable_stochasticity_in_latent:
            assert kl_terms.requires_grad
        if args.decode_reward:
            assert rew_recon_losses.requires_grad
        if args.decode_state:
            assert state_recon_losses.requires_grad



    return rew_recon_losses.mean(), state_recon_losses.mean(), kl_terms.mean(), task_recon_losses.mean(), \
           baseline_task_recon_losses.mean()


def train(vae, dataset, tasks, val_dataset, val_goals, args, eval_interval):
    '''

    :param eval_interval:
    :param tasks:
    :param val_goals:
    :param val_dataset:
    :param vae:
    :param dataset: list of lists. each list for different task contains torch tensors of s,a,r,s',t
    :param args:
    :return:
    '''
    if args.log_tensorboard:
        writer = SummaryWriter(args.full_save_path)

    num_tasks = len(dataset)

    start_time = time.time()
    total_updates = 0
    for iter_ in range(args.num_iters):
        if (iter_) % eval_interval == 0:
            print('performing evaluation:')
            val_loss, val_rew_recon_loss, val_state_recon_loss, val_task_recon_loss, val_kl_term, \
            val_baseline_task_recon_loss = get_loss_on_validation_set(vae, val_dataset, val_goals, args)
            print('Validation Loss: ', val_loss)

            # add in validation metrics
            writer.add_scalar('val_loss/vae_loss', val_loss, total_updates)
            writer.add_scalar('val_loss/rew_recon_loss', val_rew_recon_loss, total_updates)
            writer.add_scalar('val_loss/state_recon_loss', val_state_recon_loss, total_updates)
            writer.add_scalar('val_loss/task_recon_loss', val_task_recon_loss, total_updates)
            writer.add_scalar('val_loss/baseline_task_recon_loss', val_baseline_task_recon_loss, total_updates)
            writer.add_scalar('val_loss/kl', val_kl_term, total_updates)

        n_batches = int(np.ceil(dataset[0][0].shape[1] / args.vae_batch_num_rollouts_per_task))

        traj_permutation = np.random.permutation(dataset[0][0].shape[1])

        loss_tr, rew_loss_tr, state_loss_tr, task_loss_tr, kl_loss_tr = 0, 0, 0, 0, 0  # initialize loss for epoch
        baseline_task_loss_tr = 0
        n_updates = 0  # count number of updates
        for i in range(n_batches):

            if i == n_batches - 1:
                traj_indices = traj_permutation[i * args.vae_batch_num_rollouts_per_task:]
            else:
                traj_indices = traj_permutation[i * args.vae_batch_num_rollouts_per_task:
                                                (i + 1) * args.vae_batch_num_rollouts_per_task]

            n_task_batches = int(np.ceil(num_tasks / args.tasks_batch_size))
            task_permutation = np.random.permutation(num_tasks)

            for j in range(n_task_batches):  # run over tasks
                # ipdb.set_trace()
                if j == n_task_batches - 1:
                    indices = task_permutation[j * args.tasks_batch_size:]
                else:
                    indices = task_permutation[j * args.tasks_batch_size:(j + 1) * args.tasks_batch_size]

                obs, actions, rewards, next_obs, goals = [], [], [], [], []
                for idx in indices:
                    # random_subset = np.random.permutation(dataset[idx][0].shape[1], )
                    # random_subset = np.random.choice(dataset[idx][0].shape[1], args.vae_batch_num_rollouts_per_task)
                    goals.append(ptu.FloatTensor(tasks[idx][:, traj_indices, :]))
                    obs.append(ptu.FloatTensor(dataset[idx][0][:, traj_indices, :]))
                    actions.append(ptu.FloatTensor(dataset[idx][1][:, traj_indices, :]))
                    rewards.append(ptu.FloatTensor(dataset[idx][2][:, traj_indices, :]))
                    next_obs.append(ptu.FloatTensor(dataset[idx][3][:, traj_indices, :]))

                obs = torch.cat(obs, dim=1)
                actions = torch.cat(actions, dim=1)
                rewards = torch.cat(rewards, dim=1)
                next_obs = torch.cat(next_obs, dim=1)
                goals = torch.cat(goals, dim=1)
                rew_recon_loss, state_recon_loss, kl_term, task_recon_loss,\
                    baseline_task_recon_loss = update_step(vae, obs, actions, rewards, next_obs, args, goals)

                # take average (this is the expectation over p(M))
                loss = args.rew_loss_coeff * rew_recon_loss + \
                       args.state_loss_coeff * state_recon_loss + \
                       args.kl_weight * kl_term + args.task_loss_coeff * task_recon_loss

                # update
                # update
                vae.optimizer.zero_grad()
                loss.backward()
                vae.optimizer.step()


                n_updates += 1
                loss_tr += loss.item()
                rew_loss_tr += rew_recon_loss.item()
                state_loss_tr += state_recon_loss.item()
                task_loss_tr += task_recon_loss.item()
                baseline_task_loss_tr += baseline_task_recon_loss.item()
                kl_loss_tr += kl_term.item()
            # print("--- %s seconds ---" % (time.time() - start_time))
            if (i + 1) % args.log_interval == 0:
                len_bar = int((BAR_LENGTH * (i + 1)) / n_batches)
                bar = ('=' * len_bar + '>').ljust(BAR_LENGTH, '.')
                idx = str(i + 1).rjust(len(str(n_batches)), ' ')

                tmpl = '{}/{}: [{}]'.format(idx, n_batches, bar)
                print('Epoch {} '.format(iter_ + 1) + tmpl)

        print('Epoch {} '.format(iter_ + 1))
        print('Elapsed time: {:.2f}, loss: {:.4f} -- rew_loss: {:.4f} -- state_loss: {:.4f} -- task_loss: {:.4f}-- '
              'kl: {:.4f} -- Baseline task Loss: {:.4f}'
              .format(time.time() - start_time, loss_tr / n_updates, rew_loss_tr / n_updates,
                      state_loss_tr / n_updates, task_loss_tr / n_updates, kl_loss_tr / n_updates, baseline_task_loss_tr/n_updates))

        total_updates += n_updates
        # log tb
        if (iter_) % eval_interval == 0:
            print('performing evaluation:')
            val_loss, val_rew_recon_loss, val_state_recon_loss, val_task_recon_loss, val_kl_term, \
            val_baseline_task_recon_loss = get_loss_on_validation_set(vae, val_dataset, val_goals, args)
            print('Validation Loss: ', val_loss)

            # add in validation metrics
            writer.add_scalar('val_loss/vae_loss', val_loss, total_updates)
            writer.add_scalar('val_loss/rew_recon_loss', val_rew_recon_loss, total_updates)
            writer.add_scalar('val_loss/state_recon_loss', val_state_recon_loss, total_updates)
            writer.add_scalar('val_loss/task_recon_loss', val_task_recon_loss, total_updates)
            writer.add_scalar('val_loss/baseline_task_recon_loss', val_baseline_task_recon_loss, total_updates)
            writer.add_scalar('val_loss/kl', val_kl_term, total_updates)

        if args.log_tensorboard:
            writer.add_scalar('loss/vae_loss', loss_tr / n_updates, total_updates)
            writer.add_scalar('loss/rew_recon_loss', rew_loss_tr / n_updates, total_updates)
            writer.add_scalar('loss/state_recon_loss', state_loss_tr / n_updates, total_updates)
            writer.add_scalar('loss/task_recon_loss', task_loss_tr / n_updates, total_updates)
            writer.add_scalar('loss/baseline_task_recon_loss', baseline_task_loss_tr / n_updates, total_updates)
            writer.add_scalar('loss/kl', kl_loss_tr / n_updates, total_updates)

            if args.env_name != 'GridNavi-v2':  # TODO: eval for gridworld domain
                rewards_eval, reward_preds_eval = eval_vae(dataset, vae, args)
                for task in range(10):
                    writer.add_figure('reward_prediction/task_{}'.format(task),
                                      utl_eval.plot_rew_pred_vs_rew(rewards_eval[task, :],
                                                                    reward_preds_eval[task, :]),
                                      total_updates)



        if args.save_model and (iter_ + 1) % args.save_interval == 0:
            save_path = os.path.join(os.getcwd(), args.full_save_path, 'models')
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            torch.save(vae.encoder.state_dict(),
                       os.path.join(save_path, "encoder{0}.pt".format(iter_ + 1)))
            if vae.reward_decoder is not None:
                torch.save(vae.reward_decoder.state_dict(),
                           os.path.join(save_path, "reward_decoder{0}.pt".format(iter_ + 1)))
            if vae.state_decoder is not None:
                torch.save(vae.state_decoder.state_dict(),
                           os.path.join(save_path, "state_decoder{0}.pt".format(iter_ + 1)))

def reformat_goals(goals, dataset):
    bamdp_length = len(dataset[0][0])
    num_rollouts = len(dataset[0][0][0])
    task_dim = 2
    new_goals = np.empty([len(dataset), len(dataset[0][0]), len(dataset[0][0][0]), task_dim])

    for i in range(len(goals)):
        x = goals[i]
        x_list = x.tolist()
        x_list = x_list * (bamdp_length * num_rollouts)
        x = np.array(x_list)
        x = np.reshape(x, (bamdp_length, num_rollouts, task_dim))
        new_goals[i] = x
    return new_goals

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-type', default='gridworld')
    # parser.add_argument('--env-type', default='point_robot_sparse')
    # parser.add_argument('--env-type', default='cheetah_vel')
    # parser.add_argument('--env-type', default='ant_semicircle_sparse')
    args, rest_args = parser.parse_known_args()
    env = args.env_type

    # --- GridWorld ---
    if env == 'gridworld':
        args = args_gridworld.get_args(rest_args)
    # --- PointRobot ---
    elif env == 'point_robot_sparse':
        args = args_point_robot_sparse.get_args(rest_args)
    # --- Mujoco ---
    elif env == 'cheetah_vel':
        args = args_cheetah_vel.get_args(rest_args)
    elif env == 'ant_semicircle_sparse':
        args = args_ant_semicircle_sparse.get_args(rest_args)

    set_gpu_mode(torch.cuda.is_available() and args.use_gpu)

    args, env = off_utl.expand_args(args)


    dataset, goals = off_utl.load_dataset(data_dir=args.data_dir, args=args, arr_type='numpy')

    # dataset, goals = off_utl.load_dataset(args)
    if args.hindsight_relabelling:
        print('Perform reward relabelling...')
        dataset, goals = off_utl.mix_task_rollouts(dataset, env, goals, args)
    dataset = dataset[:11]
    goals = goals[:11]
    # reformat goals to be usable for task decoder # should be in form nd.array(bamdp length, num_rollouts, task, dim)
    # only do this if task pred type id description?
    # bamdp_length = len(dataset[0][0])
    # num_rollouts = len(dataset[0][0][0])
    # task_dim = 2
    # new_goals = np.empty([len(dataset), len(dataset[0][0]), len(dataset[0][0][0]), task_dim])
    #
    # for i in range(len(goals)):
    #     x = goals[i]
    #     x_list = x.tolist()
    #     x_list = x_list * (bamdp_length * num_rollouts)
    #     x = np.array(x_list)
    #     x = np.reshape(x, (bamdp_length, num_rollouts, task_dim))
    #     new_goals[i] = x


    # vis test tasks
    # vis_train_tasks(env.unwrapped, goals)     # not with GridNavi
    if args.save_model:
        dir_prefix = args.save_dir_prefix if hasattr(args, 'save_dir_prefix') \
                                             and args.save_dir_prefix is not None else ''
        args.full_save_path = os.path.join(args.save_dir, args.env_name,
                                           dir_prefix + datetime.datetime.now().strftime('__%d_%m_%H_%M_%S'))
        os.makedirs(args.full_save_path, exist_ok=True)
        config_utl.save_config_file(args, args.full_save_path)

    args.decode_reward = False
    # args.task_pred_type = 'task_id'
    args.task_pred_type = 'task_description'
    # if args.task_pred_type == 'task_description':
    goals = reformat_goals(goals, dataset)
    # ipdb.set_trace()

    vae = VAE(args)

    train_experiments = int((1 - validation_ratio) * len(dataset))
    train_dataset = dataset[:train_experiments]
    train_goals = goals[:train_experiments]
    val_dataset = dataset[train_experiments:]
    val_goals = goals[train_experiments:]

    train(vae, train_dataset, train_goals, val_dataset, val_goals, args, 5)


if __name__ == '__main__':
    main()
