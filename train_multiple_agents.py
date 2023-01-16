
import os
import argparse

import ipdb
import torch
import numpy as np
from utils import helpers as utl, offline_utils as off_utl
from learner import Learner
from data_management.storage_policy import MultiTaskPolicyStorage
from torchkit.pytorch_utils import set_gpu_mode
from data_collection_config import args_ant_semicircle_sparse, \
    args_cheetah_vel, args_point_robot_sparse, args_gridworld



def main():
    num_tasks = 21
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

    set_gpu_mode(torch.cuda.is_available())
    # ipdb.set_trace()

    if hasattr(args, 'save_buffer') and args.save_buffer:
        os.makedirs(args.main_save_dir, exist_ok=True)


    args.main_save_dir = './batch_data_multi'
    args.seed = 15
    learner = Learner(args)
    env_dir = os.path.join(learner.args.main_save_dir,
                           '{}'.format(learner.args.env_name))
    unwrapped_env = learner.env.unwrapped
    _, _, _, info = unwrapped_env.step(unwrapped_env.action_space.sample())
    reward_types = [reward_type for reward_type in list(info.keys()) if reward_type.startswith('reward')]
    for i in range(num_tasks):
        print('beggining training of agent {}'.format(i))

        learner.env.reset(task=i)
        learner.initialize_policy()

        learner.policy_storage = MultiTaskPolicyStorage(
            max_replay_buffer_size=int(learner.args.policy_buffer_size),
            obs_dim=learner.args.obs_dim,
            action_space=learner.env.action_space,
            tasks=[0],
            trajectory_len=args.max_trajectory_len,
            num_reward_arrays=len(reward_types) if reward_types and learner.args.dense_train_sparse_test else 1,
            reward_types=reward_types,
        )
        goal = learner.env.unwrapped._goal
        learner.output_dir = os.path.join(env_dir, learner.args.save_dir, 'seed_{}_'.format(learner.args.seed) +
                                       off_utl.create_goal_path_ext_from_goal(goal))
        os.makedirs(learner.output_dir, exist_ok=True)
        print(learner.output_dir)
        print('Agent goal is {}'.format(learner.env.unwrapped._goal))
        learner.train()


if __name__ == '__main__':
    main()