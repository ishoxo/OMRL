import os
import argparse
from utils import helpers as utl, offline_utils as off_utl
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
from offline_config import args_point_robot_sparse, args_gridworld
import matplotlib.pyplot as plt

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

args.vae_model_name = 'relabel__10_01_13_38_20' ## single head

set_gpu_mode(torch.cuda.is_available() and args.use_gpu)
vae_args = config_utl.load_config_file(os.path.join(args.vae_dir, args.env_name,
                                                        args.vae_model_name, 'online_config.json'))

args = config_utl.merge_configs(vae_args, args)

learner = OfflineMetaLearner(args)

# Function to relabel MDP data --> BAMDP
def Get_BAMDP_Data():
    vae_models_path = os.path.join(args.vae_dir, args.env_name,
                                   args.vae_model_name, 'models')
    vae = learner.vae
    off_utl.load_trained_vae(vae, vae_models_path)
    # load data and relabel
    save_data_path = os.path.join(args.main_data_dir, args.env_name, args.relabelled_data_dir)
    os.makedirs(save_data_path, exist_ok=True)
    dataset, goals = off_utl.load_dataset(data_dir=args.data_dir, args=args, arr_type='numpy', num_tasks=1)
    bamdp_dataset = off_utl.transform_mdps_ds_to_bamdp_ds(dataset, vae, args)

    #Now get single episode: obs, next_obs, actions, rewards.
    bamdp_dataset = bamdp_dataset[0]
    all_obs = bamdp_dataset[0]
    all_actions = bamdp_dataset[1]
    all_rewards = bamdp_dataset[2]
    all_next_obs = bamdp_dataset[3]

    episode_obs = ptu.FloatTensor(all_obs[:, 0])
    episode_actions = ptu.FloatTensor(all_actions[:, 0])
    episode_rewards = ptu.FloatTensor(all_rewards[:, 0])
    episode_next_obs = ptu.FloatTensor(all_next_obs[:, 0])
    return episode_obs, episode_actions, episode_rewards, episode_next_obs

def Get_Compression_Error(belief, reward_history, action_history, observation_history, next_observation_history, vae):
    compression_error = 0
    # ipdb.set_trace()
    samples = utl.sample_gaussian(belief[:5], belief[5:], 1)


    for i in range(len(reward_history)):
        gt_reward = reward_history[i]
        # ipdb.set_trace()
        predicted_reward = 0
        for j in range(len(samples)):
            predicted_reward += vae.reward_decoder(samples[j], next_observation_history[i][:3],
                                                   observation_history[i][:3], action_history[i]).item()/(len(samples))

        error = (gt_reward - predicted_reward) ** 2
        compression_error += error

    return compression_error

def Evaluate_Compression_Improvement():
    episode_obs, episode_actions, episode_rewards, episode_next_obs = Get_BAMDP_Data()
    # ipdb.set_trace()
    episode_length = len(episode_obs)

    for step in range(2, episode_length+1):
        reward_history = episode_rewards[:step]
        observation_history = episode_obs[:step]
        next_observation_history = episode_next_obs[:step]
        action_history = episode_actions[:step]

        current_belief = observation_history[-1, 3:]
        prior_belief = observation_history[-2, 3:]

        prior_compression_error = Get_Compression_Error(prior_belief, reward_history, action_history,
                                                        observation_history, next_observation_history, learner.vae)
        post_compression_error = Get_Compression_Error(current_belief, reward_history, action_history,
                                                       observation_history, next_observation_history,learner.vae)
        improvement = prior_compression_error - post_compression_error

        print(improvement)

def Eval_Convergence_All_History():
    episode_obs, episode_actions, episode_rewards, episode_next_obs = Get_BAMDP_Data()
    episode_length = len(episode_obs)

    compression_errors = []

    for b in range(episode_length-1):
        belief = episode_obs[:][b, 3:]
        new_belief = episode_next_obs[:][b, 3:]
        # ipdb.set_trace()
        prior_compression_error = Get_Compression_Error(belief, episode_rewards, episode_actions,
                                                        episode_obs, episode_next_obs, learner.vae)
        # post_compression_error = Get_Compression_Error(new_belief, episode_rewards, episode_actions,
        #                                                 episode_obs, episode_next_obs, learner.vae)
        # improvement = prior_compression_error - post_compression_error
        compression_errors.append(prior_compression_error)

    plt.plot(compression_errors)
    plt.ylabel('Compression Error')
    plt.xlabel('Belief at time t')
    plt.grid(True)
    plt.show()






Eval_Convergence_All_History()




# Evaluate_Compression_Improvement()
