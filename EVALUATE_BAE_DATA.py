import os
import numpy as np
import ipdb
from collections import Counter
env = "GridNavi-v2"
# env = "PointRobotSparse-v0"
data_path = "batch_data_multi/{}/data".format(env)
# data_path = "batch_data/{}/data".format(env)

""""
get times spent on goal but also look for evidence of learning
"""


def get_proportion_on_goal(rewards):
    rewards = [max(0, item[0]) for item in rewards]
    # ipdb.set_trace()
    return sum(rewards)/len(rewards)

def get_reward_entropy(path):
    all_rewards = []
    files = os.listdir(path)
    for file in files:
        file_path = path + '/' + file
        rewards = np.load(file_path + '/rewards.npy')
        rewards = [item[0] for item in rewards]
        all_rewards.append(rewards)
    all_rewards = [r for sublist in all_rewards for r in sublist]
    print(Counter(all_rewards).keys())# equals to list(set(words))
    print(Counter(all_rewards).values())  # counts the elements' frequency
    # ipdb.set_trace()

def Evaluate_data_collection(path):

    files = os.listdir(path)
    experiment_lengths = []
    experiment_goal_times = []

    goal_prop = []
    for file in files:
        file_path = path + '/' + file
        obs = np.load(file_path + '/obs.npy')
        actions = np.load(file_path + '/actions.npy')
        rewards = np.load(file_path + '/rewards.npy')
        # ipdb.set_trace()

        print('Agent is at goal {} % of the time'.format(get_proportion_on_goal(rewards)))



# get_reward_entropy(data_path)
Evaluate_data_collection(data_path)