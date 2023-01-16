import numpy as np
import os
from matplotlib import pyplot as plt
import ipdb

# path = "batch_data/PointRobotSparse-v0/data/seed_54_goal_0.934_0.358"
# path2 = "batch_data/GridNavi-v2_misc/data/seed_54_goal_0_2"
# path2 = "batch_data/PointRobotSparse-v0/data/seed_72_goal_-0.132_0.991"
# path3 = "batch_data/PointRobotSparse-v0/data/seed_73_goal_-0.975_0.221"


# path = 'batch_data/GridNavi-v2/data/seed_0_goal_1_3'
path = 'batch_data/GridNavi-v2/data/seed_16_goal_0_3'


# def Eval_data_collection(path):
#     # ipdb.set_trace()
#     files = os.listdir(path)
#     for file in files:
#         if '.npy' in file:
#             file_path = path + '/' + file
#             traj = np.load(file_path)

def Eval_episodic_reward(path, ep_length):
    # ipdb.set_trace()

    terminals_path = path + '/terminals.npy'
    rewards_path = path + '/rewards.npy'


    rewards = np.load(rewards_path)
    terminals = np.load(terminals_path)

    # ipdb.set_trace()


    episode_rewards = []

    for i in range(int(len(rewards)/ep_length)):
        ep_reward = rewards[i*ep_length:  (i+1)*ep_length+1]
        episode_rewards.append(sum(ep_reward))


    plt.title('Episode Rewards')
    plt.plot(episode_rewards)
    plt.show()

# Eval_episodic_reward(path, 15)


def Get_Experiment_Lengths(path):
    files = os.listdir(path)
    # ipdb.set_trace()
    for file in files:
        file_path = path + '/' + file
        obs = np.load(file_path + '/obs.npy')

        print('Experiment: {}, Length: {}'.format(file, len(obs)))
        # ipdb.set_trace()
        # pass


my_path = "batch_data_multi/GridNavi-v2/data"

Get_Experiment_Lengths(my_path)



