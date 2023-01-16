"""
return number of experiments
whether all expeirments are the same legnth
length of experiments



takes as inputthe env name
"""
import os
import numpy as np
import ipdb

# env_name = "PointRobotSparse-v0"
env_name = "GridNavi-v2"

BAMDP_Version = False
def Describe_Experiments(env_name):
    data_path = "batch_data_multi/{}/data".format(env_name)

    if BAMDP_Version:
        data_path = "batch_data/{}/data_bamdp".format(env_name)

    files = os.listdir(data_path)

    num_experiments = len(files)
    print('Number of Experiments: {}'.format(num_experiments))
    experiment_episodes = []
    experiment_lengths = []
    for file in files:
        print(file)
        file_path = data_path + '/' + file
        experiment_lengths.append(len(np.load(file_path + '/actions.npy')))
        if BAMDP_Version:
            terminals = np.load(file_path + '/terminals.npy')
            experiment_lengths.append(int(sum(sum(terminals))))
            # ipdb.set_trace()

        else:
            experiment_episodes.append(int(sum(np.load(file_path + '/terminals.npy'))))
        # ipdb.set_trace()

    print('Experiment Lengths: {}'.format(experiment_lengths))
    print('Number of Episodes: {}'.format(experiment_episodes))
Describe_Experiments(env_name)