




import os
import numpy as np
import ipdb

IR_types = ['KL', 'WASS', 'VAR_RED', 'EUC']
IR_type = IR_types[0]
# env_name = "GridNavi-v2"
env_name = "PointRobotSparse-v0"


path = "batch_data_multi/{}/data_bamdp".format(env_name)



def Generate_IR_data(path):

    # bamdp_data_path = "batch_data/{}/data_bamdp".format(env_name)
    # bamdp_data_path = "batch_data/GridNavi-v2/data_bamdp/goal_0.0_3.0"
    # save_data_path = "batch_data/{}/data_bamdp_{}".format(env_name, IR_type)

    #Create New Directory
    # os.makedirs(save_data_path, exist_ok=True)


    # Get task files
    files = os.listdir(path)


    for file in files:
        file_path = path + '/' + file
        # file_path = bamdp_data_path

        obs = np.load(file_path + '/obs.npy')
        next_obs = np.load(file_path + '/next_obs.npy')
        actions = np.load(file_path + '/actions.npy')
        rewards = np.load(file_path + '/rewards.npy')
        KL_rewards = np.load(file_path + '/KL_rewards.npy')
        EUC_rewards = np.load(file_path + '/EUC_rewards.npy')
        VAR_rewards = np.load(file_path + '/VAR_rewards.npy')

        hybrid_KL_1 = rewards.copy()
        hybrid_KL_01 = rewards.copy()
        hybrid_KL_001 = rewards.copy()

        #obs[i][j] gives observation at timestep i of rollout j

        num_rollouts = 60
        num_time_steps = 60
        for rollout in range(num_rollouts):

            KL_IR = KL_rewards[:, rollout]
            VAR_IR = VAR_rewards[:, rollout]
            EUC_IR = EUC_rewards[:, rollout]
            rollout_reward = rewards[:, rollout]

            if rollout == 0:
                mean_KL = sum(KL_IR)/len(KL_IR)
            else:
                mean_KL = sum(KL_rewards[:, rollout-1])/len(KL_rewards[:, rollout-1])

            normalised_KL_IR = KL_IR / mean_KL

            hybrid_KL_1[:, rollout] += normalised_KL_IR
            hybrid_KL_01[:, rollout] += 0.1*normalised_KL_IR
            hybrid_KL_001[:, rollout] += 0.01*normalised_KL_IR

        np.save(os.path.join(file_path, 'hybrid_KL_1'), hybrid_KL_1)
        np.save(os.path.join(file_path, 'hybrid_KL_01'), hybrid_KL_01)
        np.save(os.path.join(file_path, 'hybrid_KL_001'), hybrid_KL_001)

        # ipdb.set_trace()











Generate_IR_data(path)