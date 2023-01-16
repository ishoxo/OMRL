
import torch
import os
import argparse

import utils.config_utils as config_utl

from torchkit.pytorch_utils import set_gpu_mode
from offline_config import args_point_robot_sparse, args_gridworld

from Visualization.Evaluation import evaluate, visualise_behaviour

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
args.pass_belief_to_policy = False
args.pass_task_to_policy = False
args.add_nonlinearity_to_latent = False
args.sample_embeddings = False
args.norm_actions_post_sampling = False
args.num_processes =1

learner = OfflineMetaLearner(args)

# evaluate(args=learner.args,
#          policy=learner.agent,
#         ret_rms=None,
#              iter_idx=0,
#              tasks=None,
#              encoder=learner.vae.encoder,
#              num_episodes=None
#              )

visualise_behaviour(args = learner.args,
                        policy=learner.agent,
                        image_folder=None,
                        iter_idx = 0,
                        ret_rms=None,
                        tasks=None,
                        encoder=learner.vae.encoder,
                        reward_decoder=learner.vae.reward_decoder,
                        state_decoder=None,
                        task_decoder=None,
                        compute_rew_reconstruction_loss=None,
                        compute_task_reconstruction_loss=None,
                        compute_state_reconstruction_loss=None,
                        compute_kl_loss=None,
                        )

