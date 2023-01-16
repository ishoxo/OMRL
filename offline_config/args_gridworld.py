import argparse
import torch
from utils.cli import boolean_argument


def get_args(rest_args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', default='GridNavi-v2')
    parser.add_argument('--seed', type=int, default=73)

    parser.add_argument('--belief-rewards', default=False, help='use R+=E[R]')
    parser.add_argument('--num-belief-samples', default=20)
    parser.add_argument('--num-train-tasks', default=21)
    parser.add_argument('--num-eval-tasks', default=21)
    parser.add_argument('--hindsight-relabelling', default=False)

    parser.add_argument('--num-trajs-per-task', type=int, default=None,
                        help='how many trajs per task to use. If None - use all')
    parser.add_argument('--meta-batch', type=int, default=16,
                        help='number of tasks to average the gradient across')

    parser.add_argument('--num-iters', type=int, default=3000, help='number meta-training iterates')
    parser.add_argument('--rl-updates-per-iter', type=int, default=250, help='number of RL steps per iteration')
    parser.add_argument('--batch-size', type=int, default=256, help='number of transitions in RL batch (per task)')

    parser.add_argument('--dqn-layers', nargs='+', default=[64, 64])

    parser.add_argument('--policy-lr', type=float, default=0.0003, help='learning rate for policy (default: 3e-4)')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--soft-target-tau', type=float, default=0.005,
                        help='soft target network update (default: 5e-3)')
    parser.add_argument('--eval-deterministic', default=True)

    parser.add_argument('--transform-data-bamdp', default=False,
                        help='If true - perform state relabelling to bamdp, else - use existing data')

    # logging, saving, evaluation
    parser.add_argument('--log-interval', type=int, default=10,
                        help='log interval, one log per n iterations (default: 10)')
    parser.add_argument('--save-interval', type=int, default=50,
                        help='save models interval, every # iterations (default: 100)')

    parser.add_argument('--log-tensorboard', default=True, help='whether to use tb logger')
    parser.add_argument('--use-gpu', default=True, help='whether to use gpu')

    parser.add_argument('--results-log-dir', default='./logs', help='directory to save agent logs (default: ./logs)')

    parser.add_argument('--output-file-prefix', default='offline')
    # parser.add_argument('--output-file-prefix', default='offline_no_rr')

    parser.add_argument('--relabelled-data-dir', default='data_bamdp')
    # parser.add_argument('--relabelled-data-dir', default='data_bamdp_no_rr')

    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--main-data-dir', default='./batch_data_multi')

    parser.add_argument('--vae-dir', default='./trained_vae')

    parser.add_argument('--vae-model-name', default='relabel__08_01_17_38_11')
    # parser.add_argument('--vae-model-name', default='no_relabel__08_08_23_08_18')

    parser.add_argument('--policy-buffer-size', default=1e6)
    parser.add_argument('--num-rollouts-per-iter', default=5)

    args = parser.parse_args(rest_args)

    return args
