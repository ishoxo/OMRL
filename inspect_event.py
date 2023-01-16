import os
import ipdb
from tensorflow.python.summary.summary_iterator import summary_iterator
import traceback
import argparse
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np
import json
from matplotlib import pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', default='GridNavi-v2')
# parser.add_argument('--multihead_for_reward', default=False)
# parser.add_argument('--disable_stochasticity_in_latent', default=False)
# parser.add_argument('--rew_pred_type', default='deterministic')
parser.add_argument('--decode-task', default=True)
parser.add_argument('--decode-reward', default=False)
parser.add_argument('--learn_prior', default=False)
parser.add_argument('--task_pred_type', default='task_description', help='task_description or task id')



args, rest_args = parser.parse_known_args()
tag = 'policy_vis/task_1'

def get_path(args):
    configured_vaes = []
    iterations = []
    main_dir = "trained_vae/{}".format(args.env_name)
    files = os.listdir(main_dir)
    files = [file for file in files if os.path.exists(main_dir+'/'+file+'/models')]
    for file in files:
        match = True
        file_path = main_dir + '/' + file
        config = open(file_path+'/online_config.json')
        config = json.load(config)
        for k in args.__dict__:
            # ipdb.set_trace()
            if k in config.keys():
                if args.__dict__[k] == config[k]:
                    pass
                else:
                    match = False
                    break
        if match:
            configured_vaes.append(file_path)
            model_name = 'encoder'
            files = os.listdir(file_path+'/models')
            files = [file if file.startswith(model_name) else None for file in files]
            files = list(filter(None, files))

            idx = [file[len(model_name):-3] for file in files]
            idx = [int(item) for item in idx]
            iterations.append(max(idx))
    if len(configured_vaes) == 0:
        print('------------------------')
        print('No pretrained VAE matching specified config')
        print('------------------------')

    index = iterations.index(max(iterations))
    path = configured_vaes[index]
    files = os.listdir(path)
    event_files = [file for file in files if 'event' in file]
    event_path = path + '/' + event_files[0]

    return event_path


# vae_path = get_path(args)
vae_path = "trained_vae/GridNavi-v2/relabel__13_01_14_59_21/events.out.tfevents.1673618362.AZ-7VCG0F3"
# ipdb.set_trace()

def inspect_event(event_path):
    arr = []
    tags = []
    steps = []
    num_events = 0
    for event in summary_iterator(event_path):
        print(num_events)
        num_events+=1
        # ipdb.set_trace()
        if hasattr(event.summary, 'value') and len(event.summary.value) > 0:
            if event.summary.value[0].tag == tag:
                arr.append(event.summary.value[0].simple_value)
                tags.append(event.summary.value[0].tag)
                steps.append(event.step)

    ipdb.set_trace()

def tflog2pandas(path: str) -> pd.DataFrame:
    """convert single tensorflow log file to pandas DataFrame
    Parameters
    ----------
    path : str
        path to tensorflow log file
    Returns
    -------
    pd.DataFrame
        converted dataframe
    """
    DEFAULT_SIZE_GUIDANCE = {
        "compressedHistograms": 1,
        "images": 1,
        "scalars": 0,  # 0 means load all
        "histograms": 1,
    }
    runlog_data = pd.DataFrame({"metric": [], "value": [], "step": []})
    try:
        event_acc = EventAccumulator(path, DEFAULT_SIZE_GUIDANCE)
        event_acc.Reload()
        tags = event_acc.Tags()["scalars"]
        for tag in tags:
            event_list = event_acc.Scalars(tag)
            values = list(map(lambda x: x.value, event_list))
            step = list(map(lambda x: x.step, event_list))
            r = {"metric": [tag] * len(step), "value": values, "step": step}
            r = pd.DataFrame(r)
            runlog_data = pd.concat([runlog_data, r])
    # Dirty catch of DataLossError
    except Exception:
        print("Event file possibly corrupt: {}".format(path))
        traceback.print_exc()
    return runlog_data



def many_logs2pandas(event_paths):
    all_logs = pd.DataFrame()
    for path in event_paths:
        log = tflog2pandas(path)
        if log is not None:
            if all_logs.shape[0] == 0:
                all_logs = log
            else:
                all_logs = all_logs.append(log, ignore_index=True)
    return all_logs


def plot_metatraining_progress(event_path):


    runlog = tflog2pandas(event_path)

    fes = runlog[runlog['metric'] == 'returns_multi_episode/episode_1']['value']
    ses = runlog[runlog['metric'] == 'returns_multi_episode/episode_2']['value']
    tes = runlog[runlog['metric'] == 'returns_multi_episode/sum']['value']
    # ipdb.set_trace()
    ax = fes.plot(label='First Episode')
    ses.plot(ax=ax, label='Second Episode')
    tes.plot(ax=ax, label='All Episodes')
    plt.title('Agent Performance')
    plt.xlabel('Iteration')
    plt.ylabel('Episode Reward')
    plt.legend()
    plt.grid(True)
    plt.show()

def vae_training_vis(event_path):
    runlog = tflog2pandas(event_path)
    # ipdb.set_trace()

    vae_loss = runlog[runlog['metric'] == 'loss/vae_loss']['value']
    kl_loss = runlog[runlog['metric'] == 'loss/kl']['value']
    rew_loss = runlog[runlog['metric'] == 'loss/rew_recon_loss']['value']
    task_loss = runlog[runlog['metric'] == 'loss/task_recon_loss']['value']
    base_task_loss = runlog[runlog['metric'] == 'loss/baseline_task_recon_loss']['value']
    # ipdb.set_trace()
    ax = vae_loss.plot(label='Loss')
    kl_loss.plot(ax=ax, label='KL loss')
    rew_loss.plot(ax=ax, label='Rew loss')
    task_loss.plot(ax=ax, label='Task loss')
    base_task_loss.plot(ax=ax, label='Baseline task loss')
    plt.title('Context Encoder Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


def vae_val_vis(event_path):
    runlog = tflog2pandas(event_path)
    # ipdb.set_trace()

    vae_loss = runlog[runlog['metric'] == 'val_loss/vae_loss']['value']
    kl_loss = runlog[runlog['metric'] == 'val_loss/kl']['value']
    rew_loss = runlog[runlog['metric'] == 'val_loss/rew_recon_loss']['value']
    task_loss = runlog[runlog['metric'] == 'val_loss/task_recon_loss']['value']
    base_task_loss = runlog[runlog['metric'] == 'val_loss/baseline_task_recon_loss']['value']
    # ipdb.set_trace()
    ax = vae_loss.plot(label='Loss')
    kl_loss.plot(ax=ax, label='KL loss')
    rew_loss.plot(ax=ax, label='Rew loss')
    task_loss.plot(ax=ax, label='Task loss')
    base_task_loss.plot(ax=ax, label='Baseline task loss')
    plt.title('Validation Encoder Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

vae_training_vis(vae_path)

vae_val_vis(vae_path)


# plot_training_progress(event_path)
