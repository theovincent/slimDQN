import time
import random
from tqdm import tqdm
import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
from slimRL.networks.architectures.DQN import BasicDQN
from slimRL.sample_collection.replay_buffer import ReplayBuffer
from slimRL.sample_collection.utils import collect_single_sample
from experiments.base.logger import save_logs


def train(
    p: dict,
    agent: BasicDQN,
    env,
    rb: ReplayBuffer,
):

    print(f"{p['env']}__{p['algo']}__{p['seed']}__{int(time.time())}")

    random.seed(p["seed"])
    np.random.seed(p["seed"])
    torch.manual_seed(p["seed"])

    n_training_steps = 0
    env.reset()
    log_rewards = []
    log_lengths = []
    linear_lr_schedule = lambda step: 1 - step / p["n_epochs"] * (
        1 - p["end_lr"] / p["start_lr"]
    )
    scheduler = LambdaLR(agent.optimizer, lr_lambda=linear_lr_schedule)

    for idx_epoch in tqdm(range(p["n_epochs"])):
        epoch_rewards = []
        epoch_episode_lengths = []
        episode_reward = 0
        episode_length = 0
        idx_training_step = 0
        has_reset = False

        while idx_training_step < p["n_training_steps_per_epoch"] or not has_reset:

            reward, has_reset = collect_single_sample(
                env, agent, rb, p, n_training_steps
            )

            episode_reward += reward
            episode_length += 1
            if has_reset:
                epoch_rewards.append(episode_reward)
                epoch_episode_lengths.append(episode_length)
                episode_reward = 0
                episode_length = 0

            if n_training_steps > p["n_initial_samples"]:
                agent.update_online_params(n_training_steps, p["batch_size"], rb)
                agent.update_target_params(n_training_steps)

            idx_training_step += 1
            n_training_steps += 1

        log_rewards.append(epoch_rewards)
        log_lengths.append(epoch_episode_lengths)

        print(
            f"Epoch: {idx_epoch}, Avg. return = {sum(epoch_rewards)/len(epoch_rewards)}, Num episodes = {len(epoch_rewards)}"
        )
        scheduler.step()

    save_logs(p, log_rewards, log_lengths, agent)
