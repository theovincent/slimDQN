import time
import random
from tqdm import tqdm
import numpy as np
import torch
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
    losses = np.zeros((p["n_epochs"], p["n_training_steps_per_epoch"])) * np.nan
    js = np.zeros(p["n_epochs"]) * np.nan

    for idx_epoch in tqdm(range(p["n_epochs"])):
        sum_reward = 0
        n_episodes = 0
        idx_training_step = 0
        has_reset = False

        while idx_training_step < p["n_training_steps_per_epoch"] or not has_reset:

            reward, has_reset = collect_single_sample(
                env, agent, rb, p, n_training_steps
            )

            sum_reward += reward
            n_episodes += int(has_reset)

            if n_training_steps > p["n_initial_samples"]:
                losses[
                    idx_epoch,
                    np.minimum(idx_training_step, p["n_training_steps_per_epoch"] - 1),
                ] = agent.update_online_params(n_training_steps, rb)
                agent.update_target_params(n_training_steps)

            idx_training_step += 1
            n_training_steps += 1

        js[idx_epoch] = sum_reward / n_episodes
        print(
            f"Epoch: {idx_epoch}, Avg. return = {js[idx_epoch]}, Num episodes = {n_episodes}"
        )

    save_logs(p, js, losses, agent)
