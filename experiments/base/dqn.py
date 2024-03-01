import time
import random
import numpy as np
import torch
from tqdm import tqdm
from slimRL.networks.architectures.dqn import BasicDQN
from slimRL.sample_collection.replay_buffer import ReplayBuffer
from slimRL.utils.misc import linear_schedule
import matplotlib.pyplot as plt

def train(
    p: dict,
    agent: BasicDQN,
    env,
    rb: ReplayBuffer,
):
    env_id = p["env_id"]
    agent_type = p["agent"]
    seed = p["seed"]
    exploration_fraction = p["exploration_fraction"]
    learning_starts = p["learning_starts"]
    run_name = f"{env_id}__{agent_type}__{seed}__{int(time.time())}"

    print(run_name)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    obs, _ = env.reset()
    losses = np.zeros((p["n_epochs"], p["n_training_steps_per_epoch"])) * np.nan
    js = np.zeros(p["n_epochs"]) * np.nan

    for idx_epoch in tqdm(range(p["n_epochs"])):
        sum_reward = 0
        n_episodes = 0
        idx_training_step = 0
        has_reset = False

        while idx_training_step < p["n_training_steps_per_epoch"] or not has_reset:
            epsilon = linear_schedule(p["start_epsilon"], p["end_epsilon"], p["exploration_fraction"] p["duration_epsilon"], n_training_steps)
            if random.random() < epsilon:
                action = random.sample(env.single_action_space, 1)
            else:
                action = [agent.best_action(obs)]

            next_obs, reward, termination, infos = env.step(action)
            episode_end = "episode_end" in infos.keys() and infos["episode_end"]
            rb.add(obs, action, reward, termination, episode_end)
            obs = next_obs

            if termination or episode_end:
                next_obs, _ = env.reset()

            sum_reward += reward
            n_episodes += int(episode_end)

            if n_training_steps > learning_starts:
                losses[
                idx_epoch, np.minimum(idx_training_step, p["n_training_steps_per_epoch"] - 1)
                ] = agent.update_online_params(n_training_steps, rb)
                agent.update_target_params(n_training_steps)

            idx_training_step += 1
            n_training_steps += 1

        js[idx_epoch] = sum_reward / n_episodes
        
        
