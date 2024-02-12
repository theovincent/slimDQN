import time
import random
import numpy as np
import torch

from slimRL.environments.environment import Environment
from slimRL.networks.architectures.dqn import BasicDQN
from slimRL.sample_collection.replay_buffer import ReplayBuffer
from slimRL.utils.misc import linear_schedule
import matplotlib.pyplot as plt

def train(
    p: dict,
    agent: BasicDQN,
    env: Environment,
    rb: ReplayBuffer,
):
    env_id = p["env_id"]
    agent_type = p["agent"]
    seed = p["seed"]
    total_timesteps = p["total_timesteps"]
    start_epsilon = p["start_epsilon"]
    end_epsilon = p["end_epsilon"]
    exploration_fraction = p["exploration_fraction"]
    learning_starts = p["learning_starts"]
    run_name = f"{env_id}__{agent_type}__{seed}__{int(time.time())}"

    print(run_name)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    obs, _ = env.reset()
    all_episodes_rew = []
    curr_episode_rew = 0
    episode_step = 0
    for global_step in range(total_timesteps):
        epsilon = linear_schedule(start_epsilon, end_epsilon, int(exploration_fraction * total_timesteps),
                                  global_step)
        if random.random() < epsilon:
            action = random.sample(env.single_action_space, 1)
        else:
            action = [agent.best_action(obs)]

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, reward, termination, _ = env.step(action)
        curr_episode_rew += (p["gamma"]**episode_step) * reward
        episode_step += 1

        if termination:
            print(f"Step = {global_step}, Reward = {curr_episode_rew}")
            all_episodes_rew.append(curr_episode_rew)
            curr_episode_rew = 0
            episode_step = 0
            next_obs, _ = env.reset()

        rb.add(obs, action, reward, termination)

        obs = next_obs

        if global_step > learning_starts:
            agent.update_online_params(global_step, rb)
            agent.update_target_params(global_step)

    plt.plot(all_episodes_rew)
    plt.show()
    print(all_episodes_rew)
