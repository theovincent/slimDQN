import time
import random
from tqdm import tqdm
import numpy as np
import torch
from slimRL.networks.architectures.DQN import BasicDQN
from slimRL.sample_collection.replay_buffer import ReplayBuffer
from slimRL.sample_collection.schedules import linear_schedule
from experiments.base.logger import save_logs


def train(
    p: dict,
    agent: BasicDQN,
    env,
    rb: ReplayBuffer,
):
    seed = p["seed"]
    learning_starts = p["n_initial_samples"]
    run_name = f"{p['env']}__{p['algo']}__{seed}__{int(time.time())}"

    print(run_name)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    n_training_steps = 0
    obs, _ = env.reset()
    losses = np.zeros((p["n_epochs"], p["n_training_steps_per_epoch"])) * np.nan
    js = np.zeros(p["n_epochs"]) * np.nan

    for idx_epoch in tqdm(range(p["n_epochs"])):
        sum_reward = 0
        n_episodes = 0
        idx_training_step = 0
        has_reset = False

        while idx_training_step < p["n_training_steps_per_epoch"] or not has_reset:
            epsilon = linear_schedule(
                p["end_epsilon"],
                p["duration_epsilon"],
                n_training_steps,
            )
            if random.random() < epsilon:
                action = random.sample(env.single_action_space, 1)
            else:
                action = [agent.best_action(obs)]

            next_obs, reward, termination, infos = env.step(action)
            has_reset = "episode_end" in infos.keys() and infos["episode_end"]
            rb.add(obs, action, reward, termination, has_reset)
            obs = next_obs

            if termination or has_reset:
                next_obs, _ = env.reset()

            sum_reward += reward
            n_episodes += int(termination or has_reset)

            if n_training_steps > learning_starts:
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
