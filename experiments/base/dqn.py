import time
import jax
import numpy as np
import optax
from tqdm import tqdm

from experiments.base.utils import save_data
from slimdqn.networks.dqn import DQN
from slimdqn.sample_collection.replay_buffer import ReplayBuffer
from slimdqn.sample_collection.utils import collect_single_sample


def train(
    key: jax.random.PRNGKey,
    p: dict,
    agent: DQN,
    env,
    rb: ReplayBuffer,
):

    epsilon_schedule = optax.linear_schedule(1.0, p["epsilon_end"], p["epsilon_duration"])

    n_training_steps = 0
    env.reset()
    episode_returns_per_epoch = [[0]]
    episode_lengths_per_epoch = [[0]]
    cumulated_loss = 0

    for idx_epoch in tqdm(range(p["n_epochs"])):
        TIME_ACTION_SELECTION, TIME_STEP, TIME_ADD, TIME_SAMPLE, TIME_GRAD = (
            0,
            0,
            0,
            0,
            0,
        )

        time_begin_epoch = time.time()
        n_training_steps_epoch = 0
        has_reset = False

        while n_training_steps_epoch < p["n_training_steps_per_epoch"] or not has_reset:
            key, exploration_key = jax.random.split(key)
            (
                reward,
                has_reset,
                time_action_selection,
                time_step,
                time_add,
            ) = collect_single_sample(exploration_key, env, agent, rb, p, epsilon_schedule, n_training_steps)

            TIME_ACTION_SELECTION += time_action_selection
            TIME_STEP += time_step
            TIME_ADD += time_add

            n_training_steps_epoch += 1
            n_training_steps += 1

            episode_returns_per_epoch[idx_epoch][-1] += reward
            episode_lengths_per_epoch[idx_epoch][-1] += 1
            if has_reset and n_training_steps_epoch < p["n_training_steps_per_epoch"]:
                episode_returns_per_epoch[idx_epoch].append(0)
                episode_lengths_per_epoch[idx_epoch].append(0)

            if n_training_steps > p["n_initial_samples"]:
                loss, sample_time, grad_time = agent.update_online_params(n_training_steps, rb)
                cumulated_loss += loss

                TIME_SAMPLE += sample_time
                TIME_GRAD += grad_time

                target_updated = agent.update_target_params(n_training_steps)

                if target_updated:
                    p["wandb"].log({"n_training_steps": n_training_steps, "loss": cumulated_loss})
                    cumulated_loss = 0

        avg_return = np.mean(episode_returns_per_epoch[idx_epoch])
        avg_length_episode = np.mean(episode_lengths_per_epoch[idx_epoch])
        n_episodes = len(episode_lengths_per_epoch[idx_epoch])
        print(f"\nEpoch {idx_epoch}: Return {avg_return} averaged on {n_episodes} episodes.\n", flush=True)
        p["wandb"].log(
            {
                "epoch": idx_epoch,
                "n_training_steps": n_training_steps,
                "avg_return": avg_return,
                "avg_length_episode": avg_length_episode,
            }
        )

        if idx_epoch < p["n_epochs"] - 1:
            episode_returns_per_epoch.append([0])
            episode_lengths_per_epoch.append([0])

        TIME_EPOCH = time.time() - time_begin_epoch

        save_data(p, episode_returns_per_epoch, episode_lengths_per_epoch, agent.get_model())

        print(
            f"Total time for the epoch: {TIME_EPOCH} s\n",
            f"add() took {TIME_ADD} s\n",
            f"sample() took {TIME_SAMPLE} s\n",
            f"step() took {TIME_STEP} s\n",
            f"grad() took {TIME_GRAD} s\n",
            f"select_action() took {TIME_ACTION_SELECTION} s\n",
            f"remaining operations took {TIME_EPOCH - TIME_ACTION_SELECTION - TIME_STEP - TIME_ADD - TIME_SAMPLE - TIME_GRAD}\n",
            flush=True,
        )
