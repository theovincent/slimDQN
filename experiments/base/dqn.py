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
        TIME_ADD, TIME_SAMPLE, TIME_GRAD, TIME_STEP, TIME_ACTION_SELECTION = (
            0,
            0,
            0,
            0,
            0,
        )
        ADD_OPS, SAMPLE_OPS, GRAD_OPS, STEP_OPS, ACTION_SELECTION_OPS = 0, 0, 0, 0, 0

        t1 = time.time()
        n_training_steps_epoch = 0
        has_reset = False

        while n_training_steps_epoch < p["n_training_steps_per_epoch"] or not has_reset:
            key, exploration_key = jax.random.split(key)
            (
                reward,
                has_reset,
                curr_time_add,
                curr_time_step,
                curr_time_action_selection,
            ) = collect_single_sample(exploration_key, env, agent, rb, p, n_training_steps, epsilon_schedule)
            TIME_ADD += curr_time_add
            ADD_OPS += 1

            TIME_STEP += curr_time_step
            STEP_OPS += 1

            TIME_ACTION_SELECTION += curr_time_action_selection
            ACTION_SELECTION_OPS += 1

            n_training_steps_epoch += 1
            n_training_steps += 1

            episode_returns_per_epoch[idx_epoch][-1] += reward
            episode_lengths_per_epoch[idx_epoch][-1] += 1
            if has_reset and n_training_steps_epoch < p["n_training_steps_per_epoch"]:
                episode_returns_per_epoch[idx_epoch].append(0)
                episode_lengths_per_epoch[idx_epoch].append(0)

            if n_training_steps > p["n_initial_samples"]:
                loss, curr_sample_time, curr_grad_time = agent.update_online_params(n_training_steps, rb)
                cumulated_loss += loss
                TIME_SAMPLE += curr_sample_time
                SAMPLE_OPS += curr_sample_time > 0
                TIME_GRAD += curr_grad_time
                GRAD_OPS += curr_grad_time > 0
                target_updated = agent.update_target_params(n_training_steps)

                if target_updated:
                    p["wandb"].log({"n_training_steps": n_training_steps, "loss": cumulated_loss})
                    cumulated_loss = 0

        avg_return = np.mean(episode_returns_per_epoch[idx_epoch])
        avg_length_episode = np.mean(episode_lengths_per_epoch[idx_epoch])
        n_episodes = len(episode_lengths_per_epoch[idx_epoch])
        print(
            f"\nEpoch {idx_epoch}: Return {avg_return} averaged on {n_episodes} episodes.\n",
            flush=True,
        )
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

        time_delta = time.time() - t1

        save_data(p, episode_returns_per_epoch, episode_lengths_per_epoch, agent.get_model())

        print(f"Total time for the epoch: {time_delta} s\n", flush=True)
        print(
            f"{ADD_OPS} add() took {TIME_ADD} s, average = {'NaN' if ADD_OPS == 0 else TIME_ADD/ADD_OPS} s\n",
            flush=True,
        )
        print(
            f"{SAMPLE_OPS} sample() took {TIME_SAMPLE} s, average = {'NaN' if SAMPLE_OPS == 0 else TIME_SAMPLE/SAMPLE_OPS} s\n",
            flush=True,
        )
        print(
            f"{STEP_OPS} step() took {TIME_STEP} s, average = {'NaN' if STEP_OPS == 0 else TIME_STEP/STEP_OPS} s\n",
            flush=True,
        )
        print(
            f"{GRAD_OPS} grad() took {TIME_GRAD} s, average = {'NaN' if GRAD_OPS == 0 else TIME_GRAD/GRAD_OPS} s\n",
            flush=True,
        )
        print(
            f"{ACTION_SELECTION_OPS} select_action() took {TIME_ACTION_SELECTION} s, average = {'NaN' if ACTION_SELECTION_OPS == 0 else TIME_ACTION_SELECTION/ACTION_SELECTION_OPS} s\n",
            flush=True,
        )
