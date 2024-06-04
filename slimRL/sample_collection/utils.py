import os
import json
import jax
import jax.numpy as jnp
import numpy as np
from slimRL.sample_collection.replay_buffer import ReplayBuffer


def collect_single_sample(
    exploration_key,
    env,
    agent,
    rb: ReplayBuffer,
    p,
    epsilon_schedule,
    n_training_steps: int,
):

    sample_key, epsilon_key = jax.random.split(exploration_key)

    if jax.random.uniform(epsilon_key) < epsilon_schedule(n_training_steps):
        action = jax.random.choice(sample_key, jnp.arange(env.n_actions)).item()
    else:
        action = agent.best_action(agent.params, env.state).item()

    obs = env.state.copy()
    next_obs, reward, termination = env.step(action)
    truncation = env.n_steps == p["horizon"]
    rb.add(obs, action, reward, termination, truncation, next_obs)

    has_reset = termination or truncation
    if has_reset:
        env.reset()

    return reward, has_reset


def save_replay_buffer_store(rb: ReplayBuffer, save_path):

    rb_store = {}

    for key in ["observation", "action", "reward", "done"]:
        rb_store[key] = rb._store[key].tolist()

    rb_store["next_observations_trunc"] = {}
    for key in rb._store["next_observations_trunc"]:
        rb_store["next_observations_trunc"][key] = rb._store["next_observations_trunc"][
            key
        ].tolist()  # store the next observations for last state of truncated trajectories at rb_store["next_observations_trunc"][index of last state]

    rb_store["last_transition_next_obs"] = (
        rb._store["last_transition_next_obs"][
            0
        ],  # index of last recorded observation in replay buffer
        rb._store["last_transition_next_obs"][
            1
        ].tolist(),  # next state for last recorded observation in replay buffer
    )
    json.dump(
        rb_store,
        open(os.path.join(save_path, "..", "replay_buffer.json"), "w"),
    )


def load_replay_buffer_store(rb_path):

    rb_store = json.load(open(rb_path, "r"))
    for key in ["observation", "action", "reward", "done"]:
        rb_store[key] = np.array(rb_store[key])

    next_obs_keys = list(rb_store["next_observations_trunc"])
    for key in next_obs_keys:
        rb_store["next_observations_trunc"][int(key)] = np.array(
            rb_store["next_observations_trunc"].pop(
                key
            )  # as index is dumped by json in str format, pop removes str key from dict and adds an int key
        )

    rb_store["last_transition_next_obs"] = (
        rb_store["last_transition_next_obs"][0],
        np.array(rb_store["last_transition_next_obs"][1]),
    )
    return rb_store


def update_replay_buffer(env, agent, rb, p):
    if os.path.exists(os.path.join(p["save_path"], "..", "replay_buffer.json")):
        print("Replay buffer already exists. Loading...")
        rb._store = load_replay_buffer_store(
            os.path.join(p["save_path"], "..", "replay_buffer.json")
        )
        rb.episode_end_indices = set(np.where(rb._store["done"])[0].tolist())
        rb.episode_trunc_next_states = rb._store["next_observations_trunc"].copy()
        rb.add_count = len(rb._store["observation"])
    else:
        env.reset()
        for steps in range(p["replay_capacity"]):
            collect_single_sample(env, agent, rb, p, 0)
        assert sum(rb._store["reward"] == 1) > 0, "No positive reward sampled. Rerun!"
        print(
            f"Replay buffer filled with {sum(rb._store['reward'] == 1)} success samples."
        )
        save_replay_buffer_store(rb, p["save_path"])
