import os
import random
import json
import numpy as np
from slimRL.sample_collection.schedules import linear_schedule
from slimRL.sample_collection.replay_buffer import ReplayBuffer


def collect_single_sample(env, agent, rb: ReplayBuffer, p, n_training_steps: int):
    epsilon = linear_schedule(
        p.get("end_epsilon", 1),  # default values are to handle FQI and keep epsilon=1
        p.get("duration_epsilon", -1),
        n_training_steps,
    )
    if random.random() < epsilon:
        action = random.randint(0, env.n_actions - 1)
    else:
        action = agent.best_action(env.state)

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
        open(os.path.join(save_path, f"replay_buffer.json"), "w"),
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
