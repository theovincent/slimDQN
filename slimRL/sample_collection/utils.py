import os
import json
import optax
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
    _, reward, termination = env.step(action)
    truncation = env.n_steps == p["horizon"]
    rb.add(obs, action, reward, termination, truncation)

    has_reset = termination or truncation
    if has_reset:
        env.reset()

    return reward, has_reset


def save_replay_buffer_store(rb: ReplayBuffer, save_path):
    rb_store = {}

    for key in ["observations", "actions", "rewards", "dones"]:
        rb_store[key] = rb._store[key].tolist()
    rb_store["episode_trunc_indices"] = list(rb.episode_trunc_indices)
    rb_store["last_added_transition_index"] = rb.last_added_transition_index
    rb_store["add_count"] = rb.add_count

    json.dump(
        rb_store,
        open(os.path.join(save_path, "..", "replay_buffer.json"), "w"),
    )


def load_valid_transitions(rb_path):
    rb_store = json.load(open(os.path.join(rb_path), "r"))
    valid_indices = set(np.arange(len(rb_store["observations"]))) - set(rb_store["episode_trunc_indices"])
    if not rb_store["dones"][rb_store["last_added_transition_index"]]:
        valid_indices.discard(rb_store["last_added_transition_index"])
    valid_transitions = {}
    valid_transitions["next_observations"] = np.array(
        [val for idx, val in enumerate(np.roll(rb_store["observations"], -1, axis=0)) if idx in valid_indices]
    )
    for key in ["observations", "actions", "rewards", "dones"]:
        valid_transitions[key] = np.array([val for idx, val in enumerate(rb_store[key]) if idx in valid_indices])
    return valid_transitions


def update_replay_buffer(key, env, agent, rb: ReplayBuffer, p):
    if os.path.exists(os.path.join(p["save_path"], "..", "replay_buffer.json")):
        print("Replay buffer already exists. Loading...")
        rb_store = json.load(open(os.path.join(p["save_path"], "..", "replay_buffer.json"), "r"))
        rb._store = {name: np.array(rb_store[name]) for name in ["observations", "actions", "rewards", "dones"]}
        rb.episode_trunc_indices = set(rb_store["episode_trunc_indices"])
        rb.last_added_transition_index = rb_store["last_added_transition_index"]
        rb.add_count = rb_store["add_count"]

    else:
        env.reset()
        for _ in range(p["replay_capacity"]):
            key, sample_key = jax.random.split(key)
            collect_single_sample(
                sample_key,
                env,
                agent,
                rb,
                p,
                optax.linear_schedule(1.0, 1.0, -1),
                0,
            )
        assert sum(rb._store["rewards"] > 0) > 0, "No positive reward sampled. Rerun!"
        print(f"Replay buffer filled with {sum(rb._store['rewards'] > 0)} success samples.")
        save_replay_buffer_store(rb, p["save_path"])
