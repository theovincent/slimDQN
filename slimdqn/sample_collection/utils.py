from functools import partial
import time
import jax
import jax.numpy as jnp
from functools import partial

from slimdqn.sample_collection.replay_buffer import ReplayBuffer, TransitionElement


@partial(jax.jit, static_argnames=("q", "n_actions", "epsilon_fn"))
def select_action(q, params, state, key, n_actions, epsilon_fn, n_training_steps):
    uniform_key, action_key = jax.random.split(key)
    return jnp.where(
        jax.random.uniform(uniform_key) <= epsilon_fn(n_training_steps),  # if uniform < epsilon,
        jax.random.randint(action_key, (), 0, n_actions),  # take random action
        jnp.argmax(q.apply(params, state)),  # otherwise, take a greedy action
    )


def collect_single_sample(key, env, agent, rb: ReplayBuffer, p, epsilon_schedule, n_training_steps: int):
    time_being_action_selection = time.time()
    action = select_action(
        agent.q_network, agent.params, agent.state, key, env.n_actions, epsilon_schedule, n_training_steps
    )
    jax.block_until_ready(action)
    time_action_selection = time.time() - time_being_action_selection

    obs = env.observation

    time_begin_step = time.time()
    reward, absorbing = env.step(action)
    time_step = time.time() - time_begin_step

    episode_end = absorbing or env.n_steps >= p["horizon"]

    time_begin_add = time.time()
    rb.add(
        TransitionElement(
            observation=obs,
            action=action,
            reward=reward if rb._clipping is None else rb._clipping(reward),
            is_terminal=absorbing,
            episode_end=episode_end,
        )
    )
    time_add = time.time() - time_begin_add

    if episode_end:
        env.reset()

    return reward, episode_end, time_action_selection, time_step, time_add
