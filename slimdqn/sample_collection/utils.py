from functools import partial
import time
import jax
import jax.numpy as jnp
from functools import partial

from slimdqn.sample_collection.replay_buffer import ReplayBuffer
from slimdqn.sample_collection.elements import TransitionElement


@partial(jax.jit, static_argnums=(0, 2, 3))
def linearly_decaying_epsilon(decay_period, step, warmup_steps, epsilon):
    steps_left = decay_period + warmup_steps - step
    bonus = (1.0 - epsilon) * steps_left / decay_period
    bonus = jnp.clip(bonus, 0.0, 1.0 - epsilon)
    return epsilon + bonus


@partial(jax.jit, static_argnums=(0, 4, 5, 6, 8, 9))
def select_action(
    action_fn,
    params,
    state,
    rng,
    n_actions,
    epsilon_train,
    epsilon_decay_period,
    training_steps,
    min_replay_history,
    epsilon_fn,
):
    epsilon = epsilon_fn(
        epsilon_decay_period,
        training_steps,
        min_replay_history,
        epsilon_train,
    )

    rng1, rng2 = jax.random.split(rng, num=2)
    p = jax.random.uniform(rng1)
    return jnp.where(
        p <= epsilon,
        jax.random.randint(rng2, (), 0, n_actions),
        action_fn(params, state),
    )


def collect_single_sample(
    key,
    env,
    agent,
    rb: ReplayBuffer,
    p,
    n_training_steps: int,
):

    t1 = time.time()
    action = select_action(
        action_fn=agent.best_action,
        params=agent.params,
        state=env.state,
        rng=key,
        n_actions=env.n_actions,
        epsilon_train=p["epsilon_end"],
        epsilon_decay_period=p["epsilon_duration"],
        training_steps=n_training_steps,
        min_replay_history=p["n_initial_samples"],
        epsilon_fn=linearly_decaying_epsilon,
    ).astype(jnp.int8)
    time_action_selection = time.time() - t1
    # print(action)

    # if jax.random.uniform(epsilon_key) < epsilon_schedule(n_training_steps):
    #     key, sample_key = jax.random.split(key)
    #     action = jax.random.choice(sample_key, jnp.arange(env.n_actions)).item()
    # else:
    #     action = agent.best_action(agent.params, env.state).item()

    obs = env.observation

    t_s = time.time()
    reward, absorbing = env.step(action)
    time_step = time.time() - t_s

    episode_end = absorbing or env.n_steps >= p["horizon"]

    t_s = time.time()
    rb.add(
        TransitionElement(
            observation=obs,
            action=action,
            reward=reward if rb._clipping is None else rb._clipping(reward),
            is_terminal=absorbing,
            episode_end=episode_end,
        )
    )
    time_add = time.time() - t_s

    if episode_end:
        env.reset()

    return reward, episode_end, time_add, time_step, time_action_selection
