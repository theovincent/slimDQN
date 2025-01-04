from functools import partial
import time
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

from slimdqn.sample_collection.replay_buffer import ReplayBuffer
from slimdqn.sample_collection.elements import TransitionElement


@partial(jax.jit, static_argnums=(0, 4, 6))
def select_action(
    network_def,
    params,
    state,
    rng,
    n_actions,
    n_training_steps,
    epsilon_fn,
):
    epsilon = epsilon_fn(n_training_steps)

    rng1, rng2 = jax.random.split(rng, num=2)
    p = jax.random.uniform(rng1)
    return jnp.where(
        p <= epsilon,
        jax.random.randint(rng2, (), 0, n_actions),
        jnp.argmax(network_def.apply(params, state)),
    )


def collect_single_sample(
    key,
    env,
    agent,
    rb: ReplayBuffer,
    p,
    n_training_steps: int,
    epsilon_schedule
):

    t1 = time.time()
    action = select_action(
        network_def=agent.q_network,
        params=agent.params,
        state=agent.state,
        rng=key,
        n_actions=env.n_actions,
        n_training_steps=n_training_steps,
        epsilon_fn=epsilon_schedule,
    )
    time_action_selection = time.time() - t1
    # print(action)

    # if jax.random.uniform(epsilon_key) < epsilon_schedule(n_training_steps):
    #     key, sample_key = jax.random.split(key)
    #     action = jax.random.choice(sample_key, jnp.arange(env.n_actions)).item()
    # else:
    #     action = agent.best_action(agent.params, env.state).item()

    obs = env.observation.squeeze()

    t_s = time.time()
    reward, absorbing = env.step(action)
    time_step = time.time() - t_s

    episode_end = absorbing or env.n_steps >= p["horizon"]
    
    if episode_end:
        agent.state.fill(0)
    else:
        agent.state = np.roll(agent.state, -1, axis=-1)
        agent.state[..., -1] = env.observation.squeeze()

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
