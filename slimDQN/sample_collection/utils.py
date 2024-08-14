import jax
import jax.numpy as jnp

from slimDQN.sample_collection.replay_buffer import ReplayBuffer


def collect_single_sample(
    key,
    env,
    agent,
    rb: ReplayBuffer,
    p,
    epsilon_schedule,
    n_training_steps: int,
):
    key, epsilon_key = jax.random.split(key)

    if jax.random.uniform(epsilon_key) < epsilon_schedule(n_training_steps):
        key, sample_key = jax.random.split(key)
        action = jax.random.choice(sample_key, jnp.arange(env.n_actions)).item()
    else:
        action = agent.best_action(agent.params, env.state).item()

    obs = env.observation
    reward, termination = env.step(action)

    episode_end = termination or env.n_steps >= p["horizon"]
    rb.add(obs, action, reward, termination, episode_end=episode_end)

    if episode_end:
        env.reset()

    return reward, episode_end
