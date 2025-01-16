from typing import Tuple
from functools import partial
import jax
import jax.numpy as jnp


class Generator:
    def __init__(self, batch_size: int, observation_dim: Tuple[int], n_actions: int) -> None:
        self.batch_size = batch_size
        self.observation_dim = observation_dim
        self.n_actions = n_actions

    @partial(jax.jit, static_argnames="self")
    def sample(
        self,
        key: jax.random.PRNGKey,
    ) -> Tuple[jnp.ndarray]:
        states = jax.random.uniform(key, self.observation_dim)
        actions = jax.random.randint(key, (), minval=0, maxval=self.n_actions, dtype=jnp.int8)
        _, key_ = jax.random.split(key)
        rewards = jax.random.uniform(key_)
        terminals = jax.random.randint(key_, (), 0, 2)
        next_states = jax.random.uniform(key_, self.observation_dim)
        return (
            jnp.array(states, dtype=jnp.float32),  # state
            jnp.array(actions, dtype=jnp.int8),  # action
            jnp.array(rewards, dtype=jnp.float32),  # reward
            jnp.array(next_states, dtype=jnp.float32),  # next_state
            jnp.ones(1),  # next_action
            jnp.ones(1),  # next_reward
            jnp.array(terminals, dtype=jnp.bool_),  # terminal
            jnp.ones(1),  # indices
        )

    @partial(jax.jit, static_argnames="self")
    def samples(self, key: jax.random.PRNGKey) -> Tuple[jnp.ndarray]:
        return jax.vmap(self.generate_sample)(jax.random.split(key, self.batch_size))

    @partial(jax.jit, static_argnames="self")
    def state(self, key: jax.random.PRNGKey) -> jnp.ndarray:
        return jax.random.uniform(key, self.observation_dim)

    @partial(jax.jit, static_argnames="self")
    def states(self, key: jax.random.PRNGKey) -> jnp.ndarray:
        return jax.random.uniform(key, (self.batch_size,) + self.observation_dim)
