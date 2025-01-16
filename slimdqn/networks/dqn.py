from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax.core import FrozenDict

from slimdqn.networks.architectures.dqn import DQNNet
from slimdqn.sample_collection import IDX_RB
from slimdqn.sample_collection.replay_buffer import ReplayBuffer


class DQN:
    def __init__(
        self,
        q_key: jax.random.PRNGKey,
        observation_dim,
        n_actions,
        features: list,
        cnn: bool,
        learning_rate: float,
        gamma: float,
        update_horizon: int,
        update_to_data: int,
        target_update_frequency: int,
        adam_eps: float = 1e-8,
    ):
        self.q_key = q_key
        self.network = DQNNet(features, cnn, n_actions)
        self.params = self.network.init(self.q_key, jnp.zeros(observation_dim, dtype=jnp.float32))
        self.target_params = self.params.copy()

        self.optimizer = optax.adam(learning_rate, eps=adam_eps)
        self.optimizer_state = self.optimizer.init(self.params)

        self.gamma = gamma
        self.update_horizon = update_horizon
        self.update_to_data = update_to_data
        self.target_update_frequency = target_update_frequency
        self.cumulated_loss = 0

    def update_online_params(self, step: int, replay_buffer: ReplayBuffer):
        if step % self.update_to_data == 0:
            batch_samples = replay_buffer.sample_transition_batch()

            self.params, self.optimizer_state, loss = self.learn_on_batch(
                self.params, self.target_params, self.optimizer_state, batch_samples
            )
            self.cumulated_loss += loss

    def update_target_params(self, step: int):
        if step % self.target_update_frequency == 0:
            self.target_params = self.params.copy()

            logs = {"loss": self.cumulated_loss / (self.target_update_frequency / self.update_to_data)}
            self.cumulated_loss = 0

            return True, logs
        return False, {}

    @partial(jax.jit, static_argnames="self")
    def learn_on_batch(self, params: FrozenDict, params_target: FrozenDict, optimizer_state, batch_samples):
        loss, grad_loss = jax.value_and_grad(self.loss_on_batch)(params, params_target, batch_samples)
        updates, optimizer_state = self.optimizer.update(grad_loss, optimizer_state)
        params = optax.apply_updates(params, updates)

        return params, optimizer_state, loss

    def loss_on_batch(self, params: FrozenDict, params_target: FrozenDict, samples):
        return jax.vmap(self.loss, in_axes=(None, None, 0))(params, params_target, samples).mean()

    def loss(self, params: FrozenDict, params_target: FrozenDict, sample):
        # computes the loss for a single sample
        target = self.compute_target(params_target, sample)
        q_value = self.network.apply(params, sample[IDX_RB["state"]])[sample[IDX_RB["action"]]]
        return jnp.square(q_value - target)

    def compute_target(self, params: FrozenDict, samples):
        # computes the target value for single sample
        return samples[IDX_RB["reward"]] + (1 - samples[IDX_RB["terminal"]]) * (
            self.gamma**self.update_horizon
        ) * jnp.max(self.network.apply(params, samples[IDX_RB["next_state"]]))

    @partial(jax.jit, static_argnames="self")
    def best_action(self, params: FrozenDict, state: jnp.ndarray):
        # computes the best action for a single state
        return jnp.argmax(self.network.apply(params, state)).astype(jnp.int8)
