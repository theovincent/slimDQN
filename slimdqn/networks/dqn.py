from functools import partial
from typing import Dict

import time
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core import FrozenDict

from slimdqn.networks.architectures.dqn import DQNNet
from slimdqn.sample_collection.replay_buffer import ReplayBuffer
from slimdqn.sample_collection.replay_buffer import ReplayElement


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
        loss_type: str = "huber",
        adam_eps: float = 1e-8,
    ):
        self.q_key = q_key
        self.q_network = DQNNet(features, cnn, n_actions)
        self.params = self.q_network.init(self.q_key, jnp.zeros(observation_dim, dtype=jnp.float32))

        self.state = np.zeros(observation_dim)
        # self.target_params = self.params.copy()

        self.optimizer = optax.adam(learning_rate, eps=adam_eps)
        self.optimizer_state = self.optimizer.init(self.params)
        self.target_params = self.params

        self.gamma = gamma
        self.update_horizon = update_horizon
        self.update_to_data = update_to_data
        self.target_update_frequency = target_update_frequency
        self.loss_type = loss_type

    def update_online_params(self, step: int, replay_buffer: ReplayBuffer):
        if step % self.update_to_data == 0:

            time_start_sample = time.time()
            batch_samples = replay_buffer.sample()
            jax.block_until_ready(batch_samples)
            time_sample = time.time() - time_start_sample

            time_start_grad = time.time()
            self.params, self.optimizer_state, loss = self.learn_on_batch(
                self.params, self.target_params, self.optimizer_state, batch_samples
            )
            jax.block_until_ready(loss)
            time_grad = time.time() - time_start_grad

            return loss, time_sample, time_grad
        return 0, 0, 0

    def update_target_params(self, step: int):
        if step % self.target_update_frequency == 0:
            self.target_params = self.params
            return True
        return False

    @partial(jax.jit, static_argnames="self")
    def learn_on_batch(
        self,
        params: FrozenDict,
        params_target: FrozenDict,
        optimizer_state,
        batch_samples,
    ):
        loss, grad_loss = jax.value_and_grad(self.loss_on_batch)(params, params_target, batch_samples)
        updates, optimizer_state = self.optimizer.update(grad_loss, optimizer_state)
        params = optax.apply_updates(params, updates)

        return params, optimizer_state, loss

    def loss_on_batch(self, params: FrozenDict, params_target: FrozenDict, samples):
        return jax.vmap(self.loss, in_axes=(None, None, 0))(params, params_target, samples).mean()

    def loss(self, params: FrozenDict, params_target: FrozenDict, sample: ReplayElement):
        # computes the loss for a single sample
        target = self.compute_target(params_target, sample)
        q_value = self.q_network.apply(params, sample.state)[sample.action]
        return jnp.square(q_value - target)

    def compute_target(self, params: FrozenDict, sample: ReplayElement):
        # computes the target value for single sample
        return sample.reward + (1 - sample.is_terminal) * (self.gamma**self.update_horizon) * jnp.max(
            self.q_network.apply(params, sample.next_state)
        )

    @partial(jax.jit, static_argnames="self")
    def best_action(self, params: FrozenDict, state: jnp.ndarray):
        # computes the best action for a single state
        return jnp.argmax(self.q_network.apply(params, state))

    def get_model(self) -> Dict:
        return {"params": self.params}
