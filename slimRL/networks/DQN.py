import optax
import jax
from flax.core import FrozenDict
from functools import partial
import flax.linen as nn
import jax.numpy as jnp
from slimRL.sample_collection.replay_buffer import ReplayBuffer


class DQN:
    def __init__(
        self,
        q_key: jax.random.PRNGKey,
        q_inputs: dict,
        gamma: float,
        update_horizon: int,
        q_network: nn.Module,
        optimizer,
        loss_type: str,
        train_frequency: int,
        target_update_frequency: int,
    ):
        self.q_key = q_key
        self.gamma = gamma
        self.update_horizon = update_horizon
        self.q_network = q_network
        self.params = self.q_network.init(self.q_key, **q_inputs)
        self.target_params = self.params.copy()
        self.optimizer = optimizer
        self.optimizer_state = self.optimizer.init(self.params)
        self.loss_type = loss_type
        self.train_frequency = train_frequency
        self.target_update_frequency = target_update_frequency

    def loss(
        self,
        params: FrozenDict,
        params_target: FrozenDict,
        sample,
    ):  # computes the loss for a single sample
        target = self.compute_target(params_target, sample)
        q_value = self.apply(params, sample["observations"])[sample["actions"]]
        return self.metric(q_value - target, ord=self.loss_type)

    def loss_on_batch(self, params: FrozenDict, params_target: FrozenDict, samples):
        return jax.vmap(self.loss, in_axes=(None, None, 0))(
            params, params_target, samples
        ).mean()

    @staticmethod
    def metric(error: jnp.ndarray, ord: str):
        if ord == "huber":
            return optax.huber_loss(error, 0)
        elif ord == "1":
            return jnp.abs(error)
        elif ord == "2":
            return jnp.square(error)

    @partial(jax.jit, static_argnames="self")
    def learn_on_batch(
        self,
        params: FrozenDict,
        params_target: FrozenDict,
        optimizer_state,
        batch_samples,
    ):
        loss, grad_loss = jax.value_and_grad(self.loss_on_batch)(
            params, params_target, batch_samples
        )
        updates, optimizer_state = self.optimizer.update(grad_loss, optimizer_state)
        params = optax.apply_updates(params, updates)

        return params, optimizer_state, loss

    @partial(jax.jit, static_argnames="self")
    def apply(
        self, params: FrozenDict, states: jnp.ndarray
    ):  # computes the q values for single or batch of states
        return self.q_network.apply(params, states)

    @partial(jax.jit, static_argnames="self")
    def compute_target(
        self, params: FrozenDict, samples
    ):  # computes the target value for single or a batch of samples
        return samples["rewards"] + (1 - samples["dones"]) * self.gamma * jnp.max(
            self.apply(params, samples["next_observations"]), axis=-1
        )

    def update_online_params(
        self, step: int, batch_size: int, replay_buffer: ReplayBuffer
    ):
        if step % self.train_frequency == 0:
            self.q_key, batching_key = jax.random.split(self.q_key)
            batch_samples = replay_buffer.sample_transition_batch(
                batch_size, batching_key
            )

            self.params, self.optimizer_state, loss = self.learn_on_batch(
                self.params,
                self.target_params,
                self.optimizer_state,
                batch_samples,
            )

            return loss
        return jnp.nan

    def update_target_params(self, step: int):
        if step % self.target_update_frequency == 0:
            self.target_params = self.params.copy()

    @partial(jax.jit, static_argnames="self")
    def best_action(
        self, params: FrozenDict, state: jnp.ndarray
    ):  # computes the best action for a single state
        return jnp.argmax(self.apply(params, state)).astype(jnp.int8)
