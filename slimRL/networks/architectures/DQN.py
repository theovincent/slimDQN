import optax
import jax
import jax.numpy as jnp
import flax.linen as nn
from slimRL.networks.DQN import DQN


class DQNNet(nn.Module):
    env: object
    hidden_layers: list

    def setup(self):
        self.initializer = nn.initializers.variance_scaling(
            scale=1.0, mode="fan_avg", distribution="truncated_normal"
        )
        layers = []

        for hidden_size in self.hidden_layers:
            layers.append(nn.Dense(hidden_size, kernel_init=self.initializer))
            layers.append(nn.relu)

        layers.append(nn.Dense(self.env.n_actions, kernel_init=self.initializer))

        self.network = nn.Sequential(layers)
        # print(self.network)

    @nn.compact
    def __call__(self, state):
        return self.network(state)


class BasicDQN(DQN):
    def __init__(
        self,
        q_key: jax.random.PRNGKey,
        env,
        hidden_layers: list,
        gamma: float,
        update_horizon: int,
        lr: float,
        adam_eps: float,
        train_frequency: int,
        target_update_frequency: int,
        loss_type: str = "huber",
    ):
        self.env = env
        self.lr = lr
        optimizer = optax.adam(self.lr, eps=adam_eps)
        q_network = DQNNet(env, hidden_layers)
        q_inputs = {
            "state": jnp.zeros(
                jnp.array(self.env.observation_shape).prod(), dtype=jnp.float32
            )
        }
        super().__init__(
            q_key,
            q_inputs,
            gamma,
            update_horizon,
            q_network,
            optimizer,
            loss_type,
            train_frequency,
            target_update_frequency,
        )
