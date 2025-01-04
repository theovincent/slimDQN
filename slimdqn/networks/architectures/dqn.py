from typing import Sequence

import flax.linen as nn
import jax.numpy as jnp


class DQNNet(nn.Module):
    features: Sequence[int]
    cnn: bool
    n_actions: int

    @nn.compact
    def __call__(self, x):
        if self.cnn:
            initializer = nn.initializers.variance_scaling(scale=1.0, mode="fan_avg", distribution="truncated_normal")
            x = nn.relu(
                nn.Conv(features=self.features[0], kernel_size=(8, 8), strides=(4, 4), kernel_init=initializer)(
                    jnp.array(x, ndmin=4) / 255.0
                )
            )
            x = nn.relu(
                nn.Conv(features=self.features[1], kernel_size=(4, 4), strides=(2, 2), kernel_init=initializer)(x)
            )
            x = nn.relu(
                nn.Conv(features=self.features[2], kernel_size=(3, 3), strides=(1, 1), kernel_init=initializer)(x)
            )
            x = x.reshape((x.shape[0], -1))
        else:
            initializer = nn.initializers.lecun_normal()

        x = jnp.squeeze(x)

        for idx_layer in range(3 if self.cnn else 0, len(self.features)):
            x = nn.relu((nn.Dense(self.features[idx_layer], kernel_init=initializer)(x)))

        return nn.Dense(self.n_actions, kernel_init=initializer)(x)


class DQNNetNature(nn.Module):
    n_actions: int

    @nn.compact
    def __call__(self, x):
        initializer = nn.initializers.variance_scaling(scale=1.0, mode="fan_avg", distribution="truncated_normal")
        x = nn.relu(
            nn.Conv(features=32, kernel_size=(8, 8), strides=(4, 4), kernel_init=initializer)(
                jnp.array(x, ndmin=4) / 255.0
            )
        )
        x = nn.relu(
            nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2), kernel_init=initializer)(x)
        )
        x = nn.relu(
            nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1), kernel_init=initializer)(x)
        )
        x = x.reshape((x.shape[0], -1))
        x = jnp.squeeze(x)

        x = nn.relu((nn.Dense(512, kernel_init=initializer)(x)))

        return nn.Dense(self.n_actions, kernel_init=initializer)(x)