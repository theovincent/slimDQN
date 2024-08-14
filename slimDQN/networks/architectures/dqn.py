from typing import Sequence

import flax.linen as nn
import jax.numpy as jnp


class DQNNet(nn.Module):
    features: Sequence[int]
    n_actions: int

    @nn.compact
    def __call__(self, x):
        x = jnp.squeeze(x)
        for feature in self.features:
            x = nn.relu(nn.Dense(feature)(x))
        x = nn.Dense(self.n_actions)(x)
        return x
