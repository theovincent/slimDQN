from typing import Sequence
import flax.linen as nn


class DQNNet(nn.Module):
    n_actions: int
    hidden_layers: Sequence[int]

    @nn.compact
    def __call__(self, x):
        for feature in self.hidden_layers:
            x = nn.relu(nn.Dense(feature)(x))
        x = nn.Dense(self.n_actions)(x)
        return x
