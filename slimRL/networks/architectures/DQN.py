import flax.linen as nn


class DQNNet(nn.Module):
    n_actions: int
    hidden_layers: list

    def setup(self):
        self.initializer = nn.initializers.variance_scaling(scale=1.0, mode="fan_in", distribution="truncated_normal")
        layers = []

        for hidden_size in self.hidden_layers:
            layers.append(nn.Dense(hidden_size, kernel_init=self.initializer))
            layers.append(nn.relu)

        layers.append(nn.Dense(self.n_actions, kernel_init=self.initializer))

        self.network = nn.Sequential(layers)

    @nn.compact
    def __call__(self, state):
        return self.network(state)
