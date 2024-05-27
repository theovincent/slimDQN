import numpy as np
import torch.nn as nn
import torch.optim as optim
from slimRL.networks.DQN import DQN


class DQNNet(nn.Module):
    def __init__(self, env, hidden_layers: list):
        super().__init__()
        self.env = env
        layers = []
        input_size = np.array(self.env.observation_shape).prod()

        for hidden_size in hidden_layers:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size

        layers.append(nn.Linear(input_size, env.n_actions))

        self.network = nn.Sequential(*layers)
        print(self.network)

    def forward(self, x):
        return self.network(x)


class BasicDQN(DQN):
    def __init__(
        self,
        env,
        device,
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
        self.device = device
        self.lr = lr
        self.adam_eps = adam_eps
        q_network = DQNNet(env, hidden_layers).to(self.device)
        optimizer = optim.Adam(q_network.parameters(), lr=self.lr, eps=self.adam_eps)
        target_network = DQNNet(env, hidden_layers).to(self.device)
        target_network.load_state_dict(q_network.state_dict())

        super().__init__(
            gamma,
            update_horizon,
            target_network,
            q_network,
            optimizer,
            loss_type,
            train_frequency,
            target_update_frequency,
        )
