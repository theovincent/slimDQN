import torch.nn as nn
import torch.optim as optim
import numpy as np
from ..dqn import DQN


class DQNNet(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.network = nn.Sequential(
            nn.Linear(np.array(self.env.observation_shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, env.action_dim),
        )

    def forward(self, x):
        return self.network(x)


class BasicDQN(DQN):
    def __init__(
        self,
        env,
        device,
        gamma: float,
        tau: float,
        lr: float,
        train_frequency: int,
        target_update_frequency: int,
        loss_type: str = "huber",
    ):
        self.env = env
        self.device = device
        self.lr = lr
        q_network = DQNNet(env).to(self.device)
        optimizer = optim.Adam(q_network.parameters(), lr=self.lr)
        target_network = DQNNet(env).to(self.device)
        target_network.load_state_dict(q_network.state_dict())

        super().__init__(
            gamma,
            tau,
            target_network,
            q_network,
            optimizer,
            loss_type,
            train_frequency,
            target_update_frequency,
        )
