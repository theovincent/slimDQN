import torch as nn
import torch.optim as optim
import numpy as np
from ..dqn import DQN
from ...environments.environment import Environment


class DQNNet(nn.Module):
    env: Environment

    def setup(self):
        self.network = nn.Sequential(
            nn.Linear(np.array(self.env.state_shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, self.env.action_dim),
        )

    def __call__(self, state):
        return self.network(state)


class BasicDQN(DQN):
    def __init__(
            self,
            env,
            device,
            gamma: float,
            tau: float,
            seed: int,
            lr: float,
            loss_type: str,
            train_frequency: int,
            target_update_frequency: int,
            save_model: bool,
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
            save_model,
        )


