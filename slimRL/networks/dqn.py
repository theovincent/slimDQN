import torch
import torch.nn as nn
import torch.optim as optim
from slimRL.sample_collection.replay_buffer import ReplayBuffer


class DQN:
    def __init__(
            self,
            gamma: float,
            tau: float,
            target_network: nn.Module,
            q_network: nn.Module,
            optimizer: optim.Optimizer,
            loss_type: str,
            train_frequency: int,
            target_update_frequency: int,
            save_model: bool,
    ):

        self.gamma = gamma
        self.tau = tau
        self.target_network = target_network
        self.q_network = q_network
        self.optimizer = optimizer
        self.loss_type = loss_type
        self.train_frequency = train_frequency
        self.target_update_frequency = target_update_frequency
        self.save_model = save_model

    def loss_on_batch(self, td_target, estimate):
        return self.loss(self.loss_type)(td_target, estimate)

    @staticmethod
    def loss(order) -> nn.modules.loss:
        if order == "huber":
            return nn.HuberLoss()
        elif order == "1":
            return nn.L1Loss()
        elif order == "2":
            return nn.MSELoss()

    def learn_on_batch(
            self,
            batch_samples
    ):
        target = self.compute_target(batch_samples)
        curr_estimate = self.compute_curr_estimate(batch_samples)
        loss = self.loss_on_batch(target, curr_estimate)

        # print(self.optimizer)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def compute_target(self, data):
        with torch.no_grad():
            target_max, _ = self.compute_qval(self.target_network, data['next_observations']).max(dim=1)
            td_target = data['rewards'].flatten() + self.gamma * target_max * (1 - data['dones'].flatten())
        return td_target

    def compute_curr_estimate(self, data):
        q_val = self.compute_qval(self.q_network, data['observations']).gather(1, data['actions'][:, None]).squeeze()
        # print(q_val)
        return q_val

    @staticmethod
    def compute_qval(network, states):
        q_val = network(states)
        return q_val

    def update_online_params(self, step: int, replay_buffer: ReplayBuffer):
        if step % self.train_frequency == 0:
            batch_samples = replay_buffer.sample_transition_batch()

            loss = self.learn_on_batch(
                batch_samples
            )
            return loss
        return None

    def update_target_params(self, step: int):
        if step % self.target_update_frequency == 0:
            for target_network_param, q_network_param in zip(self.target_network.parameters(),
                                                             self.q_network.parameters()):
                target_network_param.data.copy_(
                    self.tau * q_network_param.data + (1.0 - self.tau) * target_network_param.data
                )

    def best_action(self,
                    state: torch.tensor):
        with torch.no_grad():
            # print(torch.argmax(self.compute_qval(self.q_network, torch.Tensor(state))))
            action = torch.argmax(self.compute_qval(self.q_network, torch.Tensor(state))).cpu().numpy()
        return action
