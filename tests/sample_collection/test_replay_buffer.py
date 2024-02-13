import unittest
import numpy as np
import torch

from slimRL.sample_collection.replay_buffer import ReplayBuffer


class TestReplayBuffer(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.random_seed = np.random.randint(1000)
        print(f"random seed {self.random_seed}")
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)

        self.buffer_size = np.random.choice([10000 * i for i in range(1, 11)])
        self.batch_size = np.random.choice([2 ** i for i in range(0, 10)])
        self.update_horizon = np.random.choice([i for i in range(1, 11)])
        self.rb = ReplayBuffer(observation_shape=(2, 2),
                               replay_capacity=self.buffer_size,
                               batch_size=self.batch_size,
                               update_horizon=self.update_horizon,
                               gamma=0.99)

    def test_add(self):
        pass

    def test_sample_index_batch(self):
        pass
    def test_sample_transition_batch(self):
        pass
    def test_is_empty(self):
        pass
    def test_is_full(self):
        pass
    def test_get_range(self):
        pass
    def test_is_valid_transition(self):
        pass


