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
        self.assertEqual(self.rb.is_empty(), True)
        for i in range(50):
            self.rb.add(observation=np.random.normal((2, 2)),
                        action=[np.random.randint(0, 5)],
                        reward=np.random.normal(),
                        terminal=np.random.choice([True, False], p=[0.2, 0.8]))
        self.assertEqual(self.rb.add_count, 50)
        self.assertEqual(self.rb.is_empty(), False)
        self.assertEqual(self.rb.is_full(), False)
        for i in range(self.rb._replay_capacity):
            self.rb.add(observation=np.random.normal((2, 2)),
                        action=[np.random.randint(0, 5)],
                        reward=np.random.normal(),
                        terminal=np.random.choice([True, False], p=[0.2, 0.8]))
        self.assertEqual(self.rb.is_full(), True)

    def test_sample(self):
        for i in range(self.rb._replay_capacity):
            self.rb.add(observation=np.random.normal((2, 2)),
                        action=[np.random.randint(0, 5)],
                        reward=np.random.normal(),
                        terminal=np.random.choice([True, False], p=[0.8, 0.2]))
        batch_size = np.random.randint(1, 1000)
        self.assertEqual(len(self.rb.sample_index_batch(batch_size)), batch_size)
        self.assertEqual(len(self.rb.sample_transition_batch(batch_size)), 5)
        self.assertEqual(len(self.rb.sample_transition_batch(batch_size)["observations"]), batch_size)
        for i in range(self.rb._replay_capacity):
            self.rb.add(observation=np.random.normal((2, 2)),
                        action=[np.random.randint(0, 5)],
                        reward=np.random.normal(),
                        terminal=False,
                        episode_end=True)
        self.assertRaises(RuntimeError, self.rb.sample_index_batch, batch_size=1)

    def test_horizon_based_buffer(self):
        pass
