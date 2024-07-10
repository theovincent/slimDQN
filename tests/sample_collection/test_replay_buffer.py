import unittest
import numpy as np
import jax

from slimRL.sample_collection.replay_buffer import ReplayBuffer


class TestReplayBuffer(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.random_seed = np.random.randint(1000)
        self.key = jax.random.PRNGKey(self.random_seed)

        self.buffer_size = np.random.choice([10000 * i for i in range(1, 11)])
        self.batch_size = np.random.choice([2**i for i in range(0, 10)])
        self.update_horizon = np.random.choice([i for i in range(1, 11)])
        self.rb = ReplayBuffer(
            observation_shape=(2, 2),
            replay_capacity=self.buffer_size,
            update_horizon=self.update_horizon,
        )

    def test_add(self):
        for i in range(50):
            self.rb.add(
                observation=np.random.normal((2, 2)),
                action=np.random.randint(0, 5),
                reward=np.random.normal(),
                terminal=np.random.choice([True, False], p=[0.2, 0.8]),
            )
        self.assertEqual(self.rb.add_count, 50)
        self.assertEqual(self.rb.is_full(), False)
        for i in range(self.rb._replay_capacity):
            self.rb.add(
                observation=np.random.normal((2, 2)),
                action=np.random.randint(0, 5),
                reward=np.random.normal(),
                terminal=np.random.choice([True, False], p=[0.2, 0.8]),
            )
        self.assertEqual(self.rb.is_full(), True)

    def test_sample(self):
        for i in range(self.rb._replay_capacity):
            self.rb.add(
                observation=np.random.normal((2, 2)),
                action=np.random.randint(0, 5),
                reward=np.random.normal(),
                terminal=np.random.choice([True, False], p=[0.8, 0.2]),
            )
        batch_size = np.random.randint(1, 1000)
        self.assertEqual(len(self.rb.sample_index_batch(batch_size, self.key)), batch_size)
        self.assertEqual(len(self.rb.sample_transition_batch(batch_size, self.key)), 5)
        self.assertEqual(
            len(self.rb.sample_transition_batch(batch_size, self.key)["observations"]),
            batch_size,
        )
        for i in range(self.rb._replay_capacity):
            self.rb.add(
                observation=np.random.normal((2, 2)),
                action=np.random.randint(0, 5),
                reward=np.random.normal(),
                terminal=False,
                episode_end=True,
            )
        self.assertRaises(
            RuntimeError,
            self.rb.sample_index_batch,
            batch_size=1,
            batching_key=self.key,
        )

    def test_horizon_based_buffer(self):
        self.small_rb = ReplayBuffer(observation_shape=(2,), replay_capacity=6, update_horizon=3)
        self.assertEqual(self.small_rb.add_count, 0)
        self.small_rb.add(observation=[1, 1], action=5, reward=-1.0, terminal=False)
        self.small_rb.add(observation=[1, 2], action=1, reward=-1.0, terminal=False)
        self.small_rb.add(observation=[1, 3], action=1, reward=-1.0, terminal=False)
        self.small_rb.add(
            observation=[1, 4],
            action=0,
            reward=-1.0,
            terminal=False,
            episode_end=True,
        )
        self.assertEqual(self.small_rb.cursor(), 4)
        self.assertEqual(self.small_rb.is_full(), False)
        self.assertEqual(self.small_rb.is_valid_transition(0), True)
        self.assertListEqual(
            self.small_rb.sample_transition_batch(1, self.key)["observations"][0].tolist(),
            [1, 1],
        )
        self.small_rb.add(observation=[2, 1], action=1, reward=-1.0, terminal=True)
        self.small_rb.add(observation=[3, 1], action=0, reward=-1.0, terminal=False)
        self.small_rb.add(observation=[3, 2], action=0, reward=-1.0, terminal=True)
        self.assertEqual(self.small_rb.cursor(), 1)
        self.assertEqual(len(self.small_rb.sample_transition_batch(3, self.key)["observations"]), 3)
        self.small_rb.add(observation=[4, 1], action=0, reward=-1.0, terminal=False)
        self.small_rb.add(observation=[4, 2], action=0, reward=-1.0, terminal=False)
        self.assertEqual(self.small_rb.is_valid_transition(0), True)
        self.assertEqual(self.small_rb.is_valid_transition(1), False)
        self.assertEqual(self.small_rb.is_valid_transition(2), False)
        self.assertEqual(self.small_rb.is_valid_transition(3), False)
        self.assertEqual(self.small_rb.is_valid_transition(4), True)
        self.assertEqual(self.small_rb.is_valid_transition(5), True)
        self.small_rb.add(observation=[4, 3], action=0, reward=-1.0, terminal=True)
        self.assertEqual(self.small_rb.is_valid_transition(1), True)
        self.assertEqual(self.small_rb.is_valid_transition(2), True)
        self.assertEqual(self.small_rb.is_valid_transition(3), True)
