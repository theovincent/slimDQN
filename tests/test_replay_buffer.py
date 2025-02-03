# Inspired by dopamine implementation: https://github.com/google/dopamine/blob/master/tests/dopamine/jax/replay_memory/replay_buffer_test.py

from absl.testing import absltest
from absl.testing import parameterized
from absl import flags
from etils import epath
import msgpack
import numpy as np
import jax

from slimdqn.sample_collection import replay_buffer
from slimdqn.sample_collection.replay_buffer import ReplayElement, TransitionElement
from slimdqn.sample_collection import samplers


# Default parameters used when creating the replay memory - mimic Atari.
OBSERVATION_SHAPE = (84, 84)
STACK_SIZE = 4
BATCH_SIZE = 32

flags.FLAGS(["--test_tmpdir", "/tmpdir"])


class ReplayBufferTest(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        self._tmpdir = epath.Path(self.create_tempdir("checkpoint").full_path)
        self._obs = np.ones((4, 3))

        self._sampling_distribution = samplers.UniformSamplingDistribution(seed=0)

    def test_element_pack_unpack(self) -> None:
        """Simple test case that packs and unpacks a replay element."""
        state = np.zeros(OBSERVATION_SHAPE + (STACK_SIZE,), dtype=np.uint8)
        next_state = np.ones(OBSERVATION_SHAPE + (STACK_SIZE,), dtype=np.uint8)
        action = 1
        reward = 1.0
        episode_end = False

        element = ReplayElement(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            is_terminal=episode_end,
            episode_end=episode_end,
        )

        packed = element.pack()
        assert packed.action == action
        assert packed.reward == reward
        assert packed.is_terminal == packed.episode_end == episode_end

        unpacked = packed.unpack()
        assert unpacked.action == action
        assert unpacked.reward == reward
        assert unpacked.is_terminal == unpacked.episode_end == episode_end

        np.testing.assert_array_equal(unpacked.state, state)
        np.testing.assert_array_equal(unpacked.next_state, next_state)

    def testAddUpToCapacity(self):
        capacity = 10
        rb = replay_buffer.ReplayBuffer(
            sampling_distribution=samplers.UniformSamplingDistribution(seed=0),
            batch_size=BATCH_SIZE,
            max_capacity=capacity,
            stack_size=STACK_SIZE,
            update_horizon=1,
            gamma=1.0,
            compress=False,
        )

        transitions = []
        for i in range(16):
            transitions.append(TransitionElement(np.full(OBSERVATION_SHAPE, i), i, i, False, False))
            rb.add(transitions[-1])
        # Since we created the ReplayBuffer with a capacity of 10, it should have
        # gotten rid of the first 5 elements added.
        self.assertLen(rb._memory, capacity)
        expected_keys = list(range(5, 5 + capacity))
        self.assertEqual(list(rb._memory.keys()), expected_keys)
        for i in expected_keys:
            np.testing.assert_array_equal(
                rb._memory[i].state,
                np.array([transition.observation for transition in transitions[i - STACK_SIZE + 1 : i + 1]]).transpose(
                    1, 2, 0
                ),
            )
            np.testing.assert_array_equal(
                rb._memory[i].next_state,
                np.array([transition.observation for transition in transitions[i - STACK_SIZE + 2 : i + 2]]).transpose(
                    1, 2, 0
                ),
            )
            self.assertEqual(rb._memory[i].action, transitions[i].action)
            self.assertEqual(rb._memory[i].reward, transitions[i].reward)
            self.assertEqual(rb._memory[i].is_terminal, int(transitions[i].is_terminal))
            self.assertEqual(rb._memory[i].episode_end, int(transitions[i].episode_end))

    def testNSteprewards(self):
        rb = replay_buffer.ReplayBuffer(
            sampling_distribution=samplers.UniformSamplingDistribution(seed=0),
            batch_size=BATCH_SIZE,
            max_capacity=10,
            stack_size=STACK_SIZE,
            update_horizon=5,
            gamma=1.0,
            compress=False,
        )

        for i in range(50):
            # add non-terminating observations with reward 2
            rb.add(TransitionElement(np.full(OBSERVATION_SHAPE, i), 0, 2.0, False))

        for _ in range(100):
            batch = rb.sample()
            # Make sure the total reward is reward per step x update_horizon.
            np.testing.assert_array_equal(batch.reward, np.ones(BATCH_SIZE) * 10.0)

    def testGetStack(self):
        zero_state = np.zeros(OBSERVATION_SHAPE + (3,))

        rb = replay_buffer.ReplayBuffer(
            sampling_distribution=samplers.UniformSamplingDistribution(seed=0),
            batch_size=BATCH_SIZE,
            max_capacity=50,
            stack_size=STACK_SIZE,
            update_horizon=5,
            gamma=1.0,
            compress=False,
        )
        for i in range(11):
            rb.add(TransitionElement(np.full(OBSERVATION_SHAPE, i), 0, 0, False))

        # ensure that the returned shapes are always correct
        for i in rb._memory:
            np.testing.assert_array_equal(rb._memory[i].state.shape, OBSERVATION_SHAPE + (4,))

        # ensure that there is the necessary 0 padding
        state = rb._memory[0].state
        np.testing.assert_array_equal(zero_state, state[:, :, :3])

        # ensure that after the padding the contents are properly stored
        state = rb._memory[3].state
        for i in range(4):
            np.testing.assert_array_equal(np.full(OBSERVATION_SHAPE, i), state[:, :, i])

    def testSampleTransitionBatch(self):
        rb = replay_buffer.ReplayBuffer(
            sampling_distribution=samplers.UniformSamplingDistribution(seed=0),
            batch_size=2,
            max_capacity=10,
            stack_size=1,
            update_horizon=1,
            gamma=0.99,
            compress=False,
        )
        num_adds = 50  # The number of transitions to add to the memory.

        # terminal transitions are not valid trajectories
        index_to_id = []
        for i in range(num_adds):
            terminal = i % 4 == 0  # Every 4 transitions is terminal.
            rb.add(TransitionElement(np.full(OBSERVATION_SHAPE, i), 0, 0, terminal, False))
            if not terminal:
                index_to_id.append(i)

        # Verify we sample the expected indices by using the same rng state.
        self._rng_key = np.random.default_rng(seed=0)
        indices = self._rng_key.integers(
            len(rb._sampling_distribution._index_to_key), size=len(rb._sampling_distribution._index_to_key)
        )

        def make_state(key: int):
            return np.full(OBSERVATION_SHAPE + (1,), key)

        expected_states = np.array(
            [make_state(index_to_id[rb._sampling_distribution._index_to_key[i]]) for i in indices]
        )
        expected_next_states = np.array(
            [make_state(index_to_id[rb._sampling_distribution._index_to_key[i]] + 1) for i in indices]
        )

        # Replicating the formula used above to determine what transitions are terminal
        expected_terminal = np.array(
            [int(((index_to_id[rb._sampling_distribution._index_to_key[i]] + 1) % 4) == 0) for i in indices]
        )
        batch = rb.sample(size=len(indices))
        np.testing.assert_array_equal(batch.state, expected_states)
        np.testing.assert_array_equal(batch.action, np.zeros(len(indices)))
        np.testing.assert_array_equal(batch.reward, np.zeros(len(indices)))
        np.testing.assert_array_equal(batch.next_state, expected_next_states)
        np.testing.assert_array_equal(batch.is_terminal, expected_terminal)

    def testSamplingWithTerminalInTrajectory(self):
        rb = replay_buffer.ReplayBuffer(
            sampling_distribution=samplers.UniformSamplingDistribution(seed=0),
            batch_size=2,
            max_capacity=10,
            stack_size=1,
            update_horizon=3,
            gamma=1.0,
            compress=False,
        )
        for i in range(rb._max_capacity):
            rb.add(
                TransitionElement(
                    np.full(OBSERVATION_SHAPE, i), action=i * 2, reward=i, is_terminal=i == 3, episode_end=False
                )
            )
        # Verify we sample the expected indices, using the same rng.
        self._rng_key = np.random.default_rng(seed=0)
        indices = self._rng_key.integers(rb.add_count, size=5)

        batch = rb.sample(size=5)

        # Since index 3 is terminal, it will not be a valid transition so renumber.
        expected_states = np.array(
            [
                np.full(OBSERVATION_SHAPE + (1,), i) if i < 3 else np.full(OBSERVATION_SHAPE + (1,), i + 1)
                for i in indices
            ]
        )
        expected_actions = np.array([i * 2 if i < 3 else (i + 1) * 2 for i in indices])
        # The reward in the replay buffer will be (an asterisk marks the terminal
        # state):
        #   [0 1 2 3* 4 5 6 7 8 9]
        # Since we're setting the update_horizon to 3, the accumulated trajectory
        # reward starting at each of the replay buffer positions will be (a '_'
        # marks an invalid transition to sample):
        #   [3 6 5 _ 15 18 21 24]
        expected_rewards = np.array([3, 6, 5, 15, 18, 21, 24])
        # Because update_horizon = 3, indices 0, 1 and 2 include terminal.
        expected_terminals = np.array([1, 1, 1, 0, 0, 0, 0])
        np.testing.assert_array_equal(batch.state, expected_states)
        np.testing.assert_array_equal(batch.action, expected_actions)
        np.testing.assert_array_equal(batch.reward, expected_rewards[indices])
        np.testing.assert_array_equal(batch.is_terminal, expected_terminals[indices])

    def testKeyMappingsForSampling(self):
        capacity = 10
        rb = replay_buffer.ReplayBuffer(
            sampling_distribution=samplers.UniformSamplingDistribution(seed=0),
            batch_size=BATCH_SIZE,
            max_capacity=capacity,
            stack_size=1,
            update_horizon=1,
            gamma=0.99,
            compress=False,
        )
        sampler = rb._sampling_distribution

        for i in range(capacity + 1):
            rb.add(TransitionElement(np.full(OBSERVATION_SHAPE, i), i, i, False, False))

        # While we haven't overwritten any elements we should have
        # global indices as being equivalent to local indices
        for i in range(capacity):
            self.assertIn(i, sampler._key_to_index)
            index = sampler._key_to_index[i]
            self.assertEqual(i, index)
            self.assertEqual(i, sampler._index_to_key[index])

        # The next key to be inserted will be `capacity` as when we add
        # `capacity + 1` the accumulator will insert: (capacity, capacity + 1)
        next_key = capacity
        rb.add(
            TransitionElement(
                np.full(OBSERVATION_SHAPE, next_key + 1),
                next_key + 1,
                next_key + 1,
                False,
                False,
            )
        )
        # We should have deleted the earliest index
        self.assertNotIn(0, sampler._key_to_index)
        # The local index corresponding to the previous key should have been swapped
        self.assertNotEqual(sampler._index_to_key[0], 0)
        # We should have inserted the new key into key -> index
        self.assertIn(next_key, sampler._key_to_index)
        # index -> key should be consistent
        self.assertEqual(next_key, sampler._index_to_key[sampler._key_to_index[next_key]])

        self._rng_key = np.random.default_rng(seed=0)
        indices = self._rng_key.integers(len(sampler._index_to_key), size=BATCH_SIZE)

        # Convert local indices to global keys
        keys = (sampler._index_to_key[index] for index in indices)

        # Fetch actual samples from the replay buffer so we can compare
        # the global indices
        samples = rb.sample()

        # Each index in our samples should have observations that are equal to
        # their global key, we can check this:
        for i, key in enumerate(keys):
            np.testing.assert_array_equal(
                samples.state[i, ...],
                np.full(OBSERVATION_SHAPE, key)[..., None],
            )
            np.testing.assert_array_equal(
                samples.next_state[i, ...],
                np.full(OBSERVATION_SHAPE, key + 1)[..., None],
            )
            self.assertEqual(samples.action[i], key)
            self.assertEqual(samples.reward[i], key)
            self.assertEqual(samples.is_terminal[i], 0)
            self.assertEqual(samples.episode_end[i], 0)

    @parameterized.parameters((True,), (False,))
    def testSave(self, compress):
        stack_size = 4
        replay_capacity = 50
        batch_size = 32
        update_horizon = 3
        gamma = 0.9
        checkpoint_duration = 7
        replay = replay_buffer.ReplayBuffer(
            sampling_distribution=self._sampling_distribution,
            batch_size=batch_size,
            max_capacity=replay_capacity,
            stack_size=stack_size,
            update_horizon=update_horizon,
            gamma=gamma,
            checkpoint_duration=checkpoint_duration,
            compress=compress,
        )
        # Store a few transitions in memory. Since update_horizon is 3, only
        # num_adds - 3 elements will actually be in the replay buffer memory.
        transitions = []
        num_adds = 15
        for i in range(num_adds):
            transitions.append(TransitionElement(self._obs * i, i, i, False, False))
            replay.add(transitions[-1])

        replay.save(self._tmpdir, 1)
        path = self._tmpdir / "1" / "replay" / "checkpoint.msgpack"
        self.assertTrue(path.exists())
        replay_pack = msgpack.unpackb(
            path.read_bytes(),
            raw=False,
            strict_map_key=False,
        )
        self.assertEqual(num_adds - update_horizon, replay_pack["add_count"])


if __name__ == "__main__":
    absltest.main()
