# Inspired by dopamine implementation: https://github.com/google/dopamine/blob/master/tests/dopamine/jax/replay_memory/replay_buffer_test.py

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import jax

from slimdqn.sample_collection import replay_buffer
from slimdqn.sample_collection.replay_buffer import ReplayElement, TransitionElement
from slimdqn.sample_collection import samplers


# Default parameters used when creating the replay memory - mimic Atari.
OBSERVATION_SHAPE = (84, 84)
STACK_SIZE = 4
BATCH_SIZE = 32


class ReplayBufferTest(parameterized.TestCase):

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
        )

        packed = element.pack()
        assert packed.action == action
        assert packed.reward == reward
        assert packed.is_terminal == episode_end

        unpacked = packed.unpack()
        assert unpacked.action == action
        assert unpacked.reward == reward
        assert unpacked.is_terminal == episode_end

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
            update_horizon=1,
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
        state = rb._memory[STACK_SIZE - 1].state
        for i in range(STACK_SIZE):
            np.testing.assert_array_equal(np.full(OBSERVATION_SHAPE, i), state[:, :, i])

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


if __name__ == "__main__":
    absltest.main()
