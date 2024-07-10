import collections

import numpy as np
import jax
import jax.numpy as jnp

ReplayElement = collections.namedtuple("shape_type", ["name", "shape", "type"])


def modulo_range(start, length, modulo):
    for i in range(length):
        yield (start + i) % modulo


class ReplayBuffer(object):
    """
    A simple out-of-graph Replay Buffer.
    Stores transitions, state, action, reward, next_state, terminal (and any
    extra contents specified) in a circular buffer and provides a uniform
    transition sampling function.
    """

    def __init__(
        self,
        observation_shape,
        replay_capacity,
        update_horizon=1,
        gamma=0.99,
        max_sample_attempts=1000,
        observation_dtype=np.float32,
        terminal_dtype=np.uint8,
        action_dtype=np.int32,
        reward_dtype=np.float32,
    ):
        self._observation_shape = observation_shape
        self._observation_dtype = observation_dtype
        self._action_dtype = action_dtype
        self._reward_dtype = reward_dtype
        self._terminal_dtype = terminal_dtype

        self._replay_capacity = replay_capacity
        self._max_sample_attempts = max_sample_attempts

        self._update_horizon = update_horizon
        self._gamma = gamma

        self._create_storage()
        self.add_count = 0
        self._cumulative_discount_vector = np.array([self._gamma**n for n in range(update_horizon)], dtype=np.float32)
        # store the indices of all the transitions where the episode was truncated
        self.episode_trunc_indices = set()
        self.last_added_transition_index = None

    def _create_storage(self):
        self._store = {}
        for storage_element in self.get_storage_signature():
            array_shape = [self._replay_capacity] + list(storage_element.shape)
            self._store[storage_element.name] = np.empty(array_shape, dtype=storage_element.type)
        # contains the index of all observations where done=False and episode was truncated

    def get_storage_signature(self):
        storage_elements = [
            ReplayElement("observations", self._observation_shape, self._observation_dtype),
            ReplayElement("actions", (), self._action_dtype),
            ReplayElement("rewards", (), self._reward_dtype),
            ReplayElement("dones", (), self._terminal_dtype),
        ]

        return storage_elements

    def add(self, observation, action, reward, terminal, episode_end=False):
        """
        If the replay memory is at capacity the oldest transition will be discarded.

        Args:
        episode_end: bool, whether this experience is the last experience in
        the episode. This is useful for tasks that terminate due to time-out,
        but do not end on a terminal state. Overloading 'terminal' may not
        be sufficient in this case, since 'terminal' is passed to the agent
        for training. 'episode_end' allows the replay buffer to determine
        episode boundaries without passing that information to the agent.
        """

        if episode_end and not terminal:
            self.episode_trunc_indices.add(self.cursor())
        else:
            self.episode_trunc_indices.discard(self.cursor())  # If present

        self._add(observation, action, reward, terminal)

    def _add(self, *args):
        cursor = self.cursor()
        self.last_added_transition_index = cursor
        for idx, e in enumerate(self.get_storage_signature()):
            self._store[e.name][cursor] = args[idx]
        self.add_count += 1

    def is_full(self):
        return self.add_count >= self._replay_capacity

    def cursor(self):
        """Index to the location where the next transition will be written."""
        return self.add_count % self._replay_capacity

    def get_range(self, array, start_index, end_index):
        """
        Returns the range of array at the index handling wraparound if necessary.
        Args:
        array:  the array to get the stack from.
        start_index: index to the start of the range to be returned. Range will wraparound if start_index is smaller than 0.
        end_index: exclusive end index. Range will wraparound if end_index
            exceeds replay_capacity.
        """

        if start_index % self._replay_capacity < end_index % self._replay_capacity:
            return_array = array[start_index:end_index, ...]
        else:
            indices = [(start_index + i) % self._replay_capacity for i in range(end_index - start_index)]
            return_array = array[indices, ...]
        return return_array

    def is_valid_transition(self, index):
        # Check the index is in the valid range
        if index < 0 or index >= self._replay_capacity:
            return False
        if not self.is_full():
            # The indices and next_indices must be smaller than the cursor.
            if index >= self.cursor() - self._update_horizon:
                for i in range(index, self.cursor()):
                    if self._store["dones"][i]:
                        return True
                return False

        # If the episode is truncated before the update horizon, it is invalid.
        for i in modulo_range(index, self._update_horizon, self._replay_capacity):
            if self._store["dones"][i]:
                break
            if i in self.episode_trunc_indices or (
                (i == self.last_added_transition_index) and (not self._store["dones"][i])
            ):
                return False

        return True

    def sample_index_batch(self, batch_size, batching_key: jax.random.PRNGKey):
        if self.is_full():
            # add_count >= self._replay_capacity
            min_id = self.cursor() - self._replay_capacity
            max_id = self.cursor()
        else:
            # add_count < self._replay_capacity
            min_id = 0
            max_id = self.cursor()

        indices = []
        attempt_count = 0
        while len(indices) < batch_size and attempt_count < self._max_sample_attempts:
            batching_key, key = jax.random.split(batching_key)
            index = jax.random.randint(key, shape=(), minval=min_id, maxval=max_id).item() % self._replay_capacity
            if self.is_valid_transition(index):
                indices.append(index)
            else:
                attempt_count += 1
        if len(indices) != batch_size:
            raise RuntimeError(
                "Max sample attempts: Tried {} times but only sampled {}"
                " valid indices. Batch size is {}".format(self._max_sample_attempts, len(indices), batch_size)
            )
        return indices

    def sample_transition_batch(self, batch_size, batching_key: jax.random.PRNGKey):
        indices = self.sample_index_batch(batch_size, batching_key)
        sampled_batch = {
            "observations": np.empty((batch_size,) + self._observation_shape, dtype=self._observation_dtype),
            "actions": np.empty((batch_size,), dtype=self._action_dtype),
            "rewards": np.empty((batch_size,), dtype=self._reward_dtype),
            "next_observations": np.empty((batch_size,) + self._observation_shape, dtype=self._observation_dtype),
            "dones": np.empty((batch_size,), dtype=self._terminal_dtype),
        }

        for batch_element, state_index in enumerate(indices):
            trajectory_indices = [(state_index + j) % self._replay_capacity for j in range(self._update_horizon)]
            trajectory_terminals = self._store["dones"][trajectory_indices]
            is_terminal_transition = trajectory_terminals.any()
            if not is_terminal_transition:
                trajectory_length = self._update_horizon
            else:
                trajectory_length = np.argmax(trajectory_terminals.astype(bool)) + 1
            next_state_index = state_index + trajectory_length
            trajectory_discount_vector = self._cumulative_discount_vector[:trajectory_length]
            trajectory_rewards = self.get_range(self._store["rewards"], state_index, next_state_index)

            for element_type, element in sampled_batch.items():
                sampled_batch["observations"][batch_element] = self._store["observations"][state_index]
                sampled_batch["actions"][batch_element] = self._store["actions"][state_index]
                # compute the discounted sum of rewards in the trajectory.
                sampled_batch["rewards"][batch_element] = np.sum(
                    trajectory_discount_vector * trajectory_rewards, axis=0
                )
                sampled_batch["next_observations"][batch_element] = self._store["observations"][
                    next_state_index % self._replay_capacity
                ]
                sampled_batch["dones"][batch_element] = is_terminal_transition

        sampled_batch["observations"] = jnp.array(sampled_batch["observations"], dtype=jnp.float32)
        sampled_batch["next_observations"] = jnp.array(sampled_batch["next_observations"], dtype=jnp.float32)
        sampled_batch["rewards"] = jnp.array(sampled_batch["rewards"], dtype=jnp.float32)
        sampled_batch["actions"] = jnp.array(sampled_batch["actions"], dtype=jnp.int32)
        sampled_batch["dones"] = jnp.array(sampled_batch["dones"], dtype=jnp.uint8)

        return sampled_batch
