import collections
import math

import numpy as np
import torch

# Defines a type describing part of the tuple returned by the replay
# memory. Each element of the tuple is a tensor of shape [batch, ...] where
# ... is defined the 'shape' field of ReplayElement. The tensor type is
# given by the 'type' field. The 'name' field is for convenience and ease of
# debugging.
ReplayElement = (
    collections.namedtuple('shape_type', ['name', 'shape', 'type']))

SampleBatch = (collections.namedtuple('sample_batch', ['observations', 'actions', 'rewards',
                                                       'next_observations', 'dones']))


def modulo_range(start, length, modulo):
    for i in range(length):
        yield (start + i) % modulo


class ReplayBuffer(object):
    """A simple out-of-graph Replay Buffer.

  Stores transitions, state, action, reward, next_state, terminal (and any
  extra contents specified) in a circular buffer and provides a uniform
  transition sampling function.

  When the states consist of stacks of observations storing the states is
  inefficient. This class writes observations and constructs the stacked states
  at sample time.

  Attributes:
    add_count: int, counter of how many transitions have been added (including
      the blank ones at the beginning of an episode).
    episode_end_indices: set[int], a set of indices corresponding to the
      end of an episode.
  """

    def __init__(self,
                 observation_shape,
                 replay_capacity,
                 batch_size,
                 update_horizon=1,
                 gamma=0.99,
                 max_sample_attempts=1000,
                 observation_dtype=np.uint8,
                 terminal_dtype=np.uint8,
                 action_shape=(),
                 action_dtype=np.int32,
                 reward_shape=(),
                 reward_dtype=np.float32,
                 ):
        """Initializes OutOfGraphReplayBuffer.

    Args:
      observation_shape: tuple of ints.
      replay_capacity: int, number of transitions to keep in memory.
      batch_size: int.
      update_horizon: int, length of update ('n' in n-step update).
      gamma: float, the discount factor.
      max_sample_attempts: int, the maximum number of attempts allowed to
        get a sample.
      observation_dtype: np.dtype, type of the observations. Defaults to
        np.uint8 for Atari 2600.
      terminal_dtype: np.dtype, type of the terminals. Defaults to np.uint8 for
        Atari 2600.
      action_shape: tuple of ints, the shape for the action vector. Empty tuple
        means the action is a scalar.
      action_dtype: np.dtype, type of elements in the action.
      reward_shape: tuple of ints, the shape of the reward vector. Empty tuple
        means the reward is a scalar.
      reward_dtype: np.dtype, type of elements in the reward.

    Raises:
      ValueError: If replay_capacity is too small to hold at least one
        transition.
    """
        self._action_shape = action_shape
        self._action_dtype = action_dtype
        self._reward_shape = reward_shape
        self._reward_dtype = reward_dtype
        self._observation_shape = observation_shape
        self._replay_capacity = replay_capacity
        self._batch_size = batch_size
        self._update_horizon = update_horizon
        self._gamma = gamma
        self._observation_dtype = observation_dtype
        self._terminal_dtype = terminal_dtype
        self._max_sample_attempts = max_sample_attempts
        self._create_storage()
        self.add_count = np.array(0)
        # When the horizon is > 1, we compute the sum of discounted rewards as a dot
        # product using the precomputed vector <gamma^0, gamma^1, ..., gamma^{n-1}>.
        self._cumulative_discount_vector = np.array(
            [math.pow(self._gamma, n) for n in range(update_horizon)],
            dtype=np.float32)
        self._next_experience_is_episode_start = True
        self.episode_end_indices = set()

    def _create_storage(self):
        """Creates the numpy arrays used to store transitions.
    """
        self._store = {}
        for storage_element in self.get_storage_signature():
            array_shape = [self._replay_capacity] + list(storage_element.shape)
            self._store[storage_element.name] = np.empty(
                array_shape, dtype=storage_element.type)

    def get_storage_signature(self):
        """Returns a default list of elements to be stored in this replay memory.
    Note - Derived classes may return a different signature.
    Returns:
      list of ReplayElements defining the type of the contents stored.
    """
        storage_elements = [
            ReplayElement('observation', self._observation_shape,
                          self._observation_dtype),
            ReplayElement('action', self._action_shape, self._action_dtype),
            ReplayElement('reward', self._reward_shape, self._reward_dtype),
            ReplayElement('done', (), self._terminal_dtype)
        ]

        return storage_elements

    def add(self,
            observation,
            action,
            reward,
            terminal,
            episode_end=False):
        """Adds a transition to the replay memory.
    If the replay memory is at capacity the oldest transition will be discarded.

    Args:
      observation: np.array with shape observation_shape.
      action: int, the action in the transition.
      reward: float, the reward received in the transition.
      terminal: np.dtype, acts as a boolean indicating whether the transition
                was terminal (1) or not (0).
      episode_end: bool, whether this experience is the last experience in
        the episode. This is useful for tasks that terminate due to time-out,
        but do not end on a terminal state. Overloading 'terminal' may not
        be sufficient in this case, since 'terminal' is passed to the agent
        for training. 'episode_end' allows the replay buffer to determine
        episode boundaries without passing that information to the agent.
    """

        if self._next_experience_is_episode_start:
            self._next_experience_is_episode_start = False

        if episode_end or terminal:
            self.episode_end_indices.add(self.cursor())
            self._next_experience_is_episode_start = True
        else:
            self.episode_end_indices.discard(self.cursor())  # If present

        action = action[0]
        self._add(observation, action, reward, terminal)

    def _add(self, *args):
        """Internal add method to add to the storage arrays.

    Args:
      *args: All the elements in a transition.
    """
        transition = {e.name: args[idx]
                      for idx, e in enumerate(self.get_storage_signature())}
        cursor = self.cursor()
        for arg_name in transition:
            self._store[arg_name][cursor] = transition[arg_name]

        self.add_count += 1

    def is_empty(self):
        """Is the Replay Buffer empty?"""
        return self.add_count == 0

    def is_full(self):
        """Is the Replay Buffer full?"""
        return self.add_count >= self._replay_capacity

    def cursor(self):
        """Index to the location where the next transition will be written."""
        return self.add_count % self._replay_capacity

    def get_range(self, array, start_index, end_index):
        """Returns the range of array at the index handling wraparound if necessary.

    Args:
      array: np.array, the array to get the stack from.
      start_index: int, index to the start of the range to be returned. Range
        will wraparound if start_index is smaller than 0.
      end_index: int, exclusive end index. Range will wraparound if end_index
        exceeds replay_capacity.

    Returns:
      np.array, with shape [end_index - start_index, array.shape[1:]].
    """
        assert end_index > start_index, 'end_index must be larger than start_index'
        assert end_index >= 0
        assert start_index < self._replay_capacity
        if not self.is_full():
            assert end_index <= self.cursor(), (
                'Index {} has not been added.'.format(start_index))

        # Fast slice read when there is no wraparound.
        if start_index % self._replay_capacity < end_index % self._replay_capacity:
            return_array = array[start_index:end_index, ...]
        # Slow list read.
        else:
            indices = [(start_index + i) % self._replay_capacity
                       for i in range(end_index - start_index)]
            return_array = array[indices, ...]
        return return_array

    def is_valid_transition(self, index):
        """Checks if the index contains a valid transition.

    Checks for collisions with the end of episodes and the current position
    of the cursor.

    Args:
      index: int, the index to the state in the transition.

    Returns:
      Is the index valid: Boolean.

    """
        # Check the index is in the valid range
        if index < 0 or index >= self._replay_capacity:
            return False
        if not self.is_full():
            # The indices and next_indices must be smaller than the cursor.
            if index >= self.cursor() - self._update_horizon:
                for i in range(index, self.cursor()):
                    if self._store['done'][i]:
                        return True
                return False

        # If the episode ends before the update horizon, without a terminal signal,
        # it is invalid.
        for i in modulo_range(index, self._update_horizon, self._replay_capacity):
            if i in self.episode_end_indices and not self._store['done'][i]:
                return False

        return True

    def sample_index_batch(self, batch_size):
        """Returns a batch of valid indices sampled uniformly.

    Args:
      batch_size: int, number of indices returned.

    Returns:
      list of ints, a batch of valid indices sampled uniformly.

    Raises:
      RuntimeError: If the batch was not constructed after maximum number of
        tries.
    """
        if self.is_full():
            # add_count >= self._replay_capacity > self._stack_size
            min_id = self.cursor() - self._replay_capacity
            max_id = self.cursor() - self._update_horizon
        else:
            # add_count < self._replay_capacity
            min_id = 0
            max_id = self.cursor() - self._update_horizon

        indices = []
        attempt_count = 0
        while (len(indices) < batch_size and
               attempt_count < self._max_sample_attempts):
            index = np.random.randint(min_id, max_id) % self._replay_capacity
            if self.is_valid_transition(index):
                indices.append(index)
            else:
                attempt_count += 1
        if len(indices) != batch_size:
            raise RuntimeError(
                'Max sample attempts: Tried {} times but only sampled {}'
                ' valid indices. Batch size is {}'.
                format(self._max_sample_attempts, len(indices), batch_size))

        return indices

    def sample_transition_batch(self, batch_size=None, indices=None):
        """Returns a batch of transitions (including any extra contents).

    If get_transition_elements has been overridden and defines elements not
    stored in self._store, an empty array will be returned and it will be
    left to the child class to fill it. For example, for the child class
    OutOfGraphPrioritizedReplayBuffer, the contents of the
    sampling_probabilities are stored separately in a sum tree.

    When the transition is terminal next_state_batch has undefined contents.

    NOTE: This transition contains the indices of the sampled elements. These
    are only valid during the call to sample_transition_batch, i.e. they may
    be used by subclasses of this replay buffer but may point to different data
    as soon as sampling is done.

    Args:
      batch_size: int, number of transitions returned. If None, the default
        batch_size will be used.
      indices: None or list of ints, the indices of every transition in the
        batch. If None, sample the indices uniformly.

    Returns:
      transition_batch: tuple of np.arrays with the shape and type as in
        get_transition_elements().

    Raises:
      ValueError: If an element to be sampled is missing from the replay buffer.
    """
        if batch_size is None:
            batch_size = self._batch_size
        if indices is None:
            indices = self.sample_index_batch(batch_size)
        assert len(indices) == batch_size

        sampled_batch = {
          'observations': np.empty((batch_size,) + self._observation_shape, dtype=self._observation_dtype),
          'actions': np.empty((batch_size,) + self._action_shape, dtype=self._action_dtype),
          'rewards': np.empty((batch_size,) + self._reward_shape, dtype=self._reward_dtype),
          'next_observations': np.empty((batch_size,) + self._observation_shape,
                                        dtype=self._observation_dtype),
          'dones': np.empty((batch_size,), dtype=self._terminal_dtype)
        }

        for batch_element, state_index in enumerate(indices):
            trajectory_indices = [(state_index + j) % self._replay_capacity
                                  for j in range(self._update_horizon)]
            trajectory_terminals = self._store['done'][trajectory_indices]
            is_terminal_transition = trajectory_terminals.any()
            if not is_terminal_transition:
                trajectory_length = self._update_horizon
            else:
                # np.argmax of a bool array returns the index of the first True.
                trajectory_length = np.argmax(trajectory_terminals.astype(bool), 0) + 1
            next_state_index = state_index + trajectory_length
            trajectory_discount_vector = (self._cumulative_discount_vector[:trajectory_length])
            trajectory_rewards = self.get_range(self._store['reward'], state_index, next_state_index)

            for element_type, element in sampled_batch.items():
                if element_type == 'observations':
                    element[batch_element] = self._store['observation'][state_index]
                elif element_type == 'actions':
                    element[batch_element] = self._store['action'][state_index]
                elif element_type == 'rewards':
                    # compute the discounted sum of rewards in the trajectory.
                    element[batch_element] = np.sum(
                        trajectory_discount_vector * trajectory_rewards, axis=0)
                elif element_type == 'next_observations':
                    element[batch_element] = self._store['observation'][next_state_index % self._replay_capacity]
                elif element_type == 'dones':
                    element[batch_element] = is_terminal_transition

                # We assume the other elements are filled in by the subclass.

        sampled_batch['observations'] = torch.tensor(sampled_batch['observations'], dtype=torch.float32)
        sampled_batch['next_observations'] = torch.tensor(sampled_batch['next_observations'], dtype=torch.float32)
        sampled_batch['rewards'] = torch.tensor(sampled_batch['rewards'], dtype=torch.float32)
        sampled_batch['actions'] = torch.tensor(sampled_batch['actions'], dtype=torch.int64)

        return sampled_batch
