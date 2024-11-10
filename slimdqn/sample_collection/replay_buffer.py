# thanks dopamine
"""Simpler implementation of the standard DQN replay memory."""
import collections
import functools
import operator
import pickle
import typing
from typing import Any, Generic, Literal, TypeVar, Iterable, NewType, Optional

from slimdqn.sample_collection import samplers
from slimdqn.sample_collection.samplers import ReplayItemID

import jax
import numpy as np
import numpy.typing as npt

# from orbax import checkpoint as orbax

from flax import struct
import numpy as np
import numpy.typing as npt
import snappy


class TransitionElement(typing.NamedTuple):
    observation: Optional[npt.NDArray[Any]]
    action: int
    reward: float
    is_terminal: bool
    episode_end: bool = False


def compress(buffer: npt.NDArray) -> npt.NDArray:
    """Compress a numpy array using snappy.

    Args:
      buffer: npt.NDArray, buffer to compress.

    Returns:
      Numpy structured array consisting of the following fields:
        data: compressed data in bytes
        shape: the shape of the uncompressed array
        dtype: a string representation of the dtype
    """
    # Compress the numpy array using snappy
    if not buffer.flags["C_CONTIGUOUS"]:
        buffer = buffer.copy(order="C")
    compressed = np.frombuffer(snappy.compress(buffer), dtype=np.uint8)
    # Construct the numpy structured array
    return np.array(
        (compressed, buffer.shape, buffer.dtype.str),
        dtype=[
            ("data", "u1", compressed.shape),
            ("shape", "i4", (len(buffer.shape),)),
            ("dtype", f"S{len(buffer.dtype.str)}"),
        ],
    )


def uncompress(compressed: npt.NDArray) -> npt.NDArray:
    """Uncompress a numpy array that has been compressed via `compress`.

    Args:
      compressed: npt.NDArray, numpy structured array with data, shape, and dtype.

    Returns:
      Uncompressed npt.NDArray
    """
    # Create shape tuple
    shape = tuple(compressed["shape"])
    # Get the dtype string
    dtype = compressed["dtype"].item()
    # Get the compressed bytes and uncompress them using snappy
    compressed_bytes = compressed["data"].tobytes()
    uncompressed = snappy.uncompress(compressed_bytes)
    # Construct the numpy array
    return np.ndarray(shape=shape, dtype=dtype, buffer=uncompressed)


class ReplayElement(struct.PyTreeNode):
    """A single replay transition element supporting compression."""

    state: npt.NDArray[np.float64]
    action: "npt.NDArray[np.int_] | npt.NDArray[np.float64] | int"
    reward: "npt.NDArray[np.float64] | float"
    next_state: npt.NDArray[np.float64]
    is_terminal: "npt.NDArray[np.bool_] | bool"
    episode_end: "npt.NDArray[np.bool_] | bool"

    def pack(self) -> "ReplayElement":
        # NOTE: pytype has a problem subclassing generics.
        # This should be solved in Py311 with `typing.Self`
        # pytype: disable=attribute-error
        return self.replace(
            state=compress(self.state),
            next_state=compress(self.next_state),
        )
        # pytype: enable=attribute-error

    def unpack(self) -> "ReplayElement":
        # NOTE: pytype has a problem subclassing generics.
        # This should be solved in Py311 with `typing.Self`
        # pytype: disable=attribute-error
        return self.replace(
            state=uncompress(self.state),
            next_state=uncompress(self.next_state),
        )
        # pytype: enable=attribute-error


ReplayElementT = TypeVar("ReplayElementT", bound=ReplayElement)


class ReplayBuffer:
    """A Jax re-implementation of the Dopamine Replay Buffer.

    Stores transitions, state, action, reward, next_state, terminal (and any
    extra contents specified) in a circular buffer and provides a uniform
    sampling mechanism.

    The main changes from the original Dopamine implementation are:
    - The original Dopamine replay buffer stored the raw observations and stacked
      them upon sampling. Although space efficient, this is time inefficient. We
      use the same compression mechanism used by dqn_zoo to store only stacked
      states in a space efficient manner.
    - Similarly, n-step transitions were computed at sample time. We only store
      n-step returns in the buffer by using an Accumulator (which is also used for
      stacking).
    - The above two points allow us to sample uniformly from the FIFO queue,
      without needing to determine invalid ranges (as in the original replay
      buffer). The computation of invalid ranges was inefficient, as it required
      verifying each sampled index, and potentially resampling them if they fell
      in an invalid range. This computation scales linearly with the batch size,
      and is thus quite inefficient.
    - The original Dopamine replay buffer maintained a static array and performed
      modulo arithmetic when adding/sampling. This is unnecessarily complicated
      and we can achieve the same result by maintaining a FIFO queue (using an
      OrderedDict structure) containing only valid transitions.
    """

    def __init__(
        self,
        sampling_distribution,
        *,
        batch_size: int,
        max_capacity: int,
        stack_size: int = 4,
        update_horizon: int = 1,
        gamma: float = 0.99,
        checkpoint_duration: int = 4,
        compress: bool = True,
    ):
        """Initializes the ReplayBuffer."""
        self.add_count = 0
        self._max_capacity = max_capacity
        self._compress = compress
        self._memory = collections.OrderedDict[ReplayItemID, ReplayElementT]()

        self._sampling_distribution = sampling_distribution

        if checkpoint_duration < 1:
            raise ValueError(f"Invalid checkpoint_duration: {checkpoint_duration}")

        self._checkpoint_duration = checkpoint_duration
        self._batch_size = batch_size

        assert update_horizon > 0, "update_horizon must be positive."
        assert stack_size > 0, "stack_size must be positive."

        self._stack_size = stack_size
        self._update_horizon = update_horizon
        self._gamma = gamma

        self._trajectory = collections.deque[TransitionElement](maxlen=self._update_horizon + self._stack_size)

    def _make_replay_element(self) -> "ReplayElementT | None":

        trajectory_len = len(self._trajectory)
        last_transition = self._trajectory[-1]
        # Check if we have a valid transition, i.e. we either
        #   1) have accumulated more transitions than the update horizon
        #   2) have a trajectory shorter than the update horizon, but the
        #      last element is terminal
        if not (trajectory_len > self._update_horizon or (trajectory_len > 1 and last_transition.is_terminal)):
            return None

        # Calculate effective horizon, this can differ from the update horizon
        # when we have n-step transitions where the last observation is terminal.
        effective_horizon = self._update_horizon
        if last_transition.is_terminal and trajectory_len <= self._update_horizon:
            effective_horizon = trajectory_len - 1

        # pytype: disable=attribute-error
        observation_shape = last_transition.observation.shape + (self._stack_size,)
        observation_dtype = last_transition.observation.dtype
        # pytype: enable=attribute-error

        o_tm1 = np.zeros(observation_shape, observation_dtype)
        # Initialize the slice for which this observation is valid.
        # The start index for o_tm1 is the start of the n-step trajectory.
        # The end index for o_tm1 is just moving over `stack size`.
        o_tm1_slice = slice(
            trajectory_len - effective_horizon - self._stack_size,
            trajectory_len - effective_horizon - 1,
        )
        # The action chosen will be the last transition in the stack.
        a_tm1 = self._trajectory[o_tm1_slice.stop].action

        o_t = np.zeros(observation_shape, observation_dtype)
        # Initialize the slice for which this observation is valid.
        # The start index for o_t is just moving backwards `stack size`.
        # The end index for o_t is just the last index of the n-step trajectory.
        o_t_slice = slice(
            trajectory_len - self._stack_size,
            trajectory_len - 1,
        )
        # Terminal information will come from the last transition in the stack
        is_terminal = self._trajectory[o_t_slice.stop].is_terminal
        episode_end = self._trajectory[o_t_slice.stop].is_terminal

        # Slice to accumulate n-step returns. This will be the end
        # transition of o_tm1 plus the effective horizon.
        # This might over-run the trajectory length in the case of n-step
        # returns where the last transition is terminal.
        gamma_slice = slice(
            o_tm1_slice.stop,
            o_tm1_slice.stop + self._update_horizon - 1,
        )
        assert o_t_slice.stop - o_tm1_slice.stop == effective_horizon
        assert o_t_slice.stop - 1 >= o_tm1_slice.stop

        # Now we'll iterate through the n-step trajectory and compute the
        # cumulant and insert the observations into the appropriate stacks
        r_t = 0.0
        for t, transition_t in enumerate(self._trajectory):
            # If we should be accumulating reward for an n-step return?
            if gamma_slice.start <= t <= gamma_slice.stop:
                r_t += transition_t.reward * (self._gamma ** (t - gamma_slice.start))

            # If we should be accumulating frames for the frame-stack?
            if o_tm1_slice.start <= t <= o_tm1_slice.stop:
                o_tm1[..., t - o_tm1_slice.start] = transition_t.observation
            if o_t_slice.start <= t <= o_t_slice.stop:
                o_t[..., t - o_t_slice.start] = transition_t.observation

        return ReplayElement(
            state=o_tm1,
            action=a_tm1,
            reward=r_t,
            next_state=o_t,
            is_terminal=is_terminal,
            episode_end=episode_end,
        )

    def accumulate(self, transition: TransitionElement) -> Iterable[ReplayElement]:
        """Add a transition to the accumulator, maybe receive valid ReplayElements.

        If the transition has a terminal or end of episode signal, it will create a
        new trajectory.

        If the transition is terminal the iterator will yield multiple elements
        corresponding to n-step returns with n ∊ [1, n].

        Args:
          transition: TransitionElement to add.

        Yields:
          A ReplayElement if there is a valid transition.
        """
        self._trajectory.append(transition)

        # If this transition terminated a trajectory we'll yield
        # as many replay elements as we can before clearing the trajectory
        if transition.is_terminal:
            while replay_element := self._make_replay_element():
                yield replay_element
                self._trajectory.popleft()
            self._trajectory.clear()
        else:
            # Attempt to yield a replay element
            if replay_element := self._make_replay_element():
                yield replay_element
            # If the transition truncates the trajectory then clear it
            # We don't yield n-1, n-2, ..., 1 step returns as
            # this transition is not terminal.
            if transition.episode_end:
                self._trajectory.clear()

    def clear(self) -> None:
        """Clear the accumulator."""
        self._trajectory.clear()

    def add(self, transition: TransitionElement, **kwargs: Any) -> None:
        """Add a transition to the replay buffer."""
        for replay_element in self.accumulate(transition):
            if self._compress:
                replay_element = replay_element.pack()

            # Add replay element to memory
            key = ReplayItemID(self.add_count)
            self._memory[key] = replay_element
            self._sampling_distribution.add(key, **kwargs)
            self.add_count += 1
            # If we're beyond our capacity...
            if self.add_count > self._max_capacity:
                # Pop the oldest item from memory and keep the key
                # so we can ask the sampling distribution to remove it
                oldest_key, _ = self._memory.popitem(last=False)
                self._sampling_distribution.remove(oldest_key)

    def sample(
        self,
        size: "int | None" = None,
        *,
        with_sample_metadata: bool = False,
    ) -> "ReplayElementT | tuple[ReplayElementT, samplers.SampleMetadata]":
        """Sample a batch of elements from the replay buffer."""
        if self.add_count < 1:
            raise ValueError("No samples in replay buffer!")
        if size is None:
            size = self._batch_size
        if size < 1:
            raise ValueError(f"Invalid size: {size}, size must be >= 1.")

        samples = self._sampling_distribution.sample(size)
        replay_elements = operator.itemgetter(*samples.keys)(self._memory)
        if not isinstance(replay_elements, tuple):
            # When size == 1, replay_elements will hold a single ReplayElement, as
            # opposed to a tuple of ReplayElements (which is what is expected in the
            # call to tree_map below). We fix this by forcing replay_elements into a
            # tuple, if necessary.
            replay_elements = (replay_elements,)
        if self._compress:
            replay_elements = map(operator.methodcaller("unpack"), replay_elements)

        batch = jax.tree_util.tree_map(lambda *xs: np.stack(xs), *replay_elements)
        return (batch, samples) if with_sample_metadata else batch

    def update(
        self,
        keys: "npt.NDArray[ReplayItemID] | ReplayItemID",
        **kwargs: Any,
    ) -> None:
        self._sampling_distribution.update(keys, **kwargs)

    def clear(self) -> None:
        """Clear the replay buffer."""
        self.add_count = 0
        self._memory.clear()
        self._transition_accumulator.clear()
        self._sampling_distribution.clear()

    # def to_state_dict(self) -> dict[str, Any]:
    #     """Serialize replay buffer to a state dictionary."""
    #     # Serialize memory. We'll serialize keys and values separately.
    #     keys = list(self._memory.keys())
    #     # To serialize values we'll flatten each transition element.
    #     # This will serialize replay elements as:
    #     #   [[state, action, reward, next_state, is_terminal, episode_end], ...]
    #     values = iter(self._memory.values())
    #     leaves, treedef = jax.tree_util.tree_flatten(next(values, None))
    #     values = [] if not leaves else [leaves, *map(treedef.flatten_up_to, values)]

    #     return {
    #         "add_count": self.add_count,
    #         "memory": {
    #             "keys": keys,
    #             "values": values,
    #             "treedef": pickle.dumps(treedef),
    #         },
    #         "sampling_distribution": self._sampling_distribution.to_state_dict(),
    #         "transition_accumulator": self._transition_accumulator.to_state_dict(),
    #     }

    # def from_state_dict(self, state_dict: dict[str, Any]) -> None:
    #     """Deserialize and mutate replay buffer using state dictionary."""
    #     self.add_count = state_dict["add_count"]
    #     self._transition_accumulator.from_state_dict(state_dict["transition_accumulator"])
    #     self._sampling_distribution.from_state_dict(state_dict["sampling_distribution"])

    #     # Restore memory
    #     memory_keys = state_dict["memory"]["keys"]
    #     # Each element of the list is a flattened replay element, unflatten them
    #     # i.e., we have storage like:
    #     #   [[state, action, reward, next_state, is_terminal, episode_end], ...]
    #     # and after unflattening we'll have:
    #     #   [ReplayElementT(...), ...]
    #     memory_treedef: jax.tree_util.PyTreeDef = pickle.loads(state_dict["memory"]["treedef"])
    #     memory_values = map(memory_treedef.unflatten, state_dict["memory"]["values"])

    #     # Create our new ordered dictionary from the restored keys and values
    #     self._memory = collections.OrderedDict[ReplayItemID, ReplayElementT](
    #         zip(memory_keys, memory_values, strict=True)
    #     )

    # @functools.lru_cache
    # def _make_checkpoint_manager(self, checkpoint_dir: str) -> orbax.CheckpointManager:
    #     """Create orbax checkpoint manager, cache the manager based on path."""
    #     return orbax.CheckpointManager(
    #         checkpoint_dir,
    #         checkpointers={
    #             "replay": orbax.Checkpointer(
    #                 checkpointers.CheckpointHandler[ReplayBuffer](),
    #             )
    #         },
    #         options=orbax.CheckpointManagerOptions(
    #             max_to_keep=self._checkpoint_duration,
    #             create=True,
    #         ),
    #     )

    # def save(self, checkpoint_dir: str, iteration_number: int):
    #     """Save the ReplayBuffer attributes into a file.

    #     Args:
    #       checkpoint_dir: the directory where numpy checkpoint files should be
    #         saved. Must already exist.
    #       iteration_number: iteration_number to use as a suffix in naming.
    #     """
    #     checkpoint_manager = self._make_checkpoint_manager(checkpoint_dir)
    #     checkpoint_manager.save(iteration_number, {"replay": self})

    # def load(self, checkpoint_dir: str, iteration_number: int):
    #     """Restores from a checkpoint.

    #     Args:
    #       checkpoint_dir: the directory where to read the checkpoint.
    #       iteration_number: iteration_number to use as a suffix in naming.
    #     """
    #     checkpoint_manager = self._make_checkpoint_manager(checkpoint_dir)
    #     # NOTE: Make sure not to pass in `items={'replay': self}` as this will
    #     # create a deep copy and we want to mutate in-place.
    #     # If we don't pass items then we get back a state dictionary
    #     # that we can use to mutate in-place.
    #     state_dict = checkpoint_manager.restore(iteration_number)
    #     self.from_state_dict(state_dict["replay"])
