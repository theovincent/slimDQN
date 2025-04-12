# Inspired by dopamine implementation: https://github.com/google/dopamine/blob/master/dopamine/jax/replay_memory/replay_buffer.py
"""Simpler implementation of the standard DQN replay memory."""
import collections
import operator
import typing
from typing import Any, Iterable, Optional

import jax
import numpy as np
import numpy.typing as npt

from flax import struct
import snappy

from slimdqn.sample_collection import ReplayItemID


class TransitionElement(typing.NamedTuple):
    observation: Optional[npt.NDArray[Any]]
    action: int
    reward: float
    is_terminal: bool
    episode_end: bool = False


class ReplayElement(struct.PyTreeNode):
    """A single replay transition element supporting compression."""

    state: npt.NDArray[np.float64]
    action: int
    reward: float
    next_state: npt.NDArray[np.float64]
    is_terminal: bool

    @staticmethod
    def compress(buffer: npt.NDArray) -> npt.NDArray:
        if not buffer.flags["C_CONTIGUOUS"]:
            buffer = buffer.copy(order="C")
        compressed = np.frombuffer(snappy.compress(buffer), dtype=np.uint8)

        return np.array(
            (compressed, buffer.shape, buffer.dtype.str),
            dtype=[
                ("data", "u1", compressed.shape),
                ("shape", "i4", (len(buffer.shape),)),
                ("dtype", f"S{len(buffer.dtype.str)}"),
            ],
        )

    @staticmethod
    def uncompress(compressed: npt.NDArray) -> npt.NDArray:
        shape = tuple(compressed["shape"])
        dtype = compressed["dtype"].item()
        compressed_bytes = compressed["data"].tobytes()
        uncompressed = snappy.uncompress(compressed_bytes)
        return np.ndarray(shape=shape, dtype=dtype, buffer=uncompressed)

    def pack(self) -> "ReplayElement":
        return self.replace(
            state=ReplayElement.compress(self.state),
            next_state=ReplayElement.compress(self.next_state),
        )

    def unpack(self) -> "ReplayElement":
        return self.replace(
            state=ReplayElement.uncompress(self.state),
            next_state=ReplayElement.uncompress(self.next_state),
        )


class ReplayBuffer:

    def __init__(
        self,
        sampling_distribution,
        batch_size: int,
        max_capacity: int,
        stack_size: int = 4,
        update_horizon: int = 1,
        gamma: float = 0.99,
        checkpoint_duration: int = 4,
        compress: bool = True,
        clipping: callable = None,
    ):
        self.add_count = 0
        self._max_capacity = max_capacity
        self._compress = compress
        self._memory = collections.OrderedDict[ReplayItemID, ReplayElement]()

        self._sampling_distribution = sampling_distribution

        self._checkpoint_duration = checkpoint_duration
        self._batch_size = batch_size

        self._stack_size = stack_size
        self._update_horizon = update_horizon
        self._gamma = gamma
        self._clipping = clipping

        self._trajectory = collections.deque[TransitionElement](maxlen=self._update_horizon + self._stack_size)

    def _terminal_replay_element(self, has_popped_left: bool) -> ReplayElement:
        # --- Only called if self._trajectory[-1] leads to a terminal state (self._trajectory[-1].terminal is True) ---
        trajectory_len = len(self._trajectory)
        if has_popped_left:
            # We need at least stack size observations to construct the state, the next state is irrelevant.
            if trajectory_len < self._stack_size:
                return None
            is_terminal = True
            # The trajectory is composed of stack_size observations + effective_horizon observations, the next state is irrelevant.
            effective_horizon = trajectory_len - self._stack_size
        else:
            if trajectory_len == 0:
                return None
            # special case where the episode starts with an observation leading to a terminal state.
            elif trajectory_len == 1:
                is_terminal = True
            # first time that the terminal flag is True, the state does not lead to a terminal state (only the next state does).
            else:
                is_terminal = False
            # If not enough transitions are available, the largest possible update horizon is the trajectory length minus 1.
            if trajectory_len < self._update_horizon:
                effective_horizon = trajectory_len - 1
            else:
                effective_horizon = self._update_horizon

        return self._replay_element_from_effective_horizon(effective_horizon, is_terminal)

    def _non_terminal_replay_element(self) -> ReplayElement:
        # --- Only called if self._trajectory[-1] does not lead to a terminal state (self._trajectory[-1].terminal is False) ---
        trajectory_len = len(self._trajectory)
        # 2 observations are needed (state and next state)
        if trajectory_len < 2:
            return None
        # If not enough transitions are available, the largest possible update horizon is the trajectory length minus 1.
        if trajectory_len < self._update_horizon + 1:
            effective_horizon = trajectory_len - 1
        else:
            effective_horizon = self._update_horizon

        return self._replay_element_from_effective_horizon(effective_horizon, False)

    def _replay_element_from_effective_horizon(self, effective_horizon, is_terminal):
        trajectory_len = len(self._trajectory)

        observation_shape = self._trajectory[0].observation.shape + (self._stack_size,)
        observation_dtype = self._trajectory[0].observation.dtype
        # The state stops at the end of the trajectory minus the effective horizon.
        # Note: if the first index is negative, black observations will be taken.
        o_tm1 = np.zeros(observation_shape, observation_dtype)
        o_tm1_slice = slice(
            trajectory_len - self._stack_size - effective_horizon, trajectory_len - 1 - effective_horizon
        )
        # The chosen action is the action of the last transition of the state.
        a_tm1 = self._trajectory[o_tm1_slice.stop].action
        # The next state is the last stack size observations.
        # +1 is added if the last transition leads to a terminal state as the next state also include the following observation.
        # Note: if the first index is negative, black observations will be taken.
        o_t = np.zeros(observation_shape, observation_dtype)
        o_t_slice = slice(trajectory_len - self._stack_size + int(is_terminal), trajectory_len - 1 + int(is_terminal))

        # Slice to accumulate n-step returns. Starts where o_tm1 stops and finishes one step before o_t stops
        gamma_slice = slice(o_tm1_slice.stop, o_t_slice.stop - 1)

        # We iterate through the n-step trajectory and compute the cumulant and insert the observations into the appropriate stacks
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

        return ReplayElement(state=o_tm1, action=a_tm1, reward=r_t, next_state=o_t, is_terminal=is_terminal)

    def accumulate(self, transition: TransitionElement) -> Iterable[ReplayElement]:
        """Add a transition to the accumulator, maybe receive valid ReplayElements.

        If the transition has a terminal or end of episode signal, it will create a
        new trajectory and yield multiple elements.
        """
        self._trajectory.append(transition)

        if transition.is_terminal:
            has_popped_left = False
            while replay_element := self._terminal_replay_element(has_popped_left):
                yield replay_element
                self._trajectory.popleft()
                has_popped_left = True
            self._trajectory.clear()
        else:
            if replay_element := self._non_terminal_replay_element():
                yield replay_element
            # If the transition truncates the trajectory then clear it
            if transition.episode_end:
                self._trajectory.clear()

    def add(self, transition: TransitionElement, **kwargs: Any) -> None:
        for replay_element in self.accumulate(transition):
            if self._compress:
                replay_element = replay_element.pack()

            key = ReplayItemID(self.add_count)
            self._memory[key] = replay_element
            self._sampling_distribution.add(key, **kwargs)
            self.add_count += 1
            if self.add_count > self._max_capacity:
                oldest_key, _ = self._memory.popitem(last=False)
                self._sampling_distribution.remove(oldest_key)

    def sample(self, size=None) -> ReplayElement | tuple[ReplayElement]:
        """Sample a batch of elements from the replay buffer."""
        assert self.add_count, ValueError("No samples in replay buffer!")

        if size is None:
            size = self._batch_size

        samples = self._sampling_distribution.sample(size)
        replay_elements = operator.itemgetter(*samples)(self._memory)
        if not isinstance(replay_elements, tuple):
            replay_elements = (replay_elements,)
        if self._compress:
            replay_elements = map(operator.methodcaller("unpack"), replay_elements)

        batch = jax.tree_util.tree_map(lambda *xs: np.stack(xs), *replay_elements)
        return batch

    def update(
        self,
        keys: npt.NDArray[ReplayItemID] | ReplayItemID,
        **kwargs: Any,
    ) -> None:
        self._sampling_distribution.update(keys, **kwargs)
