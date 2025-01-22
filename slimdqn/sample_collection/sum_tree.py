# Inspired by dopamine implementation: https://github.com/google/dopamine/blob/master/dopamine/jax/replay_memory/sum_tree.py
"""Sum Tree."""

import numpy as np
import numpy.typing as npt


class SumTree:
    """A vectorized sum tree in numpy."""

    def __init__(self, capacity: int) -> None:
        assert capacity > 0, "Capacity to sum tree must be positive."
        self._capacity = capacity
        self._depth = int(np.ceil(np.log2(capacity))) + 1

        self._first_leaf_offset = (2 ** (self._depth - 1)) - 1
        self._nodes = np.zeros((2**self._depth) - 1, dtype=np.float64)
        self.max_recorded_priority = 1.0

    def set(
        self,
        indices: "npt.NDArray[np.int_] | int",
        values: "npt.NDArray[np.float64] | float",
    ) -> None:
        """Set the value at a given leaf node index."""
        if isinstance(indices, (int, np.integer)):
            indices = np.asarray([indices], np.int32)
        if isinstance(values, (int, float, np.floating)):
            values = np.asarray([values], np.float64)
        assert indices.shape == values.shape, "Indices and values must have the same shape."
        assert (values >= 0.0).all(), "Values must be positive."
        self.max_recorded_priority = max(self.max_recorded_priority, max(values))
        node_indices = self._first_leaf_offset + indices
        delta_values = values - self._nodes[node_indices]

        # De-duplicate indices, otherwise this can result in delta_values being
        # applied multiple times to the same index, which results in the incorrect
        # priorities being set.
        node_indices, unique_idx = np.unique(node_indices, return_index=True)
        delta_values = delta_values[unique_idx]
        for _ in reversed(range(self._depth - 1)):
            # Use np.add.at which will accumulate duplicate indices
            np.add.at(self._nodes, node_indices, delta_values)
            node_indices = (node_indices - 1) // 2

        assert (node_indices == 0).all(), f"Sum tree traversal failed with {node_indices}."
        np.add.at(self._nodes, node_indices, delta_values)

    def get(self, index: "npt.NDArray[np.int_] | int") -> "npt.NDArray[np.float64] | float":
        """Get the value at a  given leaf node index."""
        return self._nodes[self._first_leaf_offset + index]

    @property
    def root(self) -> float:
        """The root value (total sum) of the sum tree."""
        return self._nodes[0]

    def query(self, targets: "npt.NDArray[np.float64] | float") -> "npt.NDArray[np.int_] | int":
        """Find the smallest index where target < cumulative value up to index.

        This functions like the CDF for a multi-nomial distribution allowing us
        to sample from this distribution through inverse CDF sampling.

        Args:
          targets: a numpy array or float of query values. Targets must be in the
            range [0, `root`] where `root` is the root value of the sum tree.

        Returns:
          either a numpy array of indices or a single index depending on the query.
        """
        if isinstance(targets, (int, float)):
            targets = np.asarray([targets], np.float64)
        if not ((targets >= 0) & (targets < self.root)).all():
            raise ValueError(f"Targets must be in the interval [0.0, {self.root}).")

        # We'll traverse the tree for all indices at once using masking
        node_indices = np.zeros_like(targets, dtype=np.int32)
        # While we're still traversing
        while (traversal_mask := node_indices < self._first_leaf_offset).any():
            # Our intermediate targets should always be less than intermediate nodes
            assert (targets < self._nodes[node_indices]).all()
            left_node_indices = 2 * node_indices + 1
            left_node_sums = self._nodes[left_node_indices]
            right_node_indices = left_node_indices + 1

            # Traverse the tree but only if that index isn't masked
            node_indices = np.where(
                traversal_mask,
                np.where(
                    targets < left_node_sums,
                    left_node_indices,
                    right_node_indices,
                ),
                node_indices,
            )
            targets = np.where(
                targets < left_node_sums,
                targets,
                targets - left_node_sums,
            )

        return node_indices - self._first_leaf_offset
