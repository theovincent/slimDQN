# Credits: https://github.com/theovincent/PBO.git
import numpy as np


def count_samples(
    dimension_one: np.ndarray,
    dimension_two: np.ndarray,
    discrete_dim_one_boxes: np.ndarray,
    discrete_dim_two_boxes: np.ndarray,
    rewards: np.ndarray,
) -> tuple:
    # for each element of dimension one, get the index where it is located in the discrete dimension.
    dimension_one = np.array(dimension_one).reshape(-1)
    indexes_dim_one_boxes = np.searchsorted(discrete_dim_one_boxes, dimension_one) - 1

    # for each element of dimension two, get the index where it is located in the discrete dimension.
    dimension_two = np.array(dimension_two).reshape(-1)
    indexes_dim_two_boxes = np.searchsorted(discrete_dim_two_boxes, dimension_two) - 1

    # only count the element pairs that are in the boxes
    dim_one_inside_boxes = np.logical_and(
        dimension_one > discrete_dim_one_boxes[0],
        dimension_one < discrete_dim_one_boxes[-1],
    )
    dim_two_inside_boxes = np.logical_and(
        dimension_two > discrete_dim_two_boxes[0],
        dimension_two < discrete_dim_two_boxes[-1],
    )
    dimensions_inside_boxes = np.logical_and(dim_one_inside_boxes, dim_two_inside_boxes)

    pruned_rewards = rewards.reshape(-1)[dimensions_inside_boxes]

    samples_count = np.zeros(
        (len(discrete_dim_one_boxes) - 1, len(discrete_dim_two_boxes) - 1)
    )
    rewards_count = np.zeros(
        (len(discrete_dim_one_boxes) - 1, len(discrete_dim_two_boxes) - 1)
    )

    indexes_dim = np.vstack(
        (
            indexes_dim_one_boxes[dimensions_inside_boxes],
            indexes_dim_two_boxes[dimensions_inside_boxes],
        )
    ).T

    for idx_in_list, (idx_dim_one, idx_dim_two) in enumerate(indexes_dim):
        samples_count[idx_dim_one, idx_dim_two] += 1
        rewards_count[idx_dim_one, idx_dim_two] += pruned_rewards[idx_in_list]

    return samples_count, (~dimensions_inside_boxes).sum(), rewards_count
