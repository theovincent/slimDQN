# Credits: https://github.com/theovincent/PBO.git
import numpy as np
from slimRL.environments.car_on_hill import CarOnHill
from experiments.car_on_hill.optimal import NX, NV


def compute_state_and_reward_distribution(rb):
    xs = (rb["observations"][:, 0]).reshape(-1)
    vs = (rb["observations"][:, 1]).reshape(-1)
    rewards = rb["rewards"].reshape(-1)

    env = CarOnHill()

    boxes_x_size = (2 * env.max_pos) / (NX - 1)
    discrete_x_boxes = np.linspace(-env.max_pos, env.max_pos + boxes_x_size, NX + 1) - boxes_x_size / 2
    boxes_v_size = (2 * env.max_velocity) / (NV - 1)
    discrete_v_boxes = np.linspace(-env.max_velocity, env.max_velocity + boxes_v_size, NV + 1) - boxes_v_size / 2

    # for each element of dimension one, get the index where it is located in the discrete dimension.
    indexes_x_boxes = np.searchsorted(discrete_x_boxes, xs) - 1
    indexes_v_boxes = np.searchsorted(discrete_v_boxes, vs) - 1

    # only count the element pairs that are in the boxes
    x_inside_boxes = np.logical_and(xs > discrete_x_boxes[0], xs < discrete_x_boxes[-1])
    v_inside_boxes = np.logical_and(vs > discrete_v_boxes[0], vs < discrete_v_boxes[-1])
    xv_inside_boxes = np.logical_and(x_inside_boxes, v_inside_boxes)

    pruned_rewards = rewards[xv_inside_boxes]

    samples_count = np.zeros((NX, NV))
    rewards_count = np.zeros((NX, NV))

    indexes_to_bin = np.vstack((indexes_x_boxes[xv_inside_boxes], indexes_v_boxes[xv_inside_boxes])).T

    for idx_index, (x_bin_index, v_bin_index) in enumerate(indexes_to_bin):
        samples_count[x_bin_index, v_bin_index] += 1
        rewards_count[x_bin_index, v_bin_index] += pruned_rewards[idx_index]

    return samples_count, rewards_count
