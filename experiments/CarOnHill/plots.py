import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from slimRL.sample_collection.count_samples import count_samples
from experiments.CarOnHill.plot_utils import plot_on_grid


def samples_plot(argvs=sys.argv[1:]):
    parser = argparse.ArgumentParser("CarOnHill FQI - Sample stats plot.")
    parser.add_argument(
        "-rb",
        "--replay_buffer_path",
        type=str,
        help="Path to replay buffer from logs/",
        required=True,
    )
    parser.add_argument(
        "-nx",
        "--n_states_x",
        type=int,
        help="No. of values to discretize x into",
        required=False,
        default=17,
    )
    parser.add_argument(
        "-nv",
        "--n_states_v",
        type=int,
        help="No. of values to discretize v into",
        required=False,
        default=17,
    )
    args = parser.parse_args(argvs)

    p = vars(args)

    rb_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../CarOnHill/logs",
        p["replay_buffer_path"],
    )

    assert os.path.exists(rb_path), f"Required replay buffer {rb_path} not found"

    rb = {k: np.array(v) for k, v in json.load(open(rb_path, "r")).items()}

    max_pos = 1.0
    max_velocity = 3.0

    boxes_x_size = (2 * max_pos) / (p["n_states_x"] - 1)
    states_x_boxes = (
        np.linspace(-max_pos, max_pos + boxes_x_size, p["n_states_x"] + 1)
        - boxes_x_size / 2
    )
    boxes_v_size = (2 * max_velocity) / (p["n_states_v"] - 1)
    states_v_boxes = (
        np.linspace(-max_velocity, max_velocity + boxes_v_size, p["n_states_v"] + 1)
        - boxes_v_size / 2
    )

    samples_stats, _, rewards_stats = count_samples(
        rb["observation"][:, 0],
        rb["observation"][:, 1],
        states_x_boxes,
        states_v_boxes,
        rb["reward"],
    )

    plot_on_grid(samples_stats, p["n_states_x"], p["n_states_v"], True)
    plot_on_grid(rewards_stats, p["n_states_x"], p["n_states_v"], True)
