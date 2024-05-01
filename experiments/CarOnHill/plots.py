import os
import sys
import json
import argparse
import numpy as np
import torch
import multiprocessing
import matplotlib.pyplot as plt
from slimRL.environments.car_on_hill import CarOnHill
from slimRL.networks.architectures.DQN import DQNNet
from slimRL.sample_collection.count_samples import count_samples
from experiments.CarOnHill.plot_utils import plot_on_grid, compute_q_on_data
from experiments.base.parser import plot_parser


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


def td_error_plot(argvs=sys.argv[1:]):
    parser = argparse.ArgumentParser("Plot returns against epochs.")
    plot_parser(parser)
    parser.add_argument(
        "-rb",
        "--replay_buffer_path",
        type=str,
        help="Path to replay buffer from logs/ (if plot on data)",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-grid",
        "--grid",
        type=bool,
        help="Plot the TD error on grid",
        required=False,
        default=False,
    )
    args = parser.parse_args(argvs)

    p = vars(args)

    base_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        p["env"],
        "logs",
    )

    assert os.path.exists(base_path), f"Required path {p['file_path']} not found"

    results_folder = {}

    for exp in p["experiment_folders"]:
        exp_folder = os.path.join(base_path, exp)
        assert os.path.exists(exp_folder), f"{exp_folder} not found"
        results_folder[exp] = exp_folder

    if p["replay_buffer_path"] is not None:
        rb_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../CarOnHill/logs",
            p["replay_buffer_path"],
        )

        assert os.path.exists(rb_path), f"Required replay buffer {rb_path} not found"

        rb = {k: np.array(v) for k, v in json.load(open(rb_path, "r")).items()}
        num_states = rb["observation"].shape[0]

    models = {}
    for exp, result_path in results_folder.items():
        for seed_run in os.listdir(result_path):
            if not os.path.isfile(os.path.join(result_path, seed_run)):
                for iteration in os.listdir(os.path.join(result_path, seed_run)):
                    models[f"{exp}/{seed_run}/{iteration}"] = torch.load(
                        os.path.join(result_path, seed_run, iteration)
                    )
    num_bellman_iterations = len(set(i.split("/")[-1] for i in models.keys()))
    num_seeds = len(set(i.split("/")[-2] for i in models.keys()))

    env = CarOnHill()
    agent = DQNNet(env)

    def evaluate(
        model_key: str,
        model_wts: torch.Tensor,
        q_estimate: dict,
        agent: DQNNet,
        observations,
    ):
        agent.load_state_dict(torch.load(model_wts))
        agent.eval()
        q_estimate[model_key] = agent(torch.Tensor(observations)).numpy()

    manager = multiprocessing.Manager()

    q_estimate = manager.dict()

    processes = []
    for model_key, model_wts in models.items():
        processes.append(
            multiprocessing.Process(
                target=evaluate,
                args=(
                    model_key,
                    model_wts,
                    q_estimate,
                    agent,
                    rb["observation"],
                ),
            )
        )

    for process in processes:
        process.start()

    for process in processes:
        process.join()

    td_error = {}
    for exp, result_path in results_folder.items():
        td_error[exp] = np.zeros((num_seeds, num_bellman_iterations))
        for idx_seed in range(num_seeds):
            for idx_iteration in range(1, num_bellman_iterations):
                target = q_estimate
                td_error[exp][i, j] = q_estimate[f"{exp}/{seed_run}/{iteration}"]

    plt.rc("font", size=15, family="serif", serif="Times New Roman")
    plt.rc("lines", linewidth=1)
    fig = plt.figure(f"Returns for {p['env']}")
    ax = fig.add_subplot(111)
    plt.xlabel("Epochs")
    plt.ylabel("Average reward")
    plt.title(f"{p['env']}")
