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
from slimRL.sample_collection.utils import load_replay_buffer_store
from experiments.CarOnHill.plot_utils import plot_on_grid
from experiments.base.parser import plot_parser
from experiments.base.utils import confidence_interval


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

    rb = load_replay_buffer_store(rb_path)

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


def evaluate(
    model_key: str,
    model_wts: torch.Tensor,
    q_estimate: dict,
    env,
    observations: torch.Tensor,
):
    agent = DQNNet(env)
    agent.load_state_dict(model_wts)
    agent.eval()
    q_estimate[model_key] = agent(observations).detach().numpy()
    print(f"Done {model_key}")


def td_error_plot(argvs=sys.argv[1:]):
    parser = argparse.ArgumentParser("Plot returns against epochs.")
    parser.add_argument(
        "-e",
        "--experiment_folders",
        nargs="+",
        help="Give the path to all experiment folders to plot from logs/",
        required=True,
    )
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
    parser.add_argument(
        "-gamma",
        "--gamma",
        type=float,
        help="Gamma for TD error",
        required=False,
        default=0.99,
    )
    args = parser.parse_args(argvs)

    p = vars(args)

    base_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../CarOnHill/logs",
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

        rb = load_replay_buffer_store(rb_path)
        rb_size = rb["observation"].shape[0]

        models = {}
        for exp, result_path in results_folder.items():
            for seed_run in os.listdir(result_path):
                if not os.path.isfile(os.path.join(result_path, seed_run)):
                    for iteration in os.listdir(os.path.join(result_path, seed_run)):
                        if "model" in iteration:
                            models[f"{exp}/{seed_run}/{iteration}"] = torch.load(
                                os.path.join(result_path, seed_run, iteration)
                            )

        num_bellman_iterations = len(set(i.split("/")[-1] for i in models.keys()))
        num_seeds = len(set(i.split("/")[-2] for i in models.keys()))
        print(
            f"Bellman iterations = {num_bellman_iterations}, Num seeds = {num_seeds}, RB size = {rb_size}"
        )

        env = CarOnHill()

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
                        env,
                        torch.Tensor(rb["observation"]),
                    ),
                )
            )
        for model_key, model_wts in models.items():
            evaluate(
                model_key + "_trunc_states",
                model_wts,
                q_estimate,
                env,
                torch.Tensor(
                    np.array(
                        [v for k, v in sorted(rb["next_observations_trunc"].items())]
                    )
                ),
            )
            evaluate(
                model_key + "_last_transition",
                model_wts,
                q_estimate,
                env,
                torch.Tensor(rb["last_transition_next_obs"][1]),
            )

        for process in processes:
            process.start()

        for process in processes:
            process.join()

        td_error = {}
        for exp, result_path in results_folder.items():
            td_error[exp] = np.zeros((num_seeds, num_bellman_iterations - 1))
            for idx_seed, seed_run in enumerate(os.listdir(result_path)):
                if not os.path.isfile(os.path.join(result_path, seed_run)):
                    for idx_iteration in range(num_bellman_iterations):
                        if idx_iteration > 0:
                            T_q = q_estimate[
                                f"{exp}/{seed_run}/model_iteration={idx_iteration-1}"
                            ].copy()
                            T_q = np.roll(T_q, -1, axis=0)
                            for idx, (pos, _) in enumerate(
                                sorted(rb["next_observations_trunc"].items())
                            ):
                                T_q[pos] = q_estimate[
                                    f"{exp}/{seed_run}/model_iteration={idx_iteration-1}_trunc_states"
                                ][idx]
                            T_q[rb["last_transition_next_obs"][0]] = q_estimate[
                                f"{exp}/{seed_run}/model_iteration={idx_iteration-1}_last_transition"
                            ]
                            T_q = rb["reward"] + p["gamma"] * np.max(T_q, axis=1) * (
                                1 - rb["done"]
                            )

                            td_error[exp][idx_seed, idx_iteration - 1] = np.linalg.norm(
                                T_q
                                - q_estimate[
                                    f"{exp}/{seed_run}/model_iteration={idx_iteration}"
                                ][np.arange(rb_size), rb["action"]]
                            )
            print(td_error[exp])

        plt.rc("font", size=15, family="serif", serif="Times New Roman")
        plt.rc("lines", linewidth=1)
        fig = plt.figure(f"TD error")
        ax = fig.add_subplot(111)
        plt.xlabel("Bellman iteration")
        plt.ylabel("$||\Gamma Q_{i-1} - Q_{i}||_2$")
        ax.set_xticks(range(1, td_error[exp].shape[1] + 1, 1)[::2])

        for i, exp in enumerate(td_error):
            td_error_mean = td_error[exp].mean(axis=0)
            td_error_std = td_error[exp].std(axis=0)
            td_error_cnf = confidence_interval(
                td_error_mean, td_error_std, td_error[exp].shape[0]
            )
            ax.plot(
                range(1, td_error[exp].shape[1] + 1, 1),
                td_error_mean,
                label=exp,
            )
            ax.fill_between(
                range(1, td_error[exp].shape[1] + 1, 1),
                td_error_cnf[0],
                td_error_cnf[1],
                alpha=0.3,
            )

        plt.legend()
        plt.tight_layout()
        plt.show()
