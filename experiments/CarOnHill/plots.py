import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from slimRL.environments.car_on_hill import CarOnHill
from slimRL.networks.architectures.DQN import DQNNet
from slimRL.sample_collection.count_samples import count_samples
from slimRL.sample_collection.utils import load_replay_buffer_store
from experiments.CarOnHill.plot_utils import plot_on_grid, plot_value


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

    env = CarOnHill()

    boxes_x_size = (2 * env.max_pos) / (p["n_states_x"] - 1)
    states_x_boxes = (
        np.linspace(-env.max_pos, env.max_pos + boxes_x_size, p["n_states_x"] + 1)
        - boxes_x_size / 2
    )
    boxes_v_size = (2 * env.max_velocity) / (p["n_states_v"] - 1)
    states_v_boxes = (
        np.linspace(
            -env.max_velocity, env.max_velocity + boxes_v_size, p["n_states_v"] + 1
        )
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
    agent,
    observations: torch.Tensor,
):

    agent.load_state_dict(model_wts)
    agent.eval()
    q_estimate[model_key] = agent(observations).detach().numpy()
    print(f"Done {model_key}")


def td_error_plot(argvs=sys.argv[1:]):
    parser = argparse.ArgumentParser("Plot TD error against Bellman iterations.")
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
        required=True,
    )
    parser.add_argument(
        "-nx",
        "--n_states_x",
        help="Discretization for position (x).",
        type=int,
        default=17,
    )
    parser.add_argument(
        "-nv",
        "--n_states_v",
        help="Discretization for velocity (v).",
        type=int,
        default=17,
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
    agent = DQNNet(env)

    q_estimate = dict()

    for model_key, model_wts in models.items():
        evaluate(
            model_key,
            model_wts,
            q_estimate,
            agent,
            torch.Tensor(rb["observation"]),
        )
        evaluate(
            model_key + "_trunc_states",
            model_wts,
            q_estimate,
            agent,
            torch.Tensor(
                np.array([v for k, v in sorted(rb["next_observations_trunc"].items())])
            ),
        )
        evaluate(
            model_key + "_last_transition",
            model_wts,
            q_estimate,
            agent,
            torch.Tensor(rb["last_transition_next_obs"][1]),
        )

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
                            ][np.arange(rb_size), rb["action"]],
                            ord=2,
                        )

        print(td_error[exp])
    plot_value(
        xlabel="Bellman iteration",
        ylabel="$||\Gamma Q_{i-1} - Q_{i}||_2$",
        x_val=range(1, td_error[exp].shape[1] + 1, 1),
        y_val=td_error,
        title="TD error on data",
        ticksize=10,
    )

    boxes_x_size = (2 * env.max_pos) / (p["n_states_x"] - 1)
    states_x_boxes = (
        np.linspace(-env.max_pos, env.max_pos + boxes_x_size, p["n_states_x"] + 1)
        - boxes_x_size / 2
    )
    boxes_v_size = (2 * env.max_velocity) / (p["n_states_v"] - 1)
    states_v_boxes = (
        np.linspace(
            -env.max_velocity, env.max_velocity + boxes_v_size, p["n_states_v"] + 1
        )
        - boxes_v_size / 2
    )
    print(states_x_boxes, states_v_boxes)

    samples_stats, _, _ = count_samples(
        rb["observation"][:, 0],
        rb["observation"][:, 1],
        states_x_boxes,
        states_v_boxes,
        rb["reward"],
    )
    print(samples_stats.shape)
    samples_stats = samples_stats[1:, 1:]
    states_x_boxes = states_x_boxes[1:-1]
    states_v_boxes = states_v_boxes[1:-1]

    samples_stats = samples_stats / samples_stats.sum()
    scaling = np.array(
        [
            samples_stats[i, j]
            for i in range(len(states_x_boxes))
            for j in range(len(states_v_boxes))
        ]
    )
    states_grid = np.array([[i, j] for i in states_x_boxes for j in states_v_boxes])
    next_states_grid = np.zeros(
        (states_grid.shape[0], env.n_actions, states_grid.shape[-1])
    )
    rewards_grid = np.zeros((states_grid.shape[0], env.n_actions))
    dones_grid = np.zeros((states_grid.shape[0], env.n_actions), dtype=int)

    for i in range(states_grid.shape[0]):
        for action in range(env.n_actions):
            env.reset(states_grid[i])
            next_state, reward, done = env.step(action)
            next_states_grid[i, action, :] = next_state
            rewards_grid[i, action] = reward
            dones_grid[i, action] = done

    q_estimate = dict()

    for model_key, model_wts in models.items():
        evaluate(
            model_key,
            model_wts,
            q_estimate,
            agent,
            torch.Tensor(states_grid),
        )
        for action in range(env.n_actions):
            evaluate(
                model_key + f"_next_states_action={action}",
                model_wts,
                q_estimate,
                agent,
                torch.Tensor(next_states_grid[:, action, :]),
            )

    td_error = {}
    for exp, result_path in results_folder.items():
        td_error[exp] = np.zeros((num_seeds, num_bellman_iterations - 1))
        for idx_seed, seed_run in enumerate(os.listdir(result_path)):
            if not os.path.isfile(os.path.join(result_path, seed_run)):
                for idx_iteration in range(num_bellman_iterations):
                    if idx_iteration > 0:
                        T_q = np.zeros((states_grid.shape[0], 2))
                        q = np.zeros((states_grid.shape[0], 2))
                        for action in range(env.n_actions):
                            next_state_q_estimate = q_estimate[
                                f"{exp}/{seed_run}/model_iteration={idx_iteration-1}_next_states_action={action}"
                            ].copy()

                            T_q[:, action] = rewards_grid[:, action] + p[
                                "gamma"
                            ] * np.max(next_state_q_estimate, axis=1) * (
                                1 - dones_grid[:, action]
                            )
                            q[:, action] = q_estimate[
                                f"{exp}/{seed_run}/model_iteration={idx_iteration}"
                            ][:, action]

                        td_error[exp][idx_seed, idx_iteration - 1] = np.linalg.norm(
                            np.multiply(T_q - q, scaling[:, np.newaxis]),
                            ord=2,
                        )

        print(td_error[exp])
    plot_value(
        xlabel="Bellman iteration",
        ylabel="$||\Gamma Q_{i-1} - Q_{i}||_2$",
        x_val=range(1, td_error[exp].shape[1] + 1, 1),
        y_val=td_error,
        title="TD error on grid",
        ticksize=10,
    )
