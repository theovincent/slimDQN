import os
import sys
import argparse
import numpy as np
import torch
import multiprocessing
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

    plot_on_grid(samples_stats, p["n_states_x"], p["n_states_v"], True).show()
    plot_on_grid(rewards_stats, p["n_states_x"], p["n_states_v"], True).show()


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
        help="Path to replay buffer from logs/",
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

    num_bellman_iterations = len(
        set(i.split("/")[-1] for i in models.keys())
    )  # because the last part of the key of models has the iteration
    num_seeds = len(
        set(i.split("/")[-2] for i in models.keys())
    )  # because the second last part of the key of models has the iteration

    print(
        f"Bellman iterations = {num_bellman_iterations}, Num seeds = {num_seeds}, RB size = {rb_size}"
    )

    env = CarOnHill()
    q_estimate = dict()

    for model_key, model in models.items():
        evaluate(
            model_key,
            model["network"],
            q_estimate,
            DQNNet(env, model["hidden_layers"]),
            torch.Tensor(rb["observation"]),
        )
        if len(rb["next_observations_trunc"]) > 0:
            evaluate(
                model_key
                + "_trunc_next_states",  # evaluate the Q value for next observations for truncated states
                model["network"],
                q_estimate,
                DQNNet(env, model["hidden_layers"]),
                torch.Tensor(
                    np.array(
                        [v for _, v in sorted(rb["next_observations_trunc"].items())]
                    )
                ),
            )
        evaluate(
            model_key
            + "_last_transition",  # evaluate the Q value for next observation for the last recorded transition
            model["network"],
            q_estimate,
            DQNNet(env, model["hidden_layers"]),
            torch.Tensor(rb["last_transition_next_obs"][1]),
        )

    td_error = {}
    for exp, result_path in results_folder.items():
        td_error[exp] = np.zeros((num_seeds, num_bellman_iterations - 1))
        idx_seed = 0
        for seed_run in os.listdir(
            result_path
        ):  # cannot iterate over num_seeds as you need idx_seed and seed_run
            if not os.path.isfile(os.path.join(result_path, seed_run)):
                for idx_iteration in range(1, num_bellman_iterations):
                    T_q = q_estimate[
                        f"{exp}/{seed_run}/model_iteration={idx_iteration-1}"
                    ].copy()
                    T_q = np.roll(T_q, -1, axis=0)
                    for idx, (pos, _) in enumerate(
                        sorted(rb["next_observations_trunc"].items())
                    ):
                        T_q[pos] = q_estimate[
                            f"{exp}/{seed_run}/model_iteration={idx_iteration-1}_trunc_next_states"
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
                    )
                    # exit(1)
                idx_seed += 1

        print(td_error[exp])

    plot_value(
        xlabel="Bellman iteration",
        ylabel="$||\Gamma Q_{i-1} - Q_{i}||_2$",
        x_val=range(1, num_bellman_iterations, 1),
        y_val=td_error,
        title="TD error on replay buffer data",
        ticksize=10,
    ).show()

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

    samples_stats, _, _ = count_samples(
        rb["observation"][:, 0],
        rb["observation"][:, 1],
        states_x_boxes,
        states_v_boxes,
        rb["reward"],
    )
    scaling = samples_stats / samples_stats.sum()

    states_x = np.linspace(-env.max_pos, env.max_pos, p["n_states_x"])
    states_v = np.linspace(-env.max_velocity, env.max_velocity, p["n_states_v"])
    states_grid = np.array([[x, v] for x in states_x for v in states_v])

    scaling = np.array(
        [scaling[i, j] for i in range(p["n_states_x"]) for j in range(p["n_states_v"])]
    )
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

    for model_key, model in models.items():
        evaluate(
            model_key,
            model["network"],
            q_estimate,
            DQNNet(env, model["hidden_layers"]),
            torch.Tensor(states_grid),
        )
        for action in range(env.n_actions):
            evaluate(
                model_key + f"_next_states_action={action}",
                model["network"],
                q_estimate,
                DQNNet(env, model["hidden_layers"]),
                torch.Tensor(next_states_grid[:, action, :]),
            )

    td_error = {}
    for exp, result_path in results_folder.items():
        td_error[exp] = np.zeros((num_seeds, num_bellman_iterations - 1))
        idx_seed = 0
        for seed_run in os.listdir(result_path):
            if not os.path.isfile(os.path.join(result_path, seed_run)):
                for idx_iteration in range(1, num_bellman_iterations):
                    T_q = np.zeros((states_grid.shape[0], env.n_actions))
                    q = np.zeros((states_grid.shape[0], env.n_actions))
                    for action in range(env.n_actions):
                        next_state_q_estimate = q_estimate[
                            f"{exp}/{seed_run}/model_iteration={idx_iteration-1}_next_states_action={action}"
                        ]

                        T_q[:, action] = rewards_grid[:, action] + p["gamma"] * np.max(
                            next_state_q_estimate, axis=1
                        ) * (1 - dones_grid[:, action])
                        q[:, action] = q_estimate[
                            f"{exp}/{seed_run}/model_iteration={idx_iteration}"
                        ][:, action]

                    td_error[exp][idx_seed, idx_iteration - 1] = np.sqrt(
                        np.sum(np.multiply(np.square(T_q - q), scaling[:, np.newaxis]))
                    )
                idx_seed += 1

        print(td_error[exp])
    plot_value(
        xlabel="Bellman iteration",
        ylabel="$||\Gamma Q_{i-1} - Q_{i}||_2$",
        x_val=range(1, td_error[exp].shape[1] + 1, 1),
        y_val=td_error,
        title="TD error on grid",
        ticksize=10,
    ).show()


def diff_from_opt_plot(argvs=sys.argv[1:]):
    parser = argparse.ArgumentParser(
        "Plot difference from optimal value against Bellman iterations."
    )
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
        help="Path to replay buffer from logs/",
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

    samples_stats, _, _ = count_samples(
        rb["observation"][:, 0],
        rb["observation"][:, 1],
        states_x_boxes,
        states_v_boxes,
        rb["reward"],
    )
    scaling = samples_stats / samples_stats.sum()

    states_x = np.linspace(-env.max_pos, env.max_pos, p["n_states_x"])
    states_v = np.linspace(-env.max_velocity, env.max_velocity, p["n_states_v"])
    states_grid = np.array([[x, v] for x in states_x for v in states_v])

    scaling = np.array(
        [scaling[i, j] for i in range(p["n_states_x"]) for j in range(p["n_states_v"])]
    )

    q_estimate = dict()

    for model_key, model in models.items():
        evaluate(
            model_key,
            model["network"],
            q_estimate,
            DQNNet(env, model["hidden_layers"]),
            torch.Tensor(states_grid),
        )

    opt_q = np.load(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f"../CarOnHill/logs/Q_nx={p['n_states_x']}_nv={p['n_states_v']}.npy",
        )
    )
    opt_q = np.array(
        [opt_q[i, j] for i in range(p["n_states_x"]) for j in range(p["n_states_v"])]
    )

    opt_v = np.load(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f"../CarOnHill/logs/V_nx={p['n_states_x']}_nv={p['n_states_v']}.npy",
        )
    )
    opt_v = np.array(
        [opt_v[i, j] for i in range(p["n_states_x"]) for j in range(p["n_states_v"])]
    )

    opt_gap_q = {}
    opt_gap_v = {}
    for exp, result_path in results_folder.items():
        opt_gap_q[exp] = np.zeros((num_seeds, num_bellman_iterations))
        opt_gap_v[exp] = np.zeros((num_seeds, num_bellman_iterations))
        idx_seed = 0
        for seed_run in os.listdir(result_path):
            if not os.path.isfile(os.path.join(result_path, seed_run)):
                for idx_iteration in range(num_bellman_iterations):
                    q = q_estimate[f"{exp}/{seed_run}/model_iteration={idx_iteration}"]
                    v = np.max(q, axis=-1)
                    opt_gap_q[exp][idx_seed, idx_iteration] = np.sqrt(
                        np.sum(
                            np.multiply(
                                np.square(opt_q - q),
                                scaling[:, np.newaxis],
                            )
                        )
                    )

                    opt_gap_v[exp][idx_seed, idx_iteration] = np.sqrt(
                        np.sum(
                            np.multiply(
                                np.square(opt_v - v),
                                scaling,
                            )
                        )
                    )
                idx_seed += 1

        print(opt_gap_q[exp], opt_gap_v[exp])

    plot_value(
        xlabel="Bellman iteration",
        ylabel="$|| Q^{*} - Q_{i}||_2$",
        x_val=range(1, num_bellman_iterations + 1, 1),
        y_val=opt_gap_q,
        title="Difference from optimal Q on grid",
        ticksize=10,
    ).show()
    plot_value(
        xlabel="Bellman iteration",
        ylabel="$|| V^{*} - V_{i}||_2$",
        x_val=range(1, num_bellman_iterations + 1, 1),
        y_val=opt_gap_v,
        title="Difference from optimal V on grid",
        ticksize=10,
    ).show()


def run_traj(agent, state, action, env, horizon, gamma):
    env.reset(state)
    _, reward, absorbing = env.step(action)
    step = 1
    performance = reward
    discount = gamma

    while not absorbing and step < horizon:
        action = np.argmax(agent(torch.Tensor(env.state)).detach().numpy())
        _, reward, absorbing = env.step(action)
        performance += discount * reward
        discount *= gamma
        step += 1

    return performance


def compute_iterated_value(
    model_key,
    model,
    states_x,
    states_v,
    iterated_q_shared,
    horizon,
    gamma,
):
    env = CarOnHill()
    agent = DQNNet(env, model["hidden_layers"])
    agent.load_state_dict(model["network"])
    agent.eval()
    for idx_state_x, state_x in enumerate(states_x):
        for idx_state_v, state_v in enumerate(states_v):
            for action in range(env.n_actions):
                iterated_q_shared[(model_key, idx_state_x, idx_state_v, action)] = (
                    run_traj(
                        agent, np.array([state_x, state_v]), action, env, horizon, gamma
                    )
                )
    print(f"Parallel Done with {model_key}")


def plot_iterated_values(argvs=sys.argv[1:]):
    parser = argparse.ArgumentParser(
        "Plot difference of Q_pi_i from optimal value and Q_i against Bellman iterations."
    )
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
        help="Path to replay buffer from logs/",
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
        "-H",
        "--horizon",
        help="Horizon.",
        type=int,
        default=100,
    )
    parser.add_argument(
        "-gamma",
        "--gamma",
        help="Gamma.",
        type=float,
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

    samples_stats, _, _ = count_samples(
        rb["observation"][:, 0],
        rb["observation"][:, 1],
        states_x_boxes,
        states_v_boxes,
        rb["reward"],
    )
    scaling = samples_stats / samples_stats.sum()

    states_x = np.linspace(-env.max_pos, env.max_pos, p["n_states_x"])
    states_v = np.linspace(-env.max_velocity, env.max_velocity, p["n_states_v"])
    states_grid = np.array([[x, v] for x in states_x for v in states_v])

    scaling = np.array(
        [scaling[i, j] for i in range(p["n_states_x"]) for j in range(p["n_states_v"])]
    )

    q_estimate = dict()

    for model_key, model in models.items():
        evaluate(
            model_key,
            model["network"],
            q_estimate,
            DQNNet(env, model["hidden_layers"]),
            torch.Tensor(states_grid),
        )

    manager = multiprocessing.Manager()

    iterated_q_shared = manager.dict()

    processes = []
    for model_key, model in models.items():
        processes.append(
            multiprocessing.Process(
                target=compute_iterated_value,
                args=(
                    model_key,
                    model,
                    states_x,
                    states_v,
                    iterated_q_shared,
                    p["horizon"],
                    p["gamma"],
                ),
            )
        )

    num_processes = 32
    for i in range(int(np.ceil(len(processes) / float(num_processes)))):
        proc_list = processes[
            i * num_processes : min((i + 1) * num_processes, len(processes))
        ]
        for process in proc_list:
            process.start()

        for process in proc_list:
            process.join()

    iterated_q = {}
    for model_key in models.keys():
        iterated_q[model_key] = np.zeros((len(states_x), len(states_v), env.n_actions))
        for idx_state_x, _ in enumerate(states_x):
            for idx_state_v, _ in enumerate(states_v):
                for action in range(env.n_actions):
                    iterated_q[model_key][idx_state_x, idx_state_v, action] = (
                        iterated_q_shared[(model_key, idx_state_x, idx_state_v, action)]
                    )

    opt_q = np.load(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../CarOnHill/logs/Q_nx=17_nv=17.npy",
        )
    )
    opt_q = np.array(
        [opt_q[i, j] for i in range(p["n_states_x"]) for j in range(p["n_states_v"])]
    )

    opt_v = np.load(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../CarOnHill/logs/V_nx=17_nv=17.npy",
        )
    )
    opt_v = np.array(
        [opt_v[i, j] for i in range(p["n_states_x"]) for j in range(p["n_states_v"])]
    )

    opt_gap_q = {}
    opt_gap_v = {}
    for exp, result_path in results_folder.items():
        opt_gap_q[exp] = np.zeros((num_seeds, num_bellman_iterations))
        opt_gap_v[exp] = np.zeros((num_seeds, num_bellman_iterations))
        idx_seed = 0
        for seed_run in os.listdir(result_path):
            if not os.path.isfile(os.path.join(result_path, seed_run)):
                for idx_iteration in range(num_bellman_iterations):
                    q_i = q_estimate[
                        f"{exp}/{seed_run}/model_iteration={idx_iteration}"
                    ]
                    q_pi_i = iterated_q[
                        f"{exp}/{seed_run}/model_iteration={idx_iteration}"
                    ]
                    q_pi_i = np.array(
                        [
                            q_pi_i[i, j]
                            for i in range(p["n_states_x"])
                            for j in range(p["n_states_v"])
                        ]
                    )
                    v_pi_i = q_pi_i[np.arange(q_pi_i.shape[0]), np.argmax(q_i, axis=-1)]
                    opt_gap_q[exp][idx_seed, idx_iteration] = np.sqrt(
                        np.sum(
                            np.multiply(
                                np.square(opt_q - q_pi_i),
                                scaling[:, np.newaxis],
                            )
                        )
                    )

                    opt_gap_v[exp][idx_seed, idx_iteration] = np.sqrt(
                        np.sum(
                            np.multiply(
                                np.square(opt_v - v_pi_i),
                                scaling,
                            )
                        )
                    )
                idx_seed += 1

        print(opt_gap_q[exp], opt_gap_v[exp])
        np.save(f"{results_folder[exp]}/Iter_Q_opt_diff.npy", opt_gap_q[exp])
        np.save(f"{results_folder[exp]}/Iter_V_opt_diff.npy", opt_gap_v[exp])

    plot_value(
        xlabel="Bellman iteration",
        ylabel="$|| Q^{*} - Q^{\pi_i}||_2$",
        x_val=range(1, num_bellman_iterations + 1, 1),
        y_val=opt_gap_q,
        title="Difference of Q_pi from optimal Q on grid",
        ticksize=10,
    ).show()
    plot_value(
        xlabel="Bellman iteration",
        ylabel="$|| V^{*} - V^{\pi_i}||_2$",
        x_val=range(1, num_bellman_iterations + 1, 1),
        y_val=opt_gap_v,
        title="Difference of V_pi from optimal V on grid",
        ticksize=10,
    ).show()


def plot_policy(argvs=sys.argv[1:]):
    parser = argparse.ArgumentParser("Plot policy on grid for a given model.")
    parser.add_argument(
        "-e",
        "--experiment_folder",
        help="Give the path to experiment folders from logs/ to plot policy",
        type=str,
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
    args = parser.parse_args(argvs)
    p = vars(args)

    for i in range(30):
        p["bellman_iteration"] = i
        base_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../CarOnHill/logs",
        )

        exp_folder = os.path.join(base_path, p["experiment_folder"])
        assert os.path.exists(exp_folder), f"{exp_folder} not found"

        models = {}
        num_seeds = 0
        for seed_run in os.listdir(exp_folder):
            if not os.path.isfile(os.path.join(exp_folder, seed_run)):
                num_seeds += 1
                models[seed_run] = torch.load(
                    os.path.join(
                        exp_folder,
                        seed_run,
                        f"model_iteration={p['bellman_iteration']}",
                    )
                )

        env = CarOnHill()
        states_x = np.linspace(-env.max_pos, env.max_pos, p["n_states_x"])
        states_v = np.linspace(-env.max_velocity, env.max_velocity, p["n_states_v"])
        states_grid = np.array([[x, v] for x in states_x for v in states_v])

        q_estimate = dict()

        for key, model in models.items():
            evaluate(
                key,
                model["network"],
                q_estimate,
                DQNNet(env, model["hidden_layers"]),
                torch.Tensor(states_grid),
            )

        policy = np.zeros((num_seeds, states_grid.shape[0]))
        for idx, (key, model) in enumerate(models.items()):
            policy[idx] = (q_estimate[key][:, 1] > q_estimate[key][:, 0]).astype(float)

        policy = np.mean(policy, axis=0)
        policy = 2 * (policy - np.min(policy)) / (np.max(policy) - np.min(policy)) - 1
        policy = policy.reshape(p["n_states_x"], p["n_states_v"])
        plt = plot_on_grid(policy, policy.shape[0], policy.shape[1], cmap="PRGn")
        plt.savefig(f"/Users/yogeshtripathi/policy={p['bellman_iteration']}.png")
