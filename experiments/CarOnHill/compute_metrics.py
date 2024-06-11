import os
import sys
import argparse
import numpy as np
import jax.numpy as jnp
import pickle
import multiprocessing
from slimRL.environments.car_on_hill import CarOnHill
from slimRL.networks.architectures.DQN import DQNNet
from slimRL.sample_collection.count_samples import count_samples
from slimRL.sample_collection.utils import load_replay_buffer_store


def evaluate(
    idx_iteration: int,
    model,
    q_estimate: list,
    observations: jnp.ndarray,
):
    env = CarOnHill()
    q_network = DQNNet(
        env,
        model["hidden_layers"],
    )
    q_estimate[idx_iteration] = np.array(q_network.apply(model["params"], observations))
    print(f"Done evaluate {idx_iteration}")


def run_traj(q_network, params, state, action, env, horizon, gamma):
    env.reset(state)
    _, reward, absorbing = env.step(action)
    step = 1
    performance = reward
    discount = gamma

    while not absorbing and step < horizon:
        action = jnp.argmax(q_network.apply(params, jnp.array(env.state))).item()
        _, reward, absorbing = env.step(action)
        performance += discount * reward
        discount *= gamma
        step += 1

    return performance


def compute_iterated_value(
    idx_iteration,
    model,
    states_x,
    states_v,
    iterated_q,
    horizon,
    gamma,
):
    env = CarOnHill()
    q_network = DQNNet(env, model["hidden_layers"])
    for idx_state_x, state_x in enumerate(states_x):
        for idx_state_v, state_v in enumerate(states_v):
            for action in range(env.n_actions):
                iterated_q[idx_iteration][idx_state_x * len(states_v) + idx_state_v][
                    action
                ] = run_traj(
                    q_network,
                    model["params"],
                    jnp.array([state_x, state_v]),
                    action,
                    env,
                    horizon,
                    gamma,
                )
    print(f"Done compute_iterated_value {idx_iteration}")


def run(argvs=sys.argv[1:]):
    parser = argparse.ArgumentParser(
        "CarOnHill FQI - Compute all relevant evaluation metrics."
    )
    parser.add_argument(
        "-e",
        "--experiment_folder",
        type=str,
        help="Give the path to experiment folder to generate metrics for from logs/",
        required=True,
    )
    parser.add_argument(
        "-s",
        "--seeds",
        nargs="+",
        help="Give all the seed values to generate metrics for",
        required=False,
        default=[],
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
    parser.add_argument(
        "-H",
        "--horizon",
        help="Horizon for computing performance loss.",
        type=int,
        default=100,
    )
    args = parser.parse_args(argvs)

    p = vars(args)

    experiment_folder_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../CarOnHill/logs",
        p["experiment_folder"],
    )

    rb = load_replay_buffer_store(
        os.path.join(experiment_folder_path, "replay_buffer.json")
    )

    env = CarOnHill()
    GAMMA = 0.95
    multiprocessing.set_start_method("spawn", force=True)

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
    states_x = np.linspace(-env.max_pos, env.max_pos, p["n_states_x"])
    states_v = np.linspace(-env.max_velocity, env.max_velocity, p["n_states_v"])
    states_grid = np.array([[x, v] for x in states_x for v in states_v])
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

    observations_rb = rb["observation"][
        [
            str(i) not in rb["next_observations_trunc"]
            for i in range(rb["observation"].shape[0])
        ]
    ]
    rewards_rb = rb["reward"][
        [
            str(i) not in rb["next_observations_trunc"]
            for i in range(rb["observation"].shape[0])
        ]
    ]
    dones_rb = rb["done"][
        [
            str(i) not in rb["next_observations_trunc"]
            for i in range(rb["observation"].shape[0])
        ]
    ]

    samples_stats, _, _ = count_samples(
        observations_rb[:, 0],
        observations_rb[:, 1],
        states_x_boxes,
        states_v_boxes,
        rewards_rb,
    )
    scaling = samples_stats / samples_stats.sum()

    args = parser.parse_args(argvs)

    p = vars(args)

    rb_size = observations_rb.shape[0]

    models = {}
    num_bellman_iterations = -1
    num_seeds = 0
    if len(p["seeds"]) == 0:
        seed_runs = [
            seed_run
            for seed_run in os.listdir(experiment_folder_path)
            if not os.path.isfile(os.path.join(experiment_folder_path, seed_run))
        ]
    else:
        seed_runs = [f"seed={seed}" for seed in p["seeds"]]

    for seed_run in seed_runs:
        iteration_runs = [
            model_iteration
            for model_iteration in os.listdir(
                os.path.join(experiment_folder_path, seed_run)
            )
            if "model_iteration" in model_iteration
        ]
        if num_bellman_iterations == -1:
            num_bellman_iterations = len(iteration_runs)
        assert num_bellman_iterations == len(
            iteration_runs
        ), "Required the same number of bellman iterations for all seeds"
        for iteration in iteration_runs:
            models[f"{seed_run}/{iteration}"] = pickle.load(
                open(
                    os.path.join(experiment_folder_path, seed_run, iteration),
                    "rb",
                )
            )

    print(
        f"Num seeds = {num_seeds}, Bellman iterations = {num_bellman_iterations}, RB size = {rb_size}"
    )

    processes = []
    manager = multiprocessing.Manager()
    q_i_rb = manager.dict()
    q_i_grid = manager.dict()
    q_i_grid_next_obs = manager.dict()
    for seed_run in seed_runs:
        q_i_rb[seed_run] = manager.list(
            list(np.nan * np.zeros((num_bellman_iterations, rb_size, 2)))
        )
        q_i_grid[seed_run] = manager.list(
            list(
                np.nan
                * np.zeros(
                    (num_bellman_iterations, p["n_states_x"] * p["n_states_v"], 2)
                )
            )
        )
        for action in range(env.n_actions):
            q_i_grid_next_obs[seed_run + f"action={action}"] = manager.list(
                list(
                    np.nan
                    * np.zeros(
                        (num_bellman_iterations, p["n_states_x"] * p["n_states_v"], 2)
                    )
                )
            )
        for idx_iteration in range(num_bellman_iterations):
            processes.append(
                multiprocessing.Process(
                    target=evaluate,
                    args=(
                        idx_iteration,
                        models[f"{seed_run}/model_iteration={idx_iteration}"],
                        q_i_rb[seed_run],
                        jnp.array(observations_rb),
                    ),
                )
            )
            processes.append(
                multiprocessing.Process(
                    target=evaluate,
                    args=(
                        idx_iteration,
                        models[f"{seed_run}/model_iteration={idx_iteration}"],
                        q_i_grid[seed_run],
                        jnp.array(states_grid),
                    ),
                )
            )
            for action in range(env.n_actions):
                multiprocessing.Process(
                    target=evaluate,
                    args=(
                        idx_iteration,
                        models[f"{seed_run}/model_iteration={idx_iteration}"],
                        q_i_grid_next_obs[seed_run + f"action={action}"],
                        jnp.array(next_states_grid[:, action, :]),
                    ),
                )

    for process in processes:
        process.start()

    for process in processes:
        process.join()

    for seed_run in seed_runs:
        np.save(
            os.path.join(experiment_folder_path, seed_run, "q_i_rb.npy"),
            q_i_rb[seed_run],
        )
        np.save(
            os.path.join(experiment_folder_path, seed_run, "q_i_grid.npy"),
            q_i_grid[seed_run],
        )
        T_q_rb = np.zeros((num_bellman_iterations - 1, rb_size))
        T_q_grid = np.zeros(
            (num_bellman_iterations - 1, p["n_states_x"] * p["n_states_v"], 2)
        )
        for idx_iteration in range(1, num_bellman_iterations):
            T_q_rb_iteration = np.array(q_i_rb[seed_run][idx_iteration - 1])
            T_q_rb_iteration = np.roll(T_q_rb_iteration, -1, axis=0)
            T_q_rb_iteration = rewards_rb + GAMMA * np.max(T_q_rb_iteration, axis=1) * (
                1 - dones_rb
            )
            T_q_rb[idx_iteration - 1] = T_q_rb_iteration
            T_q_grid_iteration = np.stack(
                [
                    q_i_grid_next_obs[seed_run + f"action={action}"][idx_iteration - 1]
                    for action in range(env.n_actions)
                ],
                axis=-1,
            )
            # print(T_q_grid_iteration.shape)
            T_q_grid_iteration = rewards_grid + GAMMA * np.max(
                T_q_grid_iteration, axis=-1
            ) * (1 - dones_grid)
            T_q_grid[idx_iteration - 1] = T_q_grid_iteration
        np.save(os.path.join(experiment_folder_path, seed_run, "T_q_rb.npy"), T_q_rb)
        np.save(
            os.path.join(experiment_folder_path, seed_run, "T_q_grid.npy"), T_q_grid
        )

    processes = []
    manager = multiprocessing.Manager()
    q_pi_i = manager.dict()
    for seed_run in seed_runs:
        for idx_iteration in range(num_bellman_iterations):
            for action in range(env.n_actions):
                q_pi_i[seed_run + f"action={action}"] = manager.list(
                    list(
                        np.nan
                        * np.zeros(
                            (
                                num_bellman_iterations,
                                p["n_states_x"] * p["n_states_v"],
                                2,
                            )
                        )
                    )
                )
                processes.append(
                    multiprocessing.Process(
                        target=compute_iterated_value,
                        args=(
                            idx_iteration,
                            models[f"{seed_run}/model_iteration={idx_iteration}"],
                            states_x,
                            states_v,
                            q_pi_i[seed_run + f"action={action}"],
                            p["horizon"],
                            GAMMA,
                        ),
                    )
                )

    for process in processes:
        process.start()

    for process in processes:
        process.join()

    for seed_run in seed_runs:
        np.save(
            os.path.join(experiment_folder_path, seed_run, "q_pi_i.npy"),
            np.stack(
                [
                    q_pi_i[seed_run + f"action={action}"]
                    for action in range(env.n_actions)
                ],
                axis=-1,
            ),
        )
