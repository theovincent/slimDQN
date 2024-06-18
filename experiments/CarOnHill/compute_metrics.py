import os
import sys
import json
import copy
import argparse
import numpy as np
import jax
import jax.numpy as jnp
import pickle
import multiprocessing
from slimRL.environments.car_on_hill import CarOnHill
from slimRL.networks.architectures.DQN import BasicDQN
from experiments.CarOnHill.sample_utils import count_samples
from slimRL.sample_collection.utils import load_replay_buffer_store
from experiments.CarOnHill.optimal import NX, NV


def evaluate_q_i(
    q: BasicDQN,
    params,
    observations: jnp.ndarray,
    q_array: list,
    idx_iteration: int,
):
    q_array[idx_iteration] = np.array(q.apply(params, observations))


def evaluate_qs(q, evaluation_items, params):
    processes = []
    n_seeds, n_bellman_iterations = len(q_array), len(q_array[0])
    for idx_seed in range(n_seeds):
        for idx_iteration in range(n_bellman_iterations):
            for observations, q_array in evaluation_items:
                processes.append(
                    multiprocessing.Process(
                        target=evaluate_q_i,
                        args=(
                            copy.deepcopy(q),
                            params[idx_seed][idx_iteration],
                            observations,
                            q_array,
                            idx_iteration,
                        ),
                    )
                )
    for process in processes:
        process.start()
    for process in processes:
        process.join()


def evaluate_q_pi_i_s_a(q: BasicDQN, params, state, action, env, horizon, gamma):
    env.reset(state)
    _, reward, absorbing = env.step(action)
    performance = reward
    discount = gamma

    while not absorbing and env.n_steps < horizon:
        action = q.best_action(params, jnp.array(env.state))
        _, reward, absorbing = env.step(action)
        performance += discount * reward
        discount *= gamma

    return performance


def evaluate_q_pi_i(
    q: BasicDQN, params, env, horizon, gamma, q_pi_array, idx_iteration
):
    states_x = np.linspace(-env.max_pos, env.max_pos, NX)
    states_v = np.linspace(-env.max_velocity, env.max_velocity, NV)
    q_pi_i = np.zeros((NX, NV, env.n_actions))
    for idx_state_x, state_x in enumerate(states_x):
        for idx_state_v, state_v in enumerate(states_v):
            for action in range(env.n_actions):
                q_pi_i[idx_state_x, idx_state_v, action] = evaluate_q_pi_i_s_a(
                    q,
                    params,
                    jnp.array([state_x, state_v]),
                    action,
                    env,
                    horizon,
                    gamma,
                )
    q_pi_array[idx_iteration] = q_pi_i.reshape(-1, env.n_actions)
    print(f"Done q_pi_i for i={idx_iteration}")


def evaluate_q_pis(q, params, horizon, gamma, env):
    n_seeds, n_bellman_iterations = len(params), len(params[0])
    manager = multiprocessing.Manager()
    q_pi_array = manager.list(
        list(np.full((n_seeds, n_bellman_iterations, NX * NV, 2), np.nan))
    )
    processes = []

    for idx_seed in range(n_seeds):
        for idx_iteration in range(n_bellman_iterations):
            processes.append(
                multiprocessing.Process(
                    target=evaluate_q_pi_i,
                    args=(
                        copy.deepcopy(q),
                        params[idx_seed][idx_iteration],
                        copy.deepcopy(env),
                        horizon,
                        gamma,
                        q_pi_array,
                        idx_iteration,
                    ),
                )
            )

    for process in processes:
        process.start()

    for process in processes:
        process.join()
    return np.array(q_pi_array)


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
        help="Give all the seed values to generate metrics for (runs for all seeds if not specified)",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-ae",
        "--approximation_error",
        action="store_true",
        help="Give this flag to compute Approximation error components(Q_i and T Q_{i-1}) for every iteration on data and grid",
    )
    parser.add_argument(
        "-perf",
        "--performance",
        action="store_true",
        help="Give this flag to compute Q_pi_i for every iteration on grid",
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
    parameters = json.load(
        open(os.path.join(experiment_folder_path, "parameters.json"), "r")
    )
    q = BasicDQN(
        q_key=jax.random.PRNGKey(0),
        env=env,
        hidden_layers=parameters["hidden_layers"],
        gamma=parameters["gamma"],
        update_horizon=-1,
        lr=-1,
        adam_eps=-1,
        train_frequency=0,
        target_update_frequency=0,
    )
    multiprocessing.set_start_method("spawn", force=True)

    samples_stats, rewards_stats = count_samples(rb)
    np.save(os.path.join(experiment_folder_path, "samples_stats.npy"), samples_stats)
    np.save(os.path.join(experiment_folder_path, "rewards_stats.npy"), rewards_stats)

    states_x = np.linspace(-env.max_pos, env.max_pos, NX)
    states_v = np.linspace(-env.max_velocity, env.max_velocity, NV)
    states_grid = np.array([[x, v] for x in states_x for v in states_v])
    rb_size = rb["observation"].shape[0]

    params = []
    n_bellman_iterations = -1
    if p["seeds"] is None:
        seed_runs = [
            seed_run
            for seed_run in os.listdir(experiment_folder_path)
            if not os.path.isfile(os.path.join(experiment_folder_path, seed_run))
        ]
    else:
        seed_runs = [f"seed={seed}" for seed in p["seeds"]]
    n_seeds = len(seed_runs)

    for idx_seed, seed_run in enumerate(seed_runs):
        params.append([])
        n_seed_iterations = len(
            [
                model_iteration
                for model_iteration in os.listdir(
                    os.path.join(experiment_folder_path, seed_run)
                )
                if "model_iteration" in model_iteration
            ]
        )
        if n_bellman_iterations == -1:
            n_bellman_iterations = n_seed_iterations
        assert (
            n_bellman_iterations == n_seed_iterations
        ), "Required the same number of bellman iterations for all seeds"
        for idx_iteration in range(n_bellman_iterations):
            params[idx_seed].append(
                pickle.load(
                    open(
                        os.path.join(
                            experiment_folder_path,
                            seed_run,
                            f"model_iteration={idx_iteration}",
                        ),
                        "rb",
                    )
                )
            )

    print(
        f"Num seeds = {n_seeds}, Bellman iterations = {n_bellman_iterations}, RB size = {rb_size}"
    )

    if p["approximation_error"]:
        manager = multiprocessing.Manager()
        q_rb = manager.list(
            list(
                np.full((n_seeds, n_bellman_iterations, rb_size, env.n_actions), np.nan)
            )
        )
        q_grid = manager.list(
            list(
                np.full((n_seeds, n_bellman_iterations, NX * NV, env.n_actions), np.nan)
            )
        )

        evaluation_items = [
            (jnp.array(rb["observation"]), q_rb),
            (jnp.array(states_grid), q_grid),
        ]
        evaluate_qs(q, evaluation_items, params)

        for idx_seed, seed_run in enumerate(seed_runs):
            np.save(
                os.path.join(experiment_folder_path, seed_run, "q_i_rb_new.npy"),
                q_rb[idx_seed],
            )
            np.save(
                os.path.join(experiment_folder_path, seed_run, "q_i_grid_new.npy"),
                q_grid[idx_seed],
            )
            t_q_rb = np.zeros((n_bellman_iterations, rb_size))
            for idx_iteration in range(n_bellman_iterations):
                t_q_rb[idx_iteration] = q.compute_target(
                    params=params[idx_seed][idx_iteration],
                    sample=rb,
                )

            np.save(
                os.path.join(experiment_folder_path, seed_run, "T_q_rb_new.npy"), t_q_rb
            )

    if p["performance"]:

        q_pi = evaluate_q_pis(
            q, params, parameters["horizon"], parameters["gamma"], env
        )

        for idx_seed, seed_run in enumerate(seed_runs):
            np.save(
                os.path.join(experiment_folder_path, seed_run, "q_pi_i_new.npy"),
                q_pi[idx_seed],
            )
