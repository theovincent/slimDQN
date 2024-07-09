import os
import sys
import json
import copy
import argparse
import numpy as np
import jax
import jax.numpy as jnp
import multiprocess
from slimRL.environments.car_on_hill import CarOnHill
from slimRL.networks.architectures.DQN import BasicDQN
from experiments.car_on_hill.sample_utils import compute_state_and_reward_distribution
from slimRL.sample_collection.utils import load_valid_transitions
from experiments.base.logger import pickle_load
from experiments.car_on_hill.optimal import NX, NV


def run(argvs=sys.argv[1:]):
    parser = argparse.ArgumentParser(
        "Car-On-Hill FQI - Compute all relevant evaluation metrics."
    )
    parser.add_argument(
        "-e",
        "--experiment_folder",
        type=str,
        help="Give the path to experiment folder to generate metrics for from exp_output/",
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
        "--approximation_error_components",
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

    multiprocess.set_start_method("spawn", force=True)

    experiment_folder_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../car_on_hill/exp_output",
        p["experiment_folder"],
    )

    # ---------Load the model parameters---------
    parameters = json.load(
        open(os.path.join(experiment_folder_path, "parameters.json"), "r")
    )

    # ---------Load the replay buffer and save samples and rewards distribution---------
    rb = load_valid_transitions(
        os.path.join(experiment_folder_path, "replay_buffer.json")
    )
    samples_stats, rewards_stats = compute_state_and_reward_distribution(rb)
    np.save(os.path.join(experiment_folder_path, "samples_stats.npy"), samples_stats)
    np.save(os.path.join(experiment_folder_path, "rewards_stats.npy"), rewards_stats)

    # ---------Extract all the seeds for computing metrics---------
    # ---------(Use all seeds, if not provided explicitly)---------
    if p["seeds"] is None:
        seed_runs = [
            seed_run
            for seed_run in os.listdir(experiment_folder_path)
            if not os.path.isfile(os.path.join(experiment_folder_path, seed_run))
        ]
    else:
        seed_runs = [f"seed_{seed}" for seed in p["seeds"]]

    # ---------Initialize environment and agent---------
    env = CarOnHill()
    q = BasicDQN(
        q_key=jax.random.PRNGKey(0),
        observation_shape=env.observation_shape,
        n_actions=env.n_actions,
        hidden_layers=parameters["hidden_layers"],
        gamma=parameters["gamma"],
        update_horizon=-1,
        lr=-1,
        train_frequency=-1,
        target_update_frequency=-1,
    )

    # ---------Load all the model weights for all seeds X iterations---------
    params_list = []
    for seed_run in seed_runs:
        params_list.append(
            [
                pickle_load(
                    os.path.join(
                        experiment_folder_path,
                        seed_run,
                        f"model_iteration_{idx_iteration}",
                    )
                )["params"]
                for idx_iteration in range(parameters["n_bellman_iterations"] + 1)
            ]
        )

    # ---------Initialize grid to compute metrics---------
    states_grid = np.array(
        [
            [x, v]
            for x in np.linspace(-env.max_pos, env.max_pos, NX)
            for v in np.linspace(-env.max_velocity, env.max_velocity, NV)
        ]
    )
    if p["approximation_error_components"]:
        evaluate_and_save_q_and_tq(
            q, params_list, rb, states_grid, experiment_folder_path, seed_runs
        )

    if p["performance"]:
        evaluate_and_save_q_pis(
            q,
            params_list,
            states_grid,
            parameters["horizon"],
            parameters["gamma"],
            env,
            experiment_folder_path,
            seed_runs,
        )


def evaluate_and_save_q_and_tq(
    q: BasicDQN, params_list, rb, states_grid, experiment_folder_path, seed_runs
):
    def evaluate_q_i(q_apply, params, evaluation_observations, q_i, idx_iteration):
        q_i[idx_iteration] = np.array(q_apply(params, evaluation_observations))

    # ---------Concatenate RB states and grid states to compute Q_i---------
    observations_for_q_i = jnp.concatenate([rb["observations"], states_grid], axis=0)
    processes = []
    n_seeds, n_bellman_iterations = len(params_list), len(params_list[0])
    manager = multiprocess.Manager()
    q_i = manager.dict()
    tq_i = manager.dict()
    for idx_seed in range(n_seeds):
        q_i[idx_seed] = manager.list([np.nan for _ in range(n_bellman_iterations)])
        tq_i[idx_seed] = manager.list(
            manager.list([np.nan for _ in range(n_bellman_iterations)])
        )
        for idx_iteration in range(n_bellman_iterations):
            processes.append(
                multiprocess.Process(
                    target=evaluate_q_i,
                    args=(
                        q.apply,
                        params_list[idx_seed][idx_iteration],
                        observations_for_q_i,
                        q_i[idx_seed],
                        idx_iteration,
                    ),
                )
            )
            processes.append(
                multiprocess.Process(
                    target=evaluate_q_i,
                    args=(
                        q.compute_target,
                        params_list[idx_seed][idx_iteration],
                        rb,
                        tq_i[idx_seed],
                        idx_iteration,
                    ),
                )
            )

    for process in processes:
        process.start()
    for process in processes:
        process.join()

    rb_size = rb["observations"].shape[0]
    for idx_seed, seed_run in enumerate(seed_runs):
        np.save(
            os.path.join(experiment_folder_path, seed_run, "q_rb.npy"),
            np.array(q_i[idx_seed])[:, :rb_size, :],
        )
        np.save(
            os.path.join(experiment_folder_path, seed_run, "q_grid.npy"),
            np.array(q_i[idx_seed])[:, rb_size:, :],
        )
        np.save(
            os.path.join(experiment_folder_path, seed_run, "Tq_rb.npy"),
            np.array(tq_i[idx_seed]),
        )


def evaluate_and_save_q_pis(
    q: BasicDQN,
    params_list,
    states_grid,
    horizon,
    gamma,
    env,
    experiment_folder_path,
    seed_runs,
):
    def evaluate_q_pi_i(
        q_best_action, params, states_grid, env, horizon, gamma, q_pi, idx_iteration
    ):
        def evaluate_q_pi_i_s_a(
            q_best_action, params, state, action, env, horizon, gamma
        ):
            env.reset(state)
            _, reward, absorbing = env.step(action)
            performance = reward
            discount = gamma

            while not absorbing and env.n_steps < horizon:
                action = q_best_action(params, jnp.array(env.state))
                _, reward, absorbing = env.step(action)
                performance += discount * reward
                discount *= gamma

            return performance

        q_pi_i = np.zeros((NX * NV, env.n_actions))
        for idx_state, state in enumerate(states_grid):
            for action in range(env.n_actions):
                q_pi_i[idx_state, action] = evaluate_q_pi_i_s_a(
                    q_best_action,
                    params,
                    jnp.array(state),
                    action,
                    env,
                    horizon,
                    gamma,
                )
        q_pi[idx_iteration] = q_pi_i

    n_seeds, n_bellman_iterations = len(params_list), len(params_list[0])
    manager = multiprocess.Manager()
    q_pi = manager.dict()
    processes = []

    for idx_seed in range(n_seeds):
        q_pi[idx_seed] = manager.list(
            manager.list([np.nan for _ in range(n_bellman_iterations)])
        )
        for idx_iteration in range(n_bellman_iterations):
            processes.append(
                multiprocess.Process(
                    target=evaluate_q_pi_i,
                    args=(
                        q.best_action,
                        params_list[idx_seed][idx_iteration],
                        states_grid,
                        copy.deepcopy(env),
                        horizon,
                        gamma,
                        q_pi[idx_seed],
                        idx_iteration,
                    ),
                )
            )

    for process in processes:
        process.start()
    for process in processes:
        process.join()

    for idx_seed, seed_run in enumerate(seed_runs):
        np.save(
            os.path.join(experiment_folder_path, seed_run, "q_pi.npy"),
            np.array(q_pi[idx_seed]),
        )
