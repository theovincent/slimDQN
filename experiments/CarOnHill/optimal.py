# Credits: https://github.com/theovincent/PBO.git

import sys
import argparse
import numpy as np
from slimRL.environments.solvers.car_on_hill import compute_optimal_values
from experiments.CarOnHill.plot_utils import plot_on_grid


def run(argvs=sys.argv[1:]):
    parser = argparse.ArgumentParser("Find optimal values for CarOnHill.")
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
        help="Horizon for truncation.",
        type=int,
        default=100,
    )
    parser.add_argument(
        "-gamma",
        "--gamma",
        help="Discounting.",
        type=float,
        default=0.99,
    )
    parser.add_argument(
        "-par",
        "--num_parallel_processes",
        help="No. of parallel processes to run at a time.",
        type=int,
        default=4,
    )
    args = parser.parse_args(argvs)
    p = vars(args)

    optimal_v, optimal_q = compute_optimal_values(
        p["n_states_x"],
        p["n_states_v"],
        p["horizon"],
        p["gamma"],
        p["num_parallel_processes"],
    )

    np.save(f"logs/V_nx={p['n_states_x']}_nv={p['n_states_v']}.npy", optimal_v)
    np.save(f"logs/Q_nx={p['n_states_x']}_nv={p['n_states_v']}.npy", optimal_q)

    plot_on_grid(
        optimal_v,
        p["n_states_x"],
        p["n_states_v"],
    )
