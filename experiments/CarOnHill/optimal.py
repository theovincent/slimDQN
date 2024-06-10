# Credits: https://github.com/theovincent/PBO.git

import os
import sys
import time
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
        default=0.95,
    )
    args = parser.parse_args(argvs)
    p = vars(args)

    t1 = time.time()
    optimal_v, optimal_q = compute_optimal_values(
        p["n_states_x"],
        p["n_states_v"],
        p["horizon"],
        p["gamma"],
    )
    t2 = time.time()
    print("Time taken (mins) = ", (t2 - t1) / 60)

    save_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../CarOnHill/logs",
    )
    np.save(f"{save_path}/V*_nx={p['n_states_x']}_nv={p['n_states_v']}.npy", optimal_v)
    np.save(f"{save_path}/Q*_nx={p['n_states_x']}_nv={p['n_states_v']}.npy", optimal_q)


def plot_optimal_q(argvs=sys.argv[1:]):
    parser = argparse.ArgumentParser("Plot optimal values for CarOnHill.")
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

    save_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../CarOnHill/logs",
    )
    optimal_v = np.load(f"{save_path}/V_nx={p['n_states_x']}_nv={p['n_states_v']}.npy")
    optimal_q = np.load(f"{save_path}/Q_nx={p['n_states_x']}_nv={p['n_states_v']}.npy")

    plot_on_grid(
        optimal_v,
        optimal_v.shape[0],
        optimal_v.shape[1],
    ).savefig(
        os.path.join(
            save_path, "plots", f"V*_nx={p['n_states_x']}_nv={p['n_states_v']}.png"
        )
    )
    plot_on_grid(
        optimal_q[:, :, 0],
        optimal_v.shape[0],
        optimal_v.shape[1],
    ).savefig(
        os.path.join(
            save_path,
            "plots",
            f"Q*(left)_nx={p['n_states_x']}_nv={p['n_states_v']}.png",
        )
    )
    plot_on_grid(
        optimal_q[:, :, 1],
        optimal_v.shape[0],
        optimal_v.shape[1],
    ).savefig(
        os.path.join(
            save_path,
            "plots",
            f"Q*(right)_nx={p['n_states_x']}_nv={p['n_states_v']}.png",
        )
    )
