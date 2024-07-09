# Credits: https://github.com/theovincent/PBO.git

import os
import sys
import argparse
import numpy as np
from slimRL.environments.solvers.car_on_hill import compute_optimal_values

NX, NV = 17, 17


def run(argvs=sys.argv[1:]):
    parser = argparse.ArgumentParser("Find optimal values for Car-On-Hill.")
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

    optimal_v, optimal_q = compute_optimal_values(
        NX,
        NV,
        p["horizon"],
        p["gamma"],
    )

    save_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../car_on_hill",
    )
    np.save(f"{save_path}/V*.npy", optimal_v)
    np.save(f"{save_path}/Q*.npy", optimal_q)
