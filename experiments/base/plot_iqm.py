import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from experiments.base.iqm import get_iqm_and_conf_parallel
from experiments.base.parser import plot_parser


def run(argvs=sys.argv[1:]):
    parser = argparse.ArgumentParser("Plot IQM against epochs.")
    plot_parser(parser)
    args = parser.parse_args(argvs)

    p = vars(args)

    base_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        p["env"],
        "logs",
    )

    assert os.path.exists(base_path), f"Required path {p['file_path']} not found"

    results_folder = []

    for exp in p["experiment_folders"]:
        exp_folder = os.path.join(base_path, exp)
        assert os.path.exists(exp_folder), f"{exp_folder} not found"
        results_folder.append(exp_folder)

    plt.rc("font", size=10, family="serif", serif="Times New Roman")
    plt.rc("lines", linewidth=1)
    fig = plt.figure(f"Returns for {p['env']}")
    ax = fig.add_subplot(111)
    plt.xlabel("Epochs")
    plt.ylabel("IQM total reward")
    plt.title(f"{p['env']}")

    returns = {}
    for result in results_folder:
        experiment = result.split("logs")[-1][1:]
        returns[experiment] = np.array(
            [
                [np.mean(i) for i in json.load(open(os.path.join(result, f), "r"))]
                for f in os.listdir(result)
                if "rewards" in f
            ]
        )

    for exp in returns:
        iqm, iqm_ci = get_iqm_and_conf_parallel(returns[exp])
        ax.plot(
            range(1, returns[exp].shape[1] + 1, 1),
            iqm,
            label=exp,
        )
        ax.fill_between(
            range(1, returns[exp].shape[1] + 1, 1),
            iqm_ci[0],
            iqm_ci[1],
            alpha=0.3,
        )
    plt.legend()
    plt.tight_layout()
    plt.show()
