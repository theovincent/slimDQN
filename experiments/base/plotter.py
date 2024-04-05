import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from experiments.base.parser import plot_parser
from experiments.base.utils import confidence_interval


def run(argvs=sys.argv[1:]):
    parser = argparse.ArgumentParser("Plot returns against epochs.")
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

    plt.rc("font", size=15, family="serif", serif="Times New Roman")
    plt.rc("lines", linewidth=1)
    fig = plt.figure(f"Returns for {p['env']}")
    ax = fig.add_subplot(111)
    plt.xlabel("Epochs")
    plt.ylabel("Average reward")
    plt.title(f"{p['env']}")

    returns = {}
    for result in results_folder:
        experiment = result.split("logs")[-1][1:]
        returns[experiment] = np.array(
            [
                np.load(os.path.join(result, f))
                for f in os.listdir(result)
                if "returns" in f
            ]
        )
    num_epochs = [ret.shape[1] for ret in returns.values()]
    num_seeds = [ret.shape[0] for ret in returns.values()]

    for i, exp in enumerate(returns):
        return_mean = returns[exp].mean(axis=0)
        return_std = returns[exp].std(axis=0)
        return_cnf = confidence_interval(return_mean, return_std, num_seeds[i])
        ax.plot(
            range(1, num_epochs[i] + 1, 1),
            return_mean,
            label=exp,
        )
        ax.fill_between(
            range(1, num_epochs[i] + 1, 1),
            return_cnf[0],
            return_cnf[1],
            alpha=0.3,
        )
    plt.legend()
    plt.tight_layout()
    plt.show()
