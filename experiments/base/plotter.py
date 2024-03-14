import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from experiments.base.parser import plot_parser
from experiments.base.utils import confidence_interval


def run(argvs=sys.argv[1:]):
    parser = argparse.ArgumentParser("Train DQN on CarOnHill.")
    plot_parser(parser)
    args = parser.parse_args(argvs)

    p = vars(args)

    p["file_path"] = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        p["env"],
        f"logs/{p['experiment_name']}",
    )

    assert os.path.exists(p["file_path"]), f"Required path {p['file_path']} not found"

    for algo in p["agents"]:
        results_folder = os.path.join(p["file_path"], algo)
        assert os.path.exists(
            results_folder
        ), f"{algo} algorithm results folder - {results_folder} not found"

    plt.rc("font", size=28, family="serif", serif="Times New Roman")
    plt.rc("lines", linewidth=3)
    fig = plt.figure(f"Returns for {p['env']}")
    ax = fig.add_subplot(111)
    fig_legend = plt.figure("Legend figure")

    returns = {}
    for algo in p["agents"]:
        results_folder = os.path.join(p["file_path"], algo)
        returns[algo] = np.array(
            [np.load(f) for f in os.listdir(results_folder) if "returns" in f]
        )
        print(f"{algo} --> {returns['algo'].shape}")
    num_epochs = max([ret.shape[1] for ret in returns.items()])
    epochs = range(1, num_epochs + 1)
    num_seeds = returns.items()[0].shape[0]

    for algo in p["agents"]:
        return_mean = returns["algo"].mean(axis=0)
        return_std = returns["algo"].std(axis=0)
        return_cnf = confidence_interval(return_mean, return_std, num_seeds)
        ax.plot(
            epochs,
            return_mean,
            label=algo,
        )
        ax.fill_between(
            epochs,
            return_cnf[0],
            return_cnf[1],
            alpha=0.3,
        )
    plt.show()
