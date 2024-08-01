import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from experiments.base.iqm import get_iqm_and_conf_parallel
from experiments.base import DISPLAY_NAME


def run(argvs=sys.argv[1:]):
    parser = argparse.ArgumentParser("Plot IQM against epochs.")
    parser.add_argument(
        "-e",
        "--experiment_folders",
        nargs="+",
        help="Give the path to all experiment folders to plot from exp_output/",
        required=True,
    )

    parser.add_argument("-env", "--env", help="Environment folder name.", type=str, required=True)
    args = parser.parse_args(argvs)

    p = vars(args)

    base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", p["env"], "exp_output")

    results_folder = []

    for exp in p["experiment_folders"]:
        exp_folder = os.path.join(base_path, exp)
        assert os.path.exists(exp_folder), f"{exp_folder} not found"
        results_folder.append(exp_folder)

    returns = {}
    parameters = {}
    for result in results_folder:
        experiment = result.split("exp_output")[-1][1:]
        returns[experiment] = np.array(
            [
                [
                    np.mean(i)
                    for i in json.load(open(os.path.join(result, "episode_returns_and_lengths", f), "r"))[
                        "episode_returns"
                    ]
                ]
                for f in os.listdir(os.path.join(result, "episode_returns_and_lengths"))
            ]
        )
        parameters[experiment] = json.load(open(os.path.join(result, "..", "parameters.json"), "r"))

    env_steps = parameters[experiment]["n_epochs"] * parameters[experiment]["n_training_steps_per_epoch"]
    plot_value(
        xlabel="Env steps",
        ylabel="IQM Return",
        x_val=np.arange(0, env_steps + 1, parameters[experiment]["n_training_steps_per_epoch"]).tolist()[1:],
        y_val=returns,
        xlim=(0, env_steps),
        xticks=np.arange(0, env_steps + 1, 1e5),
        ticksize=25,
        title=f"{DISPLAY_NAME[p['env']]}",
        fontsize=20,
        linewidth=3,
        sci_x=True,
    ).savefig(os.path.join(base_path, p["experiment_folders"][0], "performances.pdf"))
    print(
        f"Performance plot with {returns[experiment].shape[0]} seeds saved in {os.path.abspath(os.path.join(base_path, p['experiment_folders'][0], 'performances.pdf'))}"
    )


def plot_value(xlabel, ylabel, x_val, y_val, xlim, xticks, **kwargs):
    plt.rc("font", size=kwargs.get("fontsize", 15), family="serif", serif="Times New Roman")
    plt.rc("lines", linewidth=kwargs.get("linewidth", 4))
    fig = plt.figure(kwargs.get("title", ""))
    ax = fig.add_subplot(111)
    plt.xlabel(xlabel, fontsize=kwargs.get("fontsize", 15))
    plt.ylabel(ylabel, fontsize=kwargs.get("fontsize", 15))
    plt.title(kwargs.get("title", ""))
    ax.set_xticks(xticks)
    ax.set_xlim(xlim)
    if kwargs.get("yticks", None) is not None:
        ax.set_yticks(kwargs.get("yticks"))
    if kwargs.get("ylim", None) is not None:
        ax.set_ylim(kwargs.get("ylim"))

    for exp in y_val:
        y_iqm, y_cnf = get_iqm_and_conf_parallel(y_val[exp])
        ax.plot(x_val, y_iqm, label=exp)
        ax.fill_between(x_val, y_cnf[0], y_cnf[1], alpha=0.3)

    if kwargs.get("sci_x", False):
        plt.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    if kwargs.get("sci_y", False):
        plt.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    plt.legend(loc="lower right")
    plt.grid()
    plt.tight_layout()

    return plt
