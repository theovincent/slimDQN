import os
import sys
import json
import argparse
import numpy as np
from experiments.CarOnHill.plot_utils import plot_value


def run(argvs=sys.argv[1:]):
    parser = argparse.ArgumentParser("Plot IQM against epochs.")
    parser.add_argument(
        "-e",
        "--experiment_folders",
        nargs="+",
        help="Give the path to all experiment folders to plot from logs/",
        required=True,
    )

    parser.add_argument(
        "-env",
        "--env",
        help="Environment folder name.",
        type=str,
        required=True,
    )
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

    returns = {}
    parameters = {}
    for result in results_folder:
        experiment = result.split("logs")[-1][1:]
        returns[experiment] = np.array(
            [
                [np.mean(i) for i in json.load(open(os.path.join(result, f), "r"))]
                for f in os.listdir(result)
                if "rewards" in f
            ]
        )
        parameters[experiment] = json.load(
            open(os.path.join(result, "..", "parameters.json"), "r")
        )

    plot_value(
        "Env steps",
        "IQM Total reward",
        np.arange(0, returns[exp].shape[1]).tolist(),
        returns,
        ticksize=25,
        title="Sample Efficiency Curve - LunarLander",
        fontsize=20,
        linewidth=3,
    ).savefig(os.path.join(base_path, p["experiment_folders"][0], f"iqm_sec.pdf"))
