import os
import sys
import json
import argparse
import numpy as np
from experiments.car_on_hill.plot_utils import plot_value


def run(argvs=sys.argv[1:]):
    parser = argparse.ArgumentParser("Plot IQM against epochs.")
    parser.add_argument(
        "-e",
        "--experiment_folders",
        nargs="+",
        help="Give the path to all experiment folders to plot from exp_output/",
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
        "exp_output",
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
        experiment = result.split("exp_output")[-1][1:]
        returns[experiment] = np.array(
            [
                [np.mean(i) for i in json.load(open(os.path.join(result, f), "r"))]
                for f in os.listdir(result)
                if "rewards" in f
            ]
        )
        parameters[experiment] = json.load(open(os.path.join(result, "..", "parameters.json"), "r"))

    env_steps = parameters[experiment]["n_epochs"] * parameters[experiment]["n_training_steps_per_epoch"]
    plot_value(
        xlabel="Env steps",
        ylabel="IQM Total reward",
        x_val=np.arange(
            0,
            env_steps,
            parameters[experiment]["n_training_steps_per_epoch"],
        ).tolist(),
        y_val=returns,
        xlim=(
            0,
            env_steps,
        ),
        xticks=[0]
        + [
            idx * 10 ** (int(np.log10(env_steps)))
            for idx in range(1, int(np.ceil(env_steps / 10 ** (int(np.log10(env_steps))))) + 1)
        ],
        ticksize=25,
        title=f"{parameters[experiment]['env']}",
        fontsize=20,
        linewidth=3,
        sci_x=True,
    ).savefig(os.path.join(base_path, p["experiment_folders"][0], "performances.pdf"))
    print(
        f"Performance plot saved in {os.path.abspath(os.path.join(base_path, p['experiment_folders'][0], 'performances.pdf'))}"
    )
