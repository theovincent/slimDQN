import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from rliable import library as rly
from rliable import metrics
from rliable import plot_utils
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

    returns = {}
    for result in results_folder:
        experiment = result.split("logs")[-1][1:]
        returns[experiment] = np.expand_dims(
            np.array(
                [
                    [np.mean(i) for i in json.load(open(os.path.join(result, f), "r"))]
                    for f in os.listdir(result)
                    if "rewards" in f
                ]
            ),
            axis=1,
        )

    aggregate_scores, aggregate_score_cis = rly.get_interval_estimates(
        returns,
        lambda scores: np.array(
            [
                metrics.aggregate_iqm(scores[..., epoch])
                for epoch in range(scores.shape[-1])
            ]
        ),
        reps=500,
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    n_epochs = max([i.size for i in aggregate_scores.values()])

    plot_utils.plot_sample_efficiency_curve(
        np.arange(1, n_epochs + 1),
        aggregate_scores,
        aggregate_score_cis,
        algorithms=list(returns.keys()),
        xlabel="Epoch",
        ylabel="IQM Average total reward",
        ax=ax,
    )
    plt.legend()
    plt.show()
