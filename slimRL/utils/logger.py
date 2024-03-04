import os
import json
import torch
import numpy as np


def save_logs(p, returns, losses, agent):
    save_path = os.path.join(
        os.path.abspath(__file__),
        "../experiments/",
        p["env"],
        p["experiment_name"],
        p["agent"],
    )
    np.save(os.path.join(save_path, "returns_seed=" + str(p["seed"]) + ".npz"), returns)
    np.save(os.path.join(save_path, "losses_seed=" + str(p["seed"]) + ".npz"), returns)
    torch.save(agent, "model_seed=" + str(p["seed"]))

    # param_file_loc = os.path.join(os.path.abspath(__file__), "../experiments/", p["env"], p["experiment_name"], "parameters.json")
    # try:
    #     old_params = json.loads(open(param_file_loc), "r")
    #     old_params[f"---- {p["agent"]} ---"] = "-----------------------------"
    #     for p in :
    # except FileNotFoundError:
