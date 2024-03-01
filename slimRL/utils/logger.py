import os
import json
import torch
import numpy as np
def save_logs(p, returns, losses, search_param=[], agent_params = []):
    save_path = os.path.join(os.path.abspath(__file__), "../experiments/", p["env"], p["experiment_name"], p["agent"])
    file_suffix = "_".join([f"{arg}={str(p[arg])}" for arg in search_param])
    np.save(os.path.join(save_path, f"returns_{file_suffix}.npz"), returns)
    np.save(os.path.join(save_path, f"losses_{file_suffix}.npz"), returns)

    param_file_loc = os.path.join(os.path.abspath(__file__), "../experiments/", p["env"], p["experiment_name"], "parameters.json")
    old_params = json.loads(open(param_file_loc), "r")
    if f"---- {p["agent"]} ---" not in old_params.keys():
        old_params[p["agent"]] = "-----------------------------"
        # for p in agent_params:
