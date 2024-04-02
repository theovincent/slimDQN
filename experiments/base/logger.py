import os
import json
import numpy as np
import torch
from slimRL.networks.architectures.DQN import BasicDQN

SHARED_PARAMS = [
    "experiment_name",
    "env",
    "replay_capacity",
    "batch_size",
    "update_horizon",
    "gamma",
    "lr",
    "target_update_period",
    "n_initial_samples",
    "horizon",
]

AGENT_PARAMS = {
    "DQN": [
        "n_epochs",
        "n_training_steps_per_epoch",
        "update_to_data",
        "end_epsilon",
        "duration_epsilon",
    ]
}


def check_experiment(p: dict):

    returns_path = os.path.join(
        p["save_path"], "returns_seed=" + str(p["seed"]) + ".npz"
    )
    losses_path = os.path.join(p["save_path"], "losses_seed=" + str(p["seed"]) + ".npz")
    model_path = os.path.join(p["save_path"], "model_seed=" + str(p["seed"]))

    if (
        os.path.isfile(returns_path)
        or os.path.isfile(losses_path)
        or os.path.isfile(model_path)
    ):
        # check if same algorithm as been run on current env with same seed
        return "Same algorithm with same seed results already exists. Delete them and restart, or change the experiment name."

    param_path = os.path.join(
        p["save_path"],
        "..",  # parameters.json is outside the algorithm folder (in the experiment folder)
        "parameters.json",
    )

    try:
        with open(param_path, "r") as f:
            params = json.load(f)
        for param in SHARED_PARAMS:
            if params[param] != p[param]:
                return "Same experiment has been run with different shared parameters. Change the experiment name."
        if f"---- {p['algo']} ---" in params.keys():
            for param in AGENT_PARAMS[p["algo"]]:
                if params[param] != p[param]:
                    return f"Same experiment has been run with different {p['algo']} parameters. Change the experiment name."
            return "PASS_2"
        return "PASS_1"
    except FileNotFoundError:
        if os.path.exists(os.path.join(p["save_path"], "..")):
            return "There is a folder with this experiment name and no parameters.json. Delete the folder and restart, or change the experiment name."
    return "PASS_0"


def prepare_logs(p: dict):
    result = check_experiment(p)
    if "PASS" not in result:
        raise AssertionError(result)

    if result == "PASS_2":
        # same experiment with different seed, so no need to create/update anything
        return
    params = {}
    params_path = os.path.join(
        p["save_path"],
        "..",
        "parameters.json",
    )

    os.makedirs(p["save_path"])
    # need to create a directory for this experiment, algorithm combination

    if result == "PASS_0":
        # if this is totally new experiment, store shared parameters afresh
        params["---- Shared parameters ---"] = "----------------"
        for shared_param in SHARED_PARAMS:
            params[shared_param] = p[shared_param]
    elif result == "PASS_1":
        # if this experiment was run previously but not with current algorithm, load the previous parameters
        with open(params_path, "r") as f:
            params = json.load(f)

    # update params with algorithm parameters for this experiment
    params[f"---- {p['algo']} ---"] = "-----------------------------"
    for agent_param in AGENT_PARAMS[p["algo"]]:
        params[agent_param] = p[agent_param]

    params_order = SHARED_PARAMS + [
        i
        for subl in [
            [f"---- {agent} ---"] + AGENT_PARAMS[agent]
            for agent in sorted(AGENT_PARAMS)
            if f"---- {agent} ---" in params
        ]
        for i in subl
    ]
    params = {key: params[key] for key in params_order}
    # sort keys in uniform order and store

    with open(params_path, "w") as f:
        json.dump(params, f, indent=4)


def save_logs(p: dict, returns: np.array, losses: np.array, agent: BasicDQN):

    returns_path = os.path.join(
        p["save_path"], "returns_seed=" + str(p["seed"]) + ".npz"
    )
    losses_path = os.path.join(p["save_path"], "losses_seed=" + str(p["seed"]) + ".npz")
    model_path = os.path.join(p["save_path"], "model_seed=" + str(p["seed"]))

    np.save(returns_path, returns)
    np.save(losses_path, returns)
    torch.save(agent.q_network.state_dict(), model_path)
