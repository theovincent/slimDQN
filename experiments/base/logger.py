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
        p["save_path"], "returns_seed=" + str(p["seed"]) + ".npy"
    )
    losses_path = os.path.join(p["save_path"], "losses_seed=" + str(p["seed"]) + ".npy")
    model_path = os.path.join(p["save_path"], "model_seed=" + str(p["seed"]))

    if (
        os.path.isfile(returns_path)
        or os.path.isfile(losses_path)
        or os.path.isfile(model_path)
    ):
        # check if same algorithm as been run on current env with same seed
        raise AssertionError(
            "Same algorithm with same seed results already exists. Delete them and restart, or change the experiment name."
        )

    params_path = os.path.join(
        p["save_path"],
        "..",  # parameters.json is outside the algorithm folder (in the experiment folder)
        "parameters.json",
    )

    if os.path.exists(params_path):
        params = json.load(open(params_path, "r"))
        for param in SHARED_PARAMS:
            if params[param] != p[param]:
                raise AssertionError(
                    "Same experiment has been run with different shared parameters. Change the experiment name."
                )
        if f"---- {p['algo']} ---" in params.keys():
            for param in AGENT_PARAMS[p["algo"]]:
                if params[param] != p[param]:
                    raise AssertionError(
                        f"Same experiment has been run with different {p['algo']} parameters. Change the experiment name."
                    )
    else:
        if os.path.exists(os.path.join(p["save_path"], "..")):
            raise AssertionError(
                "There is a folder with this experiment name and no parameters.json. Delete the folder and restart, or change the experiment name."
            )


def store_params(p: dict):
    params_path = os.path.join(
        p["save_path"],
        "..",
        "parameters.json",
    )

    if os.path.exists(params_path):
        params = json.load(open(params_path, "r"))

    else:
        params = {}

        # store shared params
        params["---- Shared parameters ---"] = "----------------"
        for shared_param in SHARED_PARAMS:
            params[shared_param] = p[shared_param]

    if f"---- {p['algo']} ---" not in params.keys():
        # store algo params
        params[f"---- {p['algo']} ---"] = "-----------------------------"
        for agent_param in AGENT_PARAMS[p["algo"]]:
            params[agent_param] = p[agent_param]
    # set parameter order for sorting all keys in a pre-defined order

    algo_params = []
    for agent in sorted(AGENT_PARAMS):
        if f"---- {agent} ---" in params:
            algo_params = algo_params + [f"---- {agent} ---"] + AGENT_PARAMS[agent]

    params_order = SHARED_PARAMS + algo_params

    # sort keys in uniform order and store
    params = {key: params[key] for key in params_order}

    with open(params_path, "w") as f:
        json.dump(params, f, indent=4)


def prepare_logs(p: dict):
    check_experiment(p)
    os.makedirs(
        p["save_path"], exist_ok=True
    )  # need to create a directory for this experiment, algorithm combination
    store_params(p)


def save_logs(p: dict, returns: np.array, losses: np.array, agent: BasicDQN):

    returns_path = os.path.join(p["save_path"], "returns_seed=" + str(p["seed"]))
    losses_path = os.path.join(p["save_path"], "losses_seed=" + str(p["seed"]))
    model_path = os.path.join(p["save_path"], "model_seed=" + str(p["seed"]))

    np.save(returns_path, returns)
    np.save(losses_path, returns)
    torch.save(agent.q_network.state_dict(), model_path)
