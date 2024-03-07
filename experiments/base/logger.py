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
    "update_to_data",
    "target_update_period",
    "n_initial_samples",
    "end_epsilon",
    "duration_epsilon",
    "horizon",
]

AGENT_PARAMS = {"DQN": ["n_epochs", "n_training_steps_per_epoch"]}


def save_logs(p: dict, returns: np.array, losses: np.array, agent: BasicDQN):
    save_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../../", p["save_path"]
    )
    os.makedirs(save_path, exist_ok=True)

    returns_path = os.path.join(save_path, "returns_seed=" + str(p["seed"]) + ".npz")
    losses_path = os.path.join(save_path, "losses_seed=" + str(p["seed"]) + ".npz")
    model_path = os.path.join(save_path, "model_seed=" + str(p["seed"]))

    if (
        os.path.isfile(returns_path)
        or os.path.isfile(losses_path)
        or os.path.isfile(model_path)
    ):
        raise AssertionError(
            "Experiment results already exists. Delete them and restart, or change the experiment name."
        )

    np.save(returns_path, returns)
    np.save(losses_path, returns)
    torch.save(agent.q_network.state_dict(), model_path)

    param_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../",
        p["save_path"],
        "..",
        "parameters.json",
    )

    params = {}

    try:
        with open(param_path, "r") as f:
            params = json.load(f)
        for param in SHARED_PARAMS:
            if params[param] != p[param]:
                raise AssertionError(
                    "Same experiment has been run with different shared parameters. Change the experiment name."
                )
        if f"---- {p['agent']} ---" in params.keys():
            raise AssertionError(
                f"Same experiment with {p['agent']} has been run already. Change the experiment name."
            )
    except FileNotFoundError:
        params["---- Shared parameters ---"] = "----------------"
        for shared_param in SHARED_PARAMS:
            params[shared_param] = p[shared_param]

    params[f"---- {p['agent']} ---"] = "-----------------------------"
    for agent_param in AGENT_PARAMS[p["agent"]]:
        params[agent_param] = p[agent_param]

    with open(param_path, "w") as f:
        json.dump(params, f, indent=4)
