import os
import json
import numpy as np
import torch
from slimRL.networks.architectures.dqn import BasicDQN

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

AGENT_PARAMS = {"dqn": ["n_epochs", "n_training_steps_per_epoch"]}


def save_logs(p: dict, returns: np.array, losses: np.array, agent: BasicDQN):
    save_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../experiments/",
        p["env"],
        p["experiment_name"],
        p["agent"],
    )
    os.makedirs(save_path, exist_ok=True)

    returns_path = os.path.join(save_path, "returns_seed=" + str(p["seed"]) + ".npz")
    losses_path = os.path.join(save_path, "losses_seed=" + str(p["seed"]) + ".npz")
    model_path = os.path.join(save_path, "model_seed=" + str(p["seed"]))
    np.save(returns_path, returns)
    np.save(losses_path, returns)
    torch.save(agent.q_network.state_dict(), model_path)

    param_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../experiments/",
        p["env"],
        p["experiment_name"],
        "parameters.json",
    )

    params = {}

    try:
        with open(param_path, "r") as f:
            params = json.load(f)
    except FileNotFoundError or json.decoder.JSONDecodeError:
        params["---- Shared parameters ---"] = "----------------"
        for shared_param in SHARED_PARAMS:
            params[shared_param] = p[shared_param]

    params["---- " + p["agent"] + " ---"] = "-----------------------------"
    for agent_param in AGENT_PARAMS[p["agent"]]:
        params[agent_param] = p[agent_param]

    with open(param_path, "w") as f:
        json.dump(params, f, indent=4)
