import os
import json
import pickle
import jax
from slimRL.networks.DQN import DQN

SHARED_PARAMS = [
    "experiment_name",
    "env",
    "replay_capacity",
    "batch_size",
    "update_horizon",
    "gamma",
    "lr",
    "horizon",
    "hidden_layers",
]

AGENT_PARAMS = {
    "DQN": [
        "n_epochs",
        "n_training_steps_per_epoch",
        "update_to_data",
        "end_epsilon",
        "duration_epsilon",
        "target_update_period",
        "n_initial_samples",
    ],
    "FQI": ["n_bellman_iterations", "n_fitting_steps"],
}


def check_experiment(p: dict):
    # check if the experiment is valid
    returns_path = os.path.join(p["save_path"], "returns_seed_" + str(p["seed"]) + ".npy")
    losses_path = os.path.join(p["save_path"], "losses_seed_" + str(p["seed"]) + ".npy")
    model_path = os.path.join(p["save_path"], "model_seed_" + str(p["seed"]))

    assert not (
        os.path.exists(returns_path) or os.path.exists(losses_path) or os.path.exists(model_path)
    ), "Same algorithm with same seed results already exists. Delete them and restart, or change the experiment name."

    params_path = os.path.join(
        os.path.split(p["save_path"])[0],  # parameters.json is outside the algorithm folder (in the experiment folder)
        "parameters.json",
    )

    if os.path.exists(params_path):
        params = json.load(open(params_path, "r"))
        for param in SHARED_PARAMS:
            assert (
                params[param] == p[param]
            ), "Same experiment has been run with different shared parameters. Change the experiment name."
        if f"---- {p['algo']} ---" in params.keys():
            for param in AGENT_PARAMS[p["algo"]]:
                assert (
                    params[param] == p[param]
                ), f"Same experiment has been run with different {p['algo']} parameters. Change the experiment name."
    else:
        assert not os.path.exists(
            os.path.join(p["save_path"], "..")
        ), "There is a folder with this experiment name and no parameters.json. Delete the folder and restart, or change the experiment name."


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

    with open(params_path.replace(".json", ".tmp"), "w") as f:
        json.dump(params, f, indent=4)
    os.rename(params_path.replace(".json", ".tmp"), params_path)


def prepare_logs(p: dict):
    check_experiment(p)
    os.makedirs(p["save_path"], exist_ok=True)  # need to create a directory for this experiment, algorithm combination
    store_params(p)


def pickle_load(path):
    return pickle.load(open(path, "rb"))


def pickle_dump(obj, path):
    return pickle.dump(obj, open(path, "wb"))


def save_logs(p: dict, log_rewards: list, log_lengths: list, agent: DQN):
    rewards_path = os.path.join(p["save_path"], f"rewards_seed_{p['seed']}.json")
    lengths_path = os.path.join(p["save_path"], f"lengths_seed_{p['seed']}.json")
    model_path = os.path.join(p["save_path"], f"model_seed_{p['seed']}")

    json.dump(log_rewards, open(rewards_path, "w"))
    json.dump(log_lengths, open(lengths_path, "w"))
    model = {
        "params": jax.device_get(agent.params),
        "hidden_layers": agent.q_network.hidden_layers,
    }
    pickle_dump(model, model_path)
