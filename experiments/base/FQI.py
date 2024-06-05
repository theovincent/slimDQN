import os
import time
import pickle
from tqdm import tqdm
from slimRL.networks.architectures.DQN import BasicDQN
from slimRL.sample_collection.replay_buffer import ReplayBuffer


def train(
    p: dict,
    agent: BasicDQN,
    rb: ReplayBuffer,
):

    print(f"{p['env']}__{p['algo']}__{p['seed']}__{int(time.time())}")

    n_grad_steps = int((p["n_fitting_steps"] * p["replay_capacity"]) / p["batch_size"])

    for idx_bellman_iteration in tqdm(range(p["n_bellman_iterations"])):
        for _ in range(n_grad_steps):
            agent.update_online_params(0, p["batch_size"], rb)
        agent.update_target_params(0)

        model_path = os.path.join(
            p["save_path"], f"model_iteration={idx_bellman_iteration}"
        )
        model = {}
        model["params"] = agent.params
        model["hidden_layers"] = agent.q_network.hidden_layers
        pickle.dump(model, open(model_path, "wb"))
