import os
import jax
from tqdm import tqdm
from experiments.base.logger import pickle_dump
from slimRL.networks.architectures.DQN import BasicDQN
from slimRL.sample_collection.replay_buffer import ReplayBuffer


def train(
    p: dict,
    agent: BasicDQN,
    rb: ReplayBuffer,
):

    n_grad_steps = int((p["n_fitting_steps"] * p["replay_capacity"]) / p["batch_size"])

    model = {
        "params": jax.device_get(agent.params),
        "hidden_layers": agent.q_network.hidden_layers,
    }
    model_path = os.path.join(p["save_path"], f"model_iteration_0")
    pickle_dump(model, model_path)
    for idx_bellman_iteration in tqdm(range(p["n_bellman_iterations"])):
        for _ in range(n_grad_steps):
            agent.update_online_params(0, p["batch_size"], rb)
        agent.update_target_params(0)

        model = {
            "params": jax.device_get(agent.params),
            "hidden_layers": agent.q_network.hidden_layers,
        }
        model_path = os.path.join(
            p["save_path"], f"model_iteration_{idx_bellman_iteration+1}"
        )
        pickle_dump(model, model_path)
