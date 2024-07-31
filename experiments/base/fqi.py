import os
import jax
import pickle
from tqdm import tqdm
from slimRL.networks.dqn import DQN
from slimRL.sample_collection.replay_buffer import ReplayBuffer


def train(
    key: jax.random.PRNGKey,
    p: dict,
    agent: DQN,
    rb: ReplayBuffer,
):
    n_grad_steps = int((p["n_fitting_steps"] * p["replay_capacity"]) / p["batch_size"])

    model = {"params": jax.device_get(agent.params), "hidden_layers": agent.q_network.hidden_layers}
    model_path = os.path.join(p["save_path"], "model_iteration_0")
    pickle.dump(model, open(model_path, "wb"))
    for idx_bellman_iteration in tqdm(range(p["n_bellman_iterations"])):
        for _ in range(n_grad_steps):
            key, grad_key = jax.random.split(key)
            agent.update_online_params(grad_key, 0, p["batch_size"], rb)
        agent.update_target_params(0)

        model = {"params": jax.device_get(agent.params), "hidden_layers": agent.q_network.hidden_layers}
        model_path = os.path.join(p["save_path"], f"model_iteration_{idx_bellman_iteration+1}")
        pickle.dump(model, open(model_path, "wb"))
