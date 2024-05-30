import os
import time
import random
from tqdm import tqdm
import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
from slimRL.networks.architectures.DQN import BasicDQN
from slimRL.sample_collection.replay_buffer import ReplayBuffer


def train(
    p: dict,
    agent: BasicDQN,
    rb: ReplayBuffer,
):

    print(f"{p['env']}__{p['algo']}__{p['seed']}__{int(time.time())}")

    random.seed(p["seed"])
    np.random.seed(p["seed"])
    torch.manual_seed(p["seed"])

    n_grad_steps = p["n_fitting_steps"] * int(
        np.ceil(p["replay_capacity"] / p["batch_size"])
    )
    linear_lr_schedule = lambda step: 1 - step / n_grad_steps * (
        1 - p["end_lr"] / p["start_lr"]
    )

    for idx_bellman_iteration in tqdm(range(p["n_bellman_iterations"])):
        scheduler = LambdaLR(agent.optimizer, lr_lambda=linear_lr_schedule)
        for _ in range(n_grad_steps):
            agent.update_online_params(0, p["batch_size"], rb)
            scheduler.step()
        agent.update_target_params(0)

        model_path = os.path.join(
            p["save_path"], f"model_iteration={idx_bellman_iteration}"
        )
        torch.save(
            {
                "hidden_layers": agent.q_network.hidden_layers,
                "network": agent.q_network.state_dict(),
            },
            model_path,
        )
