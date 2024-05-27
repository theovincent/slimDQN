import os
import time
import random
from tqdm import tqdm
import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
from slimRL.networks.architectures.DQN import BasicDQN
from slimRL.sample_collection.replay_buffer import ReplayBuffer
from slimRL.sample_collection.utils import save_replay_buffer_store
from slimRL.sample_collection.schedules import linear_schedule


def train(
    p: dict,
    agent: BasicDQN,
    rb: ReplayBuffer,
):

    print(f"{p['env']}__{p['algo']}__{p['seed']}__{int(time.time())}")

    random.seed(p["seed"])
    np.random.seed(p["seed"])
    torch.manual_seed(p["seed"])

    save_replay_buffer_store(rb, p["save_path"])

    linear_lr_schedule = lambda step: 1 - step / p["n_fitting_steps"] * (
        1 - p["end_lr"] / p["start_lr"]
    )

    for idx_bellman_iteration in tqdm(range(p["n_bellman_iterations"])):
        best_loss = float("inf")
        patience = 0
        scheduler = LambdaLR(agent.optimizer, lr_lambda=linear_lr_schedule)
        for grad_step in range(p["n_fitting_steps"]):
            cumulative_loss = 0
            for _ in range(int(np.ceil(p["replay_capacity"] / p["batch_size"]))):
                loss = agent.update_online_params(0, rb)
                cumulative_loss += loss
            if cumulative_loss < best_loss:
                patience = 0
                best_loss = cumulative_loss
            else:
                patience += 1

            if patience > p["patience"]:
                break
            scheduler.step()
        agent.update_target_params(0)

        model_path = os.path.join(
            p["save_path"], f"model_iteration={idx_bellman_iteration}"
        )
        torch.save(agent.q_network.state_dict(), model_path)
