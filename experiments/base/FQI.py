import os
import time
import random
import json
from tqdm import tqdm
import numpy as np
import torch
from slimRL.networks.architectures.DQN import BasicDQN
from slimRL.sample_collection.replay_buffer import ReplayBuffer
from slimRL.sample_collection.utils import collect_single_sample
from experiments.base.logger import save_logs


def train(
    p: dict,
    agent: BasicDQN,
    rb: ReplayBuffer,
):

    print(f"{p['env']}__{p['algo']}__{p['seed']}__{int(time.time())}")

    random.seed(p["seed"])
    np.random.seed(p["seed"])
    torch.manual_seed(p["seed"])

    for idx_bellman_iteration in tqdm(range(p["n_bellman_iterations"])):
        best_loss = float("inf")
        patience = 0

        for grad_step in tqdm(range(p["n_fitting_steps"])):
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
        print(f"Iteration: {idx_bellman_iteration}, Best loss = {best_loss}")

        agent.update_target_params(0)

        model_path = os.path.join(p["save_path"], "model_seed=" + str(p["seed"]))
        torch.save(agent.q_network.state_dict(), model_path)
