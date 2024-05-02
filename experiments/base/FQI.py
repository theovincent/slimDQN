import os
import time
import random
import json
from tqdm import tqdm
import numpy as np
import torch
from slimRL.networks.architectures.DQN import BasicDQN
from slimRL.sample_collection.replay_buffer import ReplayBuffer
from slimRL.sample_collection.utils import save_replay_buffer_store
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

    save_replay_buffer_store(rb, p["save_path"])

    for idx_bellman_iteration in tqdm(range(p["n_bellman_iterations"])):
        for grad_step in range(p["n_fitting_steps"]):
            for _ in range(int(np.ceil(p["replay_capacity"] / p["batch_size"]))):
                agent.update_online_params(0, rb)
        agent.update_target_params(0)

        model_path = os.path.join(
            p["save_path"], f"model_iteration={idx_bellman_iteration}"
        )
        torch.save(agent.q_network.state_dict(), model_path)
