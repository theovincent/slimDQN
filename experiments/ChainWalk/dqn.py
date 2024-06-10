import os
import sys
import argparse
import numpy as np
import torch
from experiments.base.parser import dqn_parser
from experiments.ChainWalk.parser import chain_parser
from slimRL.environments.chain import Chain
from slimRL.sample_collection.replay_buffer import ReplayBuffer
from slimRL.networks.architectures.DQN import BasicDQN
from experiments.base.DQN import train
from experiments.base.logger import prepare_logs


def run(argvs=sys.argv[1:]):
    parser = argparse.ArgumentParser("Train DQN on ChainWalk.")
    dqn_parser(parser)
    chain_parser(parser)

    args = parser.parse_args(argvs)

    p = vars(args)
    p["env"] = "ChainWalk-" + str(p["chain_size"])
    p["algo"] = "DQN"
    p["save_path"] = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"logs/{p['experiment_name']}/{p['env']}/{p['algo']}",
    )
    p["hidden_layers"] = [int(h) for h in p["hidden_layers"]]

    prepare_logs(p)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = Chain(state_n=p["chain_size"], prob=p["transition_prob"])
    rb = ReplayBuffer(
        observation_shape=env.observation_shape,
        replay_capacity=p["replay_capacity"],
        update_horizon=p["update_horizon"],
        gamma=p["gamma"],
    )
    agent = BasicDQN(
        env,
        hidden_layers=p["hidden_layers"],
        device=device,
        gamma=p["gamma"],
        update_horizon=p["update_horizon"],
        lr=p["lr"],
        adam_eps=p["lr_epsilon"],
        train_frequency=p["update_to_data"],
        target_update_frequency=p["target_update_period"],
    )
    train(p, agent, env, rb)
