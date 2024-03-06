import sys
import argparse
import torch
import numpy as np
from experiments.base.parser import dqn_parse
from experiments.base.utils import load_parameters
from slimRL.environments.chain import ChainDQN
from slimRL.sample_collection.replay_buffer import ReplayBuffer
from slimRL.networks.architectures.dqn import BasicDQN
from experiments.base.dqn import train


def run(argvs=sys.argv[1:]):
    import warnings

    warnings.simplefilter(action="ignore", category=FutureWarning)

    parser = argparse.ArgumentParser("Train DQN on CarOnHill.")
    dqn_parse(parser)
    parser.add_argument(
        "-size",
        "--chain_size",
        help="Chain size",
        type=int,
        required=True,
    )
    parser.add_argument(
        "-pr",
        "--transition_prob",
        help="The probability of success of an action (transition probability)",
        type=float,
        required=True,
    )

    args = parser.parse_args(argvs)
    param_file = args.params_file
    p = load_parameters(param_file, "chain", "dqn", args.seed)
    p["chain_size"] = args.chain_size
    p["transition_prob"] = args.transition_prob

    device = torch.device(
        "cuda" if torch.cuda.is_available() and p["use_gpu"] else "cpu"
    )
    env = ChainDQN(state_n=p["chain_size"], prob=p["transition_prob"])
    rb = ReplayBuffer(
        observation_shape=env.observation_shape,
        replay_capacity=p["replay_capacity"],
        batch_size=p["batch_size"],
        update_horizon=p["update_horizon"],
        gamma=p["gamma"],
    )
    agent = BasicDQN(
        env,
        device=device,
        gamma=p["gamma"],
        tau=p["tau"],
        lr=p["lr"],
        loss_type=p["loss_type"],
        train_frequency=p["train_frequency"],
        target_update_frequency=p["target_update_frequency"],
        save_model=p["save_model"],
    )
    train(p, agent, env, rb)
