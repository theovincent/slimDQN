import os
import sys
import argparse
import torch
from experiments.base.parser import dqn_parser
from slimRL.environments.car_on_hill import CarOnHill
from slimRL.sample_collection.replay_buffer import ReplayBuffer
from slimRL.networks.architectures.dqn import BasicDQN
from experiments.base.dqn import train


def run(argvs=sys.argv[1:]):
    import warnings

    warnings.simplefilter(action="ignore", category=FutureWarning)

    parser = argparse.ArgumentParser("Train DQN on CarOnHill.")
    dqn_parser(parser)
    args = parser.parse_args(argvs)

    p = vars(args)
    p["env"] = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
    p["agent"] = os.path.basename(os.path.abspath(__file__)).split(".")[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = CarOnHill(horizon=p["horizon"])
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
        lr=p["lr"],
        train_frequency=p["update_to_data"],
        target_update_frequency=p["target_update_period"],
    )
    train(p, agent, env, rb)
