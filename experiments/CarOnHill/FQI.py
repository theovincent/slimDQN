import os
import sys
import argparse
import torch
import random
from experiments.base.parser import fqi_parser
from slimRL.environments.car_on_hill import CarOnHill
from slimRL.sample_collection.replay_buffer import ReplayBuffer
from slimRL.networks.architectures.DQN import BasicDQN
from experiments.base.FQI import train
from experiments.base.logger import prepare_logs
from slimRL.sample_collection.utils import collect_single_sample


def run(argvs=sys.argv[1:]):
    parser = argparse.ArgumentParser("Train FQI on CarOnHill.")
    fqi_parser(parser)
    args = parser.parse_args(argvs)

    p = vars(args)
    p["env"] = "CarOnHill"
    p["algo"] = "FQI"
    p["save_path"] = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"logs/{p['experiment_name']}/{p['algo']}",
    )

    prepare_logs(p)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = CarOnHill()
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
        update_horizon=p["update_horizon"],
        lr=p["lr"],
        adam_eps=p["lr_epsilon"],
        train_frequency=-1,
        target_update_frequency=-1,
    )

    env.reset()
    for steps in range(p["replay_capacity"]):
        collect_single_sample(env, agent, rb, p, 0)

    print("Replay buffer filled.")

    train(p, agent, rb)
