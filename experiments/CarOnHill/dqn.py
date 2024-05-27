import os
import sys
import argparse
import torch
from experiments.base.parser import dqn_parser
from slimRL.environments.car_on_hill import CarOnHill
from slimRL.sample_collection.replay_buffer import ReplayBuffer
from slimRL.networks.architectures.DQN import BasicDQN
from experiments.base.DQN import train
from experiments.base.logger import prepare_logs


def run(argvs=sys.argv[1:]):
    parser = argparse.ArgumentParser("Train DQN on CarOnHill.")
    dqn_parser(parser)
    args = parser.parse_args(argvs)

    p = vars(args)
    p["env"] = "CarOnHill"
    p["algo"] = "DQN"
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
        lr=p["start_lr"],
        adam_eps=p["lr_epsilon"],
        train_frequency=p["update_to_data"],
        target_update_frequency=p["target_update_period"],
    )
    train(p, agent, env, rb)
