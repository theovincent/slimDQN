import sys
import argparse
import torch
from experiments.base.parser import addparse
from experiments.base.load_parameters import load_parameters
from slimRL.environments.car_on_hill import CarOnHill
from slimRL.sample_collection.replay_buffer import ReplayBuffer
from slimRL.networks.architectures.dqn import BasicDQN
from experiments.base.dqn import train


def run(argvs=sys.argv[1:]):
    import warnings

    warnings.simplefilter(action="ignore", category=FutureWarning)

    parser = argparse.ArgumentParser("Train DQN on CarOnHill.")
    addparse(parser)
    args = parser.parse_args(argvs)

    p = load_parameters(args)
    p["env_id"] = "car_on_hill"
    p["agent"] = "dqn"

    device = torch.device(
        "cuda" if torch.cuda.is_available() and p["use_gpu"] else "cpu"
    )

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
        tau=p["tau"],
        lr=p["lr"],
        loss_type=p["loss_type"],
        train_frequency=p["n_training_steps_per_online_update"],
        target_update_frequency=p["n_training_steps_per_target_update"],
    )
    train(p, agent, env, rb)