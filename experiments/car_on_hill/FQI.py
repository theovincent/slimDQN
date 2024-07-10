import os
import sys
import time
import argparse
import jax
from experiments.base.parser import fqi_parser
from slimRL.environments.car_on_hill import CarOnHill
from slimRL.sample_collection.replay_buffer import ReplayBuffer
from slimRL.networks.DQN import DQN
from experiments.base.FQI import train
from experiments.base.logger import prepare_logs
from slimRL.sample_collection.utils import update_replay_buffer


def run(argvs=sys.argv[1:]):
    print(f"---Car-On-Hill__FQI__{time.strftime('%d-%m-%Y %H:%M:%S')}---")
    parser = argparse.ArgumentParser("Train FQI on Car-On-Hill.")
    fqi_parser(parser)
    args = parser.parse_args(argvs)

    p = vars(args)
    p["env"] = "Car-On-Hill"
    p["algo"] = "FQI"
    p["save_path"] = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"exp_output/{p['experiment_name']}/{p['algo']}/seed_{p['seed']}",
    )
    p["hidden_layers"] = [int(h) for h in p["hidden_layers"]]

    prepare_logs(p)

    q_key = jax.random.PRNGKey(p["seed"])

    env = CarOnHill()
    rb = ReplayBuffer(
        observation_shape=env.observation_shape,
        replay_capacity=p["replay_capacity"],
        update_horizon=p["update_horizon"],
        gamma=p["gamma"],
    )

    agent = DQN(
        q_key,
        env.observation_shape[0],
        env.n_actions,
        hidden_layers=p["hidden_layers"],
        lr=p["lr"],
        gamma=p["gamma"],
        update_horizon=p["update_horizon"],
        train_frequency=-1,
        target_update_frequency=-1,
    )

    update_replay_buffer(jax.random.PRNGKey(0), env, agent, rb, p)

    train(p, agent, rb)
