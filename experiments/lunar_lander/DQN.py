import os
import sys
import time
import argparse
import jax
from experiments.base.parser import dqn_parser
from slimRL.environments.lunar_lander import LunarLander
from slimRL.sample_collection.replay_buffer import ReplayBuffer
from slimRL.networks.DQN import DQN
from experiments.base.DQN import train
from experiments.base.logger import prepare_logs


def run(argvs=sys.argv[1:]):
    print(f"---Lunar Lander__DQN__{time.strftime('%d-%m-%Y %H:%M:%S')}---")
    parser = argparse.ArgumentParser("Train DQN on Lunar Lander.")
    dqn_parser(parser)
    args = parser.parse_args(argvs)

    p = vars(args)
    p["env"] = "Lunar Lander"
    p["algo"] = "DQN"
    p["save_path"] = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"exp_output/{p['experiment_name']}/{p['algo']}",
    )
    p["hidden_layers"] = [int(h) for h in p["hidden_layers"]]

    prepare_logs(p)

    q_key, train_key = jax.random.split(jax.random.PRNGKey(p["seed"]))

    env = LunarLander()
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
        train_frequency=p["update_to_data"],
        target_update_frequency=p["target_update_period"],
    )
    train(train_key, p, agent, env, rb)
