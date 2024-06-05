import os
import sys
import argparse
import jax
import optax
from experiments.base.parser import fqi_parser
from slimRL.environments.car_on_hill import CarOnHill
from slimRL.sample_collection.replay_buffer import ReplayBuffer
from slimRL.networks.architectures.DQN import BasicDQN
from experiments.base.FQI import train
from experiments.base.logger import prepare_logs
from slimRL.sample_collection.utils import update_replay_buffer


def run(argvs=sys.argv[1:]):
    parser = argparse.ArgumentParser("Train FQI on CarOnHill.")
    fqi_parser(parser)
    args = parser.parse_args(argvs)

    p = vars(args)
    p["env"] = "CarOnHill"
    p["algo"] = "FQI"
    p["save_path"] = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"logs/{p['experiment_name']}/{p['algo']}/seed={p['seed']}",
    )
    p["hidden_layers"] = [int(h) for h in p["hidden_layers"]]

    prepare_logs(p)

    q_key, explore_key = jax.random.split(jax.random.PRNGKey(p["seed"]))

    env = CarOnHill()
    rb = ReplayBuffer(
        observation_shape=env.observation_shape,
        replay_capacity=p["replay_capacity"],
        update_horizon=p["update_horizon"],
        gamma=p["gamma"],
    )

    agent = BasicDQN(
        q_key,
        env,
        hidden_layers=p["hidden_layers"],
        gamma=p["gamma"],
        update_horizon=p["update_horizon"],
        lr_schedule=optax.linear_schedule(
            p["start_lr"],
            p["end_lr"],
            int(
                (
                    p["n_bellman_iterations"]
                    * p["n_fitting_steps"]
                    * p["replay_capacity"]
                )
                / p["batch_size"]
            ),
        ),
        adam_eps=p["lr_epsilon"],
        train_frequency=-1,
        target_update_frequency=-1,
    )

    update_replay_buffer(explore_key, env, agent, rb, p)

    train(p, agent, rb)
