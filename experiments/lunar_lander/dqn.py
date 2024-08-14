import os
import sys

import jax

from experiments.base.dqn import train
from experiments.base.utils import prepare_logs
from slimDQN.environments.lunar_lander import LunarLander
from slimDQN.networks.dqn import DQN
from slimDQN.sample_collection.replay_buffer import ReplayBuffer


def run(argvs=sys.argv[1:]):
    env_name, algo_name = os.path.abspath(__file__).split("/")[-2], os.path.abspath(__file__).split("/")[-1][:-3]
    p = prepare_logs(env_name, algo_name, argvs)

    q_key, train_key = jax.random.split(jax.random.PRNGKey(p["seed"]))

    env = LunarLander()
    rb = ReplayBuffer(
        observation_shape=env.observation_shape,
        replay_capacity=p["replay_buffer_capacity"],
        batch_size=p["batch_size"],
        update_horizon=p["update_horizon"],
        gamma=p["gamma"],
    )
    agent = DQN(
        q_key,
        env.observation_shape[0],
        env.n_actions,
        features=p["features"],
        learning_rate=p["learning_rate"],
        gamma=p["gamma"],
        update_horizon=p["update_horizon"],
        update_to_data=p["update_to_data"],
        target_update_frequency=p["target_update_frequency"],
    )
    train(train_key, p, agent, env, rb)


if __name__ == "__main__":
    run()
