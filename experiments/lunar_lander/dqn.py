import os
import sys
import jax
from slimDQN.environments.lunar_lander import LunarLander
from slimDQN.sample_collection.replay_buffer import ReplayBuffer
from slimDQN.networks.dqn import DQN
from experiments.base.dqn import train
from experiments.base.utils import prepare_logs


def run(argvs=sys.argv[1:]):
    env_name, algo_name = os.path.abspath(__file__).split(os.sep)[-2], os.path.abspath(__file__).split(os.sep)[-1][:-3]
    p = prepare_logs(env_name, algo_name, argvs)

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
        update_to_data=p["update_to_data"],
        target_update_frequency=p["target_update_frequency"],
    )
    train(train_key, p, agent, env, rb)
