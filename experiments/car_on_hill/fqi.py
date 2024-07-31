import os
import sys
import jax
from experiments.base.parser import fqi_parser
from slimRL.environments.car_on_hill import CarOnHill
from slimRL.sample_collection.replay_buffer import ReplayBuffer
from slimRL.networks.dqn import DQN
from experiments.base.fqi import train
from experiments.base.utils import prepare_logs
from slimRL.sample_collection.utils import update_replay_buffer


def run(argvs=sys.argv[1:]):

    env_name = os.path.abspath(__file__).split(os.sep)[-2]
    p = fqi_parser(env_name, argvs)

    prepare_logs(p)

    key = jax.random.PRNGKey(p["seed"])
    q_key, train_key = jax.random.split(key)

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
        update_to_data=-1,
        target_update_frequency=-1,
    )

    update_replay_buffer(jax.random.PRNGKey(0), env, agent, rb, p)

    train(train_key, p, agent, rb)
