import sys
import argparse
import jax.numpy as jnp
from experiments.base.logger import pickle_load
from slimRL.environments.car_on_hill import CarOnHill
from slimRL.networks.architectures.DQN import DQNNet
from slimRL.environments.visualization.car_on_hill import render


def run(argvs=sys.argv[1:]):
    import warnings

    warnings.simplefilter(action="ignore", category=FutureWarning)

    parser = argparse.ArgumentParser("Visualize DQN performance on Car-On-Hill.")
    parser.add_argument(
        "-m",
        "--model",
        help="Model path.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-s",
        "--steps",
        help="Steps for which agent is visualized.",
        type=int,
        required=False,
        default=10000,
    )
    parser.add_argument(
        "-H",
        "--horizon",
        help="Horizon of the episode.",
        type=int,
        required=False,
        default=300,
    )
    args = parser.parse_args(argvs)

    env = CarOnHill()
    model = pickle_load(args.model)
    q_network = DQNNet(env.n_actions, model["hidden_layers"])

    obs = env.reset()
    total_reward = 0
    for _ in range(args.steps):
        render(env)
        action = jnp.argmax(q_network.apply(model["params"], env.state)).item()
        next_obs, reward, termination = env.step(action)
        total_reward += reward

        if termination or env.n_steps > args.horizon:
            print("Total reward = ", total_reward)
            total_reward = 0
            next_obs = env.reset()

        obs = next_obs
