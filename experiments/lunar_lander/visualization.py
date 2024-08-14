import argparse
import json
import os
import pickle
import sys

import jax.numpy as jnp
from slimDQN.environments.lunar_lander import LunarLander
from slimDQN.networks.architectures.dqn import DQNNet


def run(argvs=sys.argv[1:]):
    parser = argparse.ArgumentParser("Visualize agent's performances.")
    parser.add_argument(
        "-en",
        "--experiment_name",
        help="Experiment name.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-an",
        "--algo_name",
        help="Algorithm name.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-s",
        "--seed",
        help="Seed.",
        type=int,
        required=True,
    )
    args = parser.parse_args(argvs)

    env = LunarLander(render_mode="human")
    env_name = os.path.abspath(__file__).split("/")[-2]

    p_path = f"experiments/{env_name}/exp_output/{args.experiment_name}/parameters.json"
    p = json.load(open(p_path, "rb"))
    q_network = DQNNet(p["shared_parameters"]["features"], env.n_actions)

    model_path = f"experiments/{env_name}/exp_output/{args.experiment_name}/{args.algo_name}/models/{args.seed}"
    model = pickle.load(open(model_path, "rb"))

    for _ in range(5):
        total_reward = 0
        absorbing = False
        env.reset()

        while not absorbing:
            env.env.render()

            action = jnp.argmax(q_network.apply(model["params"], env.state)).item()

            reward, absorbing = env.step(action)
            total_reward += reward

        print("Total reward: ", total_reward, flush=True)


if __name__ == "__main__":
    run()
