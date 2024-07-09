import sys
import argparse
import jax.numpy as jnp
from experiments.base.logger import pickle_load
from slimRL.environments.lunar_lander import LunarLander
from slimRL.networks.architectures.DQN import DQNNet


def run(argvs=sys.argv[1:]):
    import warnings

    warnings.simplefilter(action="ignore", category=FutureWarning)

    parser = argparse.ArgumentParser("Visualize DQN performance on Lunar Lander.")
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
        default=1000,
    )
    args = parser.parse_args(argvs)

    env = LunarLander(render_mode="human")
    model = pickle_load(args.model)
    q_network = DQNNet(env.n_actions, model["hidden_layers"])

    obs = env.reset()
    total_reward = 0
    for _ in range(args.steps):
        env.env.render()
        action = jnp.argmax(q_network.apply(model["params"], env.state)).item()
        next_obs, reward, termination = env.step(action)
        total_reward += reward

        if termination or env.n_steps > args.horizon:
            print("Total reward = ", total_reward)
            total_reward = 0
            next_obs = env.reset()

        obs = next_obs
