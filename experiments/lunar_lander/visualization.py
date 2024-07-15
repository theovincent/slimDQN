import sys
import argparse
import jax.numpy as jnp
import pickle
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
    model = pickle.load(open(args.model, "rb"))
    q_network = DQNNet(env.n_actions, model["hidden_layers"])

    env.reset()
    total_reward = 0
    for _ in range(args.steps):
        env.env.render()
        action = jnp.argmax(q_network.apply(model["params"], env.state)).item()
        _, reward, termination = env.step(action)
        total_reward += reward

        if termination or env.n_steps > args.horizon:
            print("Total reward = ", total_reward)
            total_reward = 0
            env.reset()
