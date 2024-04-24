import sys
import argparse
import torch
from slimRL.environments.car_on_hill import CarOnHill
from slimRL.networks.architectures.DQN import DQNNet
from slimRL.environments.visualization.car_on_hill import render


def run(argvs=sys.argv[1:]):
    import warnings

    warnings.simplefilter(action="ignore", category=FutureWarning)

    parser = argparse.ArgumentParser("Visualize DQN performance on CarOnHill.")
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
    agent = DQNNet(env)
    agent.load_state_dict(torch.load(args.model))
    agent.eval()

    obs = env.reset()

    for _ in range(args.steps):
        render(env)
        action = torch.argmax(agent(torch.Tensor(obs))).numpy()
        next_obs, reward, termination = env.step(action)

        if termination or env.n_steps > args.horizon:
            next_obs = env.reset()

        obs = next_obs
