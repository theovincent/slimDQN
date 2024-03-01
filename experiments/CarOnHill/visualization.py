import sys
import argparse
import torch
from slimRL.environments.car_on_hill import CarOnHill

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
        default=10000
    )
    args = parser.parse_args(argvs)

    env = CarOnHill()
    agent = torch.load(args.model)


    obs, _ = env.reset()
    for _ in range(args.steps):
        env.render()
        action = [agent.best_action(obs)]
        next_obs, reward, termination, infos = env.step(action)
        episode_end = "episode_end" in infos.keys() and infos["episode_end"]

        if termination or episode_end:
            next_obs, _ = env.reset()

        obs = next_obs