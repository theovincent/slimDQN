import sys
import argparse
import torch
from slimRL.environments.lunar_lander import LunarLander
from slimRL.networks.architectures.DQN import DQNNet


def run(argvs=sys.argv[1:]):
    import warnings

    warnings.simplefilter(action="ignore", category=FutureWarning)

    parser = argparse.ArgumentParser("Visualize DQN performance on LunarLander.")
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
    args = parser.parse_args(argvs)

    env = LunarLander(render_mode="human")
    agent = DQNNet(env)
    agent.load_state_dict(torch.load(args.model))
    agent.eval()

    obs, _ = env.reset()
    for _ in range(args.steps):
        env.env.render()
        action = [torch.argmax(agent(torch.Tensor(obs))).numpy()]
        next_obs, reward, termination, infos = env.step(action)
        episode_end = "episode_end" in infos.keys() and infos["episode_end"]

        if termination or episode_end:
            next_obs, _ = env.reset()

        obs = next_obs
