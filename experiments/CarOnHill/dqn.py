import sys
import argparse
import torch
from experiments.base.parser import addparse
from experiments.base.load_parameters import load_parameters
from slimRL.environments.car_on_hill import CarOnHillDQN
from slimRL.sample_collection.replay_buffer import ReplayBuffer
from slimRL.networks.architectures.dqn import BasicDQN
from experiments.base.dqn import train

def run(argvs=sys.argv[1:]):
    import warnings
    warnings.simplefilter(action="ignore", category=FutureWarning)

    parser = argparse.ArgumentParser("Train DQN on CarOnHill.")
    addparse(parser)
    args = parser.parse_args(argvs)
    param_file = args.params_file

    p = load_parameters(param_file, "car_on_hill", "dqn", args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and p["use_gpu"] else "cpu")
    env = CarOnHillDQN(horizon=200)
    rb = ReplayBuffer(observation_shape=env.observation_shape,
                      replay_capacity=p["replay_capacity"],
                      batch_size=p["batch_size"],
                      update_horizon=p["update_horizon"],
                      gamma=p["gamma"],
                      )
    agent = BasicDQN(env,
                     device=device,
                     gamma=p["gamma"],
                     tau=p["tau"],
                     lr=p["lr"],
                     loss_type=p["loss_type"],
                     train_frequency=p["train_frequency"],
                     target_update_frequency=p["target_update_frequency"],
                     save_model=p["save_model"],
                     )
    train(p, agent, env, rb)

    obs, _ = env.reset()
    for global_step in range(10000//10):
        env.render()
        action = [agent.best_action(obs)]
        next_obs, reward, termination, infos = env.step(action)
        episode_end = "episode_end" in infos.keys() and infos["episode_end"]

        if termination or episode_end:
            next_obs, _ = env.reset()

        obs = next_obs
