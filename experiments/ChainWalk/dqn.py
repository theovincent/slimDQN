import numpy as np
import torch
from experiments.base.load_parameters import load_parameters
from slimRL.environments.chain import generate_simple_chain
from slimRL.sample_collection.replay_buffer import ReplayBuffer
from slimRL.networks.architectures.dqn import BasicDQN
from experiments.base.dqn import train

def run(param_file):
    p = load_parameters(param_file, "chain", "dqn")
    device = torch.device("cuda" if torch.cuda.is_available() and p["use_gpu"] else "cpu")
    mu = np.zeros(p["chain_size"])
    mu[p["chain_size"]//2] = 1
    env = generate_simple_chain(p["chain_size"], [0], 1.0, 1.0, mu=mu, horizon=p["chain_size"]*10)
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

if __name__ == "__main__":
    param_file = "chain.json"
    run(param_file)
