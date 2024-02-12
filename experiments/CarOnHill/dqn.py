import torch
from experiments.base.load_parameters import load_parameters
from slimRL.environments.car_on_hill import CarOnHill
from slimRL.sample_collection.replay_buffer import ReplayBuffer
from slimRL.networks.architectures.dqn import BasicDQN
from experiments.base.dqn import train

def run(param_file):
    p = load_parameters(param_file, "car_on_hill", "dqn")
    device = torch.device("cuda" if torch.cuda.is_available() and p["use_gpu"] else "cpu")
    env = CarOnHill()
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
    param_file = "./car_on_hill_dqn.json"
    run(param_file)
