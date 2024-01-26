import time
import random
import numpy as np
import torch
import json
import matplotlib.pyplot as plt
from slimRL.environments.chain import generate_simple_chain
from slimRL.networks.architectures.dqn import BasicDQN
from slimRL.rl_utils.replay_buffer import ReplayBuffer

from slimRL.rl_utils.misc import linear_schedule


def train(**kwargs):
    env_id = "chain"
    agent_type = "dqn"
    seed = kwargs.get("seed", 42)
    use_gpu = kwargs.get("use_gpu", True)
    gamma = kwargs.get("gamma", 0.99)
    tau = kwargs.get("tau", 1.0)
    lr = kwargs.get("lr", 2.5e-4)
    loss_type = kwargs.get("loss_type", 'huber')
    batch_size = kwargs.get("batch_size", 32)
    train_frequency = kwargs.get("train_frequency", 10)
    target_update_frequency = kwargs.get("target_update_frequency", 500)
    save_model = kwargs.get("save_mode", False)
    replay_capacity = kwargs.get("replay_capacity", 10000)
    update_horizon = kwargs.get("update_horizon", 1)
    total_timesteps = kwargs.get("total_timesteps", 500000)
    start_epsilon = kwargs.get("start_epsilon", 1.0)
    end_epsilon = kwargs.get("end_epsilon", 0.05)
    exploration_fraction = kwargs.get("exploration_fraction", 0.2)
    learning_starts = kwargs.get("learning_starts", 1000)
    run_name = f"{env_id}__{agent_type}__{seed}__{int(time.time())}"

    print(run_name)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.backends.cudnn.deterministic = torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")

    # env setup
    env = generate_simple_chain(4, [0], 1.0, 1)
    agent = BasicDQN(env,
                     device=device,
                     gamma=gamma,
                     tau=tau,
                     lr=lr,
                     loss_type=loss_type,
                     train_frequency=train_frequency,
                     target_update_frequency=target_update_frequency,
                     save_model=save_model,
                     )

    rb = ReplayBuffer(observation_shape=env.observation_shape,
                      replay_capacity=replay_capacity,
                      batch_size=batch_size,
                      update_horizon=update_horizon,
                      gamma=gamma,
                      )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = env.reset()
    all_episodes_rew = []
    curr_episode_rew = 0
    for global_step in range(total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(start_epsilon, end_epsilon, int(exploration_fraction * total_timesteps),
                                  global_step)
        if random.random() < epsilon:
            action = random.sample(env.single_action_space, 1)[0]
        else:
            action = agent.best_action(obs)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, reward, termination, _ = env.step(action)

        curr_episode_rew += reward

        if termination:
            print(f"Step = {global_step}, Reward = {curr_episode_rew}")
            all_episodes_rew.append(curr_episode_rew)
            curr_episode_rew = 0
            next_obs, _ = env.reset()

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        rb.add(obs, action, reward, termination)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > learning_starts:
            agent.update_online_params(global_step, rb)
            agent.update_target_params(global_step)

    return all_episodes_rew


if __name__ == "__main__":
    hyperparams_file = "../hyperparams/default.json"
    params = json.load(open(hyperparams_file, "r"))
    rew_list = train(**params)
    plt.plot(rew_list)
    plt.show()
