# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from rl_utils import ReplayBuffer

from slimRL.environments.environment import Environment


class DQN():
    def __init__(
            self,
            gamma: float,
            tau: float,
            target_network: nn.Module,
            q_network: nn.Module,
            optimizer: optim.Optimizer,
            loss_type: str,
            train_frequency: int,
            target_update_frequency: int,
            save_model: bool,
    ):

        self.gamma = gamma
        self.tau = tau
        self.target_network = target_network
        self.q_network = q_network
        self.optimizer = optimizer,
        self.loss_type = loss_type
        self.train_frequency = train_frequency
        self.target_update_frequency = target_update_frequency
        self.save_model = save_model

    def loss_on_batch(self, td_target, estimate):
        return self.loss(self.loss_type)(td_target, estimate)

    @staticmethod
    def loss(order) -> nn.modules.loss:
        if order == "huber":
            return nn.HuberLoss()
        elif order == "1":
            return nn.L1Loss()
        elif order == "2":
            return nn.MSELoss()

    def learn_on_batch(
            self,
            batch_samples
    ):
        target = self.compute_target(batch_samples)
        curr_estimate = self.compute_qval(batch_samples)
        loss = self.loss_on_batch(target, curr_estimate)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def compute_target(self, data):
        with torch.no_grad():
            target_max, _ = self.compute_qval(self.target_network, data.next_observations).max(dim=1)
            td_target = data.rewards.flatten() + self.gamma * target_max * (1 - data.dones.flatten())
        return td_target

    def compute_curr_estimate(self, data):
        q_val = self.compute_qval(self.q_network, data.observations).gather(1, data.actions).squeeze()
        return q_val

    @staticmethod
    def compute_qval(network, states):
        q_val = network(states)
        return q_val

    def update_online_params(self, step: int, replay_buffer: ReplayBuffer):
        if step % self.train_frequency == 0:
            batch_samples = replay_buffer.sample_transition_batch()

            loss = self.learn_on_batch(
                batch_samples
            )
            return loss
        return None

    def update_target_params(self, step: int):
        if step % self.target_update_frequency == 0:
            for target_network_param, q_network_param in zip(self.target_network.parameters(),
                                                             self.q_network.parameters()):
                target_network_param.data.copy_(
                    self.tau * q_network_param.data + (1.0 - self.tau) * target_network_param.data
                )

    def best_action(self,
                    state: torch.tensor):
        with torch.no_grad():
            action = torch.argmax(self.compute_qval(self.q_network, torch.Tensor(state)), dim=1).cpu().numpy()
        return action


'''
class DQN(nn.Module):
    def __init__(self,
                 env,--->only state and action dim needed
                 total_timesteps=500000,-->training code
                 lr=2.5e-4,---> done
                 buffer_size=10000,--->rb code
                 gamma=0.99,--->done
                 tau=1.0,--->done
                 target_network_frequency=500,--->done
                 batch_size=128,--->training code
                 start_epsilon=1,--->
                 end_epsilon=0.05,
                 exploration_fraction=0.5,
                 learning_starts=10000,
                 train_frequency=10,
                 seed=42,
                 cuda=True,
                 save_model=True,
                 ):
        super().__init__()
        self.network = DQN

    def forward(self, x):
        return self.network(x)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:

poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )
    args = tyro.cli(Args)
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    q_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps,
                                  global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = q_network(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                with torch.no_grad():
                    target_max, _ = target_network(data.next_observations).max(dim=1)
                    td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
                old_val = q_network(data.observations).gather(1, data.actions).squeeze()
                # loss = F.mse_loss(td_target, old_val)
                # 
                # if global_step % 100 == 0:
                #     writer.add_scalar("losses/td_loss", loss, global_step)
                #     writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                #     print("SPS:", int(global_step / (time.time() - start_time)))
                #     writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                # 
                # # optimize the model
                # optimizer.zero_grad()
                # loss.backward()
                # optimizer.step()

            # update target network
            if global_step % args.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                    )

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(q_network.state_dict(), model_path)
        print(f"model saved to {model_path}")
        from cleanrl_utils.evals.dqn_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=QNetwork,
            device=device,
            epsilon=0.05,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(args, episodic_returns, repo_id, "DQN", f"runs/{run_name}", f"videos/{run_name}-eval")

    envs.close()
    writer.close()
'''
