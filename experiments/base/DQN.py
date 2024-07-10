from tqdm import tqdm
import jax
import optax
from slimRL.networks.DQN import DQN
from slimRL.sample_collection.replay_buffer import ReplayBuffer
from slimRL.sample_collection.utils import collect_single_sample
from experiments.base.logger import save_logs


def train(
    key: jax.random.PRNGKey,
    p: dict,
    agent: DQN,
    env,
    rb: ReplayBuffer,
):
    epsilon_schedule = optax.linear_schedule(1.0, p["end_epsilon"], p["duration_epsilon"])

    n_training_steps = 0
    env.reset()
    log_rewards = []
    log_lengths = []

    for idx_epoch in tqdm(range(p["n_epochs"])):
        epoch_rewards = []
        epoch_episode_lengths = []
        episode_reward = 0
        episode_length = 0
        idx_training_step = 0
        has_reset = False

        while idx_training_step < p["n_training_steps_per_epoch"] or not has_reset:
            key, exploration_key = jax.random.split(key)
            reward, has_reset = collect_single_sample(
                exploration_key, env, agent, rb, p, epsilon_schedule, n_training_steps
            )

            episode_reward += reward
            episode_length += 1
            if has_reset:
                epoch_rewards.append(episode_reward)
                epoch_episode_lengths.append(episode_length)
                episode_reward = 0
                episode_length = 0

            if n_training_steps > p["n_initial_samples"]:
                agent.update_online_params(n_training_steps, p["batch_size"], rb)
                agent.update_target_params(n_training_steps)

            idx_training_step += 1
            n_training_steps += 1

        log_rewards.append(epoch_rewards)
        log_lengths.append(epoch_episode_lengths)

        print(
            f"Epoch: {idx_epoch}, Avg. return = {sum(epoch_rewards)/len(epoch_rewards)}, Num episodes = {len(epoch_rewards)}"
        )

    save_logs(p, log_rewards, log_lengths, agent)
