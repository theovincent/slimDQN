import random
from slimRL.sample_collection.schedules import linear_schedule
from slimRL.sample_collection.replay_buffer import ReplayBuffer


def collect_single_sample(env, agent, rb: ReplayBuffer, p, n_training_steps: int):
    epsilon = linear_schedule(
        p["end_epsilon"],
        p["duration_epsilon"],
        n_training_steps,
    )
    if random.random() < epsilon:
        action = random.sample(range(env.n_actions), 1)[0]
    else:
        action = agent.best_action(env.state)

    obs = env.state.copy()
    _, reward, termination = env.step(action)
    truncation = False
    if env.n_steps == p["horizon"]:
        truncation = True
    has_reset = termination or truncation
    rb.add(obs, action, reward, termination, truncation)

    if has_reset:
        env.reset()

    return reward, has_reset
