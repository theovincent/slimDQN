import random
from slimRL.sample_collection.schedules import linear_schedule
from slimRL.sample_collection.replay_buffer import ReplayBuffer


def collect_single_sample(env, agent, rb: ReplayBuffer, p, n_training_steps: int):
    epsilon = linear_schedule(
        p.get("end_epsilon", 1),  # default values are to handle FQI and keep epsilon=1
        p.get("duration_epsilon", -1),
        n_training_steps,
    )
    if random.random() < epsilon:
        action = random.randint(0, env.n_actions - 1)
    else:
        action = agent.best_action(env.state)

    obs = env.state.copy()
    _, reward, termination = env.step(action)
    truncation = env.n_steps == p["horizon"]
    rb.add(obs, action, reward, termination, truncation)

    has_reset = termination or truncation
    if has_reset:
        env.reset()

    return reward, has_reset
