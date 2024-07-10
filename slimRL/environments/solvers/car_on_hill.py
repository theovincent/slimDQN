# Credits: https://github.com/theovincent/PBO


import numpy as np
import multiprocess
from slimRL.environments.car_on_hill import CarOnHill


def optimal_steps_to_absorbing(env: CarOnHill, state: np.ndarray, max_steps: int):
    current_states = [state]
    step = 0

    while len(current_states) > 0 and step < max_steps:
        next_states = []
        for state_ in current_states:
            for action in range(2):
                env.reset(state_)
                next_state, reward, _ = env.step(action)
                if reward == 1:
                    return True, step + 1
                elif reward == 0:
                    next_states.append(next_state)
                ## if reward == -1 we pass

        step += 1
        current_states = next_states
        current_states = sorted(current_states, key=lambda x: x[0], reverse=True)

    return False, step


def compute_optimal_q_value(eval_state, idx_state_x, idx_state_v, action, horizon, gamma, optimal_q):
    env = CarOnHill()
    env.reset(eval_state)
    next_state, reward, absorbing = env.step(action)

    if absorbing:
        optimal_q[(idx_state_x, idx_state_v, action)] = reward
    else:
        success, steps_to_absorbing = optimal_steps_to_absorbing(env, next_state, horizon - 1)

        optimal_v_next_state = gamma ** (steps_to_absorbing) if success else -(gamma**steps_to_absorbing)

        optimal_q[(idx_state_x, idx_state_v, action)] = reward + gamma * optimal_v_next_state


def compute_optimal_values(n_states_x, n_states_v, horizon, gamma):
    env = CarOnHill()
    states_x = np.linspace(-env.max_pos, env.max_pos, n_states_x)
    states_v = np.linspace(-env.max_velocity, env.max_velocity, n_states_v)

    manager = multiprocess.Manager()
    optimal_q = np.zeros((n_states_x, n_states_v, 2))
    optimal_v = np.zeros((n_states_x, n_states_v))

    optimal_q_shared = manager.dict()

    processes = []
    for idx_state_x, state_x in enumerate(states_x):
        for idx_state_v, state_v in enumerate(states_v):
            for action in range(2):
                eval_state = np.array([state_x, state_v])
                processes.append(
                    multiprocess.Process(
                        target=compute_optimal_q_value,
                        args=(
                            eval_state,
                            idx_state_x,
                            idx_state_v,
                            action,
                            horizon,
                            gamma,
                            optimal_q_shared,
                        ),
                    )
                )

    for process in processes:
        process.start()

    for process in processes:
        process.join()

    for idx_state_x, state_x in enumerate(states_x):
        for idx_state_v, state_v in enumerate(states_v):
            for action in range(2):
                optimal_q[idx_state_x, idx_state_v, action] = optimal_q_shared[(idx_state_x, idx_state_v, action)]
    optimal_v = np.max(optimal_q, axis=2)
    return optimal_v, optimal_q
