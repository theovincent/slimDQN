# Credits: https://github.com/theovincent/PBO


from tqdm import tqdm
import numpy as np
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

    return False, step


def compute_optimal_values(n_states_x, n_states_v, horizon, gamma):

    env = CarOnHill()
    states_x = np.linspace(-env.max_pos, env.max_pos, n_states_x)
    states_v = np.linspace(-env.max_velocity, env.max_velocity, n_states_v)

    optimal_v = np.zeros((n_states_x, n_states_v))
    optimal_q = np.zeros((n_states_x, n_states_v, 2))

    for idx_state_x, state_x in tqdm(enumerate(states_x)):
        for idx_state_v, state_v in tqdm(enumerate(states_v), leave=False):
            for action in range(2):
                eval_state = np.array([state_x, state_v])
                env.reset(eval_state)
                next_state, reward, absorbing = env.step(action)

                if absorbing:
                    optimal_q[idx_state_x, idx_state_v, action] = reward
                else:
                    success, step_to_absorbing = optimal_steps_to_absorbing(
                        env, next_state, horizon
                    )
                    if step_to_absorbing == 0:
                        optimal_v_next_state = 0
                    else:
                        optimal_v_next_state = (
                            gamma ** (step_to_absorbing - 1)
                            if success
                            else -(gamma ** (step_to_absorbing - 1))
                        )

                    optimal_q[idx_state_x, idx_state_v, action] = (
                        reward + gamma * optimal_v_next_state
                    )
                print(f"Done with {state_x}, {state_v}, {action}")
            optimal_v[idx_state_x, idx_state_v] = np.max(
                optimal_q[idx_state_x, idx_state_v]
            )
    return optimal_v, optimal_q
