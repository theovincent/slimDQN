# Reference: https://github.com/MushroomRL/mushroom-rl.git

import numpy as np
from slimRL.environments.finite_mdp import FiniteMDP

def generate_simple_chain(state_n, goal_states, prob, rew, mu=None,
                          horizon=100):
    """
    Simple chain generator.
    """
    p = compute_probabilities(state_n, prob, goal_states)
    r = compute_reward(state_n, goal_states, rew)
    print(p.shape,'\n', p)
    print(r.shape, '\n', r)

    assert mu is None or len(mu) == state_n

    return FiniteMDP(p, r, mu, horizon)


def compute_probabilities(state_n, prob, goal_states):
    """
    Compute the transition probability matrix.
    0 = right, 1 = left
    """
    p = np.zeros((state_n, 2, state_n))

    for i in range(state_n):
        if i == 0:
            p[i, 1, i] = 1.
        else:
            p[i, 1, i] = 1. - prob
            p[i, 1, i - 1] = prob

        if i == state_n - 1:
            p[i, 0, i] = 1.
        else:
            p[i, 0, i] = 1. - prob
            p[i, 0, i + 1] = prob

    for g in goal_states:
        p[g, :, :] = 0

    return p


def compute_reward(state_n, goal_states, rew):
    r = np.zeros((state_n, 2, state_n))

    for g in goal_states:
        r[g, :, g] = rew
        if g != 0:
            r[g - 1, 0, g] = rew

        if g != state_n - 1:
            r[g + 1, 1, g] = rew

    return r
